"""Run LoCoMo QA evaluation against Genesys.

For each QA pair: recall memories via Genesys, generate answer with gpt-4o-mini, save results.
Each conversation is queried in its own user scope (locomo-{sample_id}) to prevent cross-contamination.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import httpx
import openai

DEFAULT_API = "http://localhost:8000"
DATA_FILE = Path(__file__).parent / "locomo10.json"
ANSWER_MODEL = "gpt-4o-mini"

# gpt-4o-mini: 500 RPM on Tier 1, much higher on Tier 2+. Use 400 for headroom.
RPM_LIMIT = 400

# Category 5 (adversarial) has known broken ground truth — exclude by default
EXCLUDE_CATEGORIES = {5}

ANSWER_PROMPT = """Below is context retrieved from a conversation between two people. The date of each memory is written in brackets.

{context}

Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. Use DATE of CONVERSATION to answer with an approximate date.

Question: {question} Short answer:"""


async def recall_memories(client: httpx.AsyncClient, api_url: str, query: str, user_id: str, k: int = 10) -> list[dict]:
    """Call Genesys memory_recall via REST API, scoped to a specific user."""
    try:
        resp = await client.post(
            f"{api_url}/api/recall",
            json={"query": query, "k": k, "read_only": True},
            headers={"X-User-Id": user_id},
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("memories", result.get("results", []))
    except Exception as e:
        print(f"  RECALL ERROR: {e}")
        return []


async def generate_answer(
    oai_client: openai.AsyncOpenAI,
    question: str,
    memories: list[dict],
) -> str:
    """Generate answer using gpt-4o-mini given retrieved context."""
    context_parts = []
    for m in memories:
        content = m.get("content_full") or m.get("content_summary") or m.get("content", "")
        created = m.get("created_at", "")
        if created:
            context_parts.append(f"- [{created}] {content}")
        else:
            context_parts.append(f"- {content}")

    context = "\n".join(context_parts) if context_parts else "(no memories retrieved)"

    prompt = ANSWER_PROMPT.format(context=context, question=question)

    try:
        resp = await oai_client.chat.completions.create(
            model=ANSWER_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ANSWER ERROR: {e}")
        return "(error generating answer)"


async def main():
    parser = argparse.ArgumentParser(description="Run LoCoMo eval against Genesys")
    parser.add_argument("--api", default=DEFAULT_API, help="Genesys API URL")
    parser.add_argument("--samples", type=str, default="all", help="Comma-separated sample indices or 'all'")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--k", type=int, default=20, help="Number of memories to recall per question")
    parser.add_argument("--include-cat5", action="store_true", help="Include category 5 (adversarial)")
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)

    if args.samples == "all":
        indices = list(range(len(data)))
    else:
        indices = [int(x) for x in args.samples.split(",")]

    exclude = set() if args.include_cat5 else EXCLUDE_CATEGORIES

    oai_client = openai.AsyncOpenAI()
    results = []
    total = 0
    skipped = 0

    # Collect all QA items with their user_id for isolation
    all_qa = []
    for idx in indices:
        sample = data[idx]
        sample_id = sample.get("sample_id", idx)
        user_id = f"locomo-{sample_id}"
        for qi, qa in enumerate(sample["qa"]):
            if qa["category"] not in exclude:
                all_qa.append((sample_id, user_id, qi, qa))
            else:
                skipped += 1

    print(f"Running LoCoMo eval: {len(all_qa)} questions (skipped {skipped} cat-5), k={args.k}, model={ANSWER_MODEL}")
    print(f"Per-conversation user isolation enabled")
    print(f"Rate limit: {RPM_LIMIT} RPM → estimated {len(all_qa) / RPM_LIMIT:.0f} minutes")

    start_time = time.time()
    sem = asyncio.Semaphore(RPM_LIMIT)

    async def process_one(http_client, sample_id, user_id, qi, qa):
        async with sem:
            question = qa["question"]
            gold_answer = qa["answer"]
            evidence = qa.get("evidence", [])

            try:
                memories = await recall_memories(http_client, args.api, question, user_id, k=args.k)
                predicted = await generate_answer(oai_client, question, memories)
            except Exception as e:
                print(f"  ERROR on q{qi} ({sample_id}): {e}")
                memories = []
                predicted = "(error)"

            return {
                "sample_id": sample_id,
                "question_idx": qi,
                "category": qa["category"],
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted,
                "evidence": evidence,
                "num_memories_retrieved": len(memories),
            }

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            process_one(client, sid, uid, qi, qa)
            for sid, uid, qi, qa in all_qa
        ]

        # Process in chunks and report progress
        chunk_size = 50
        for chunk_start in range(0, len(tasks), chunk_size):
            chunk = tasks[chunk_start:chunk_start + chunk_size]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            total += len(chunk_results)
            elapsed = time.time() - start_time
            print(f"  Processed {total}/{len(all_qa)} questions ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - start_time
    print(f"\nDone: {total} questions answered, {skipped} skipped (cat 5), {elapsed:.0f}s total")

    # Save results
    out_path = args.output or str(Path(__file__).parent / "locomo_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
