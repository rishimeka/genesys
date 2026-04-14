"""LoCoMo full pipeline: per-conversation independent runs.

For each conversation:
  1. Clear entire database
  2. Ingest that conversation's memories
  3. Run that conversation's eval questions
  4. Collect results

Then aggregate and judge all results at the end.
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
JUDGE_MODEL = "gpt-4o-mini"
EXCLUDE_CATEGORIES = {5}

ANSWER_PROMPT = """Below is context retrieved from a conversation between two people. The date of each memory is written in brackets.

{context}

Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible. Use DATE of CONVERSATION to answer with an approximate date.

Question: {question} Short answer:"""

JUDGE_PROMPT = """You will be given the following data:
(1) a question (posed by one user to another user)
(2) a 'gold' (ground truth) answer
(3) a generated answer which you will score as CORRECT/WRONG

The gold answer will usually be a concise and short answer that includes the referenced topic. Be generous with grading — as long as the generated answer touches on the same topic as the gold answer, it should be counted as CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

Respond with a JSON object with two keys: "reasoning" (brief explanation) and "label" (either "CORRECT" or "WRONG")."""


def parse_locomo_datetime(dt_str: str) -> str:
    from datetime import datetime
    dt_str = dt_str.strip()
    try:
        dt = datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        try:
            dt = datetime.strptime(dt_str, "%I:%M %p on %d %B %Y")
        except ValueError:
            return "2023-01-01T00:00:00+00:00"
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


async def clear_all(client: httpx.AsyncClient, api_url: str, user_id: str):
    """Clear all memories for a user scope."""
    resp = await client.post(
        f"{api_url}/api/admin/clear-user",
        headers={"X-User-Id": user_id},
    )
    resp.raise_for_status()


async def ingest_conversation(client: httpx.AsyncClient, sample: dict, api_url: str, user_id: str) -> int:
    """Ingest one conversation with one memory per session."""
    conv = sample["conversation"]
    headers = {"X-User-Id": user_id}

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    # Build one memory per session: concatenate all turns with speaker labels
    sem = asyncio.Semaphore(20)

    async def store_one(payload):
        async with sem:
            resp = await client.post(f"{api_url}/api/memories", json=payload, headers=headers)
            resp.raise_for_status()
            return 1

    payloads = []
    for session_key in session_keys:
        session_num = int(session_key.split("_")[1])
        dt_key = f"{session_key}_date_time"
        timestamp = parse_locomo_datetime(conv.get(dt_key, ""))
        turns = conv[session_key]
        lines = [f"{turn['speaker']}: {turn['text']}" for turn in turns]
        content = "\n".join(lines)
        payloads.append({
            "content": content,
            "source_session": f"locomo_session_{session_num}",
            "created_at": timestamp,
        })

    results = await asyncio.gather(*[store_one(p) for p in payloads], return_exceptions=True)
    return sum(1 for r in results if r == 1)


async def recall_memories(client: httpx.AsyncClient, api_url: str, query: str, user_id: str, k: int) -> list[dict]:
    resp = await client.post(
        f"{api_url}/api/recall",
        json={"query": query, "k": k, "read_only": True},
        headers={"X-User-Id": user_id},
    )
    resp.raise_for_status()
    result = resp.json()
    return result.get("memories", result.get("results", []))


async def generate_answer(oai: openai.AsyncOpenAI, question: str, memories: list[dict]) -> str:
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

    resp = await oai.chat.completions.create(
        model=ANSWER_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


async def judge_one(oai: openai.AsyncOpenAI, sem: asyncio.Semaphore, result: dict) -> dict:
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=result["question"],
            gold_answer=result["gold_answer"],
            predicted_answer=result["predicted_answer"],
        )
        for attempt in range(5):
            try:
                resp = await oai.chat.completions.create(
                    model=JUDGE_MODEL, max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                try:
                    parsed = json.loads(raw)
                    verdict = parsed.get("label", "WRONG").upper()
                except json.JSONDecodeError:
                    verdict = "CORRECT" if "CORRECT" in raw.upper() and "WRONG" not in raw.upper() else "WRONG"
                return {**result, "verdict": verdict, "is_correct": verdict == "CORRECT"}
            except Exception as e:
                if "rate" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                print(f"  JUDGE ERROR: {e}")
                return {**result, "verdict": "ERROR", "is_correct": False}


async def main():
    parser = argparse.ArgumentParser(description="LoCoMo full pipeline")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--samples", type=str, default="all")
    parser.add_argument("--include-cat5", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)

    if args.samples == "all":
        indices = list(range(len(data)))
    else:
        indices = [int(x) for x in args.samples.split(",")]

    exclude = set() if args.include_cat5 else EXCLUDE_CATEGORIES

    oai = openai.AsyncOpenAI()
    all_results = []
    start = time.time()

    for idx in indices:
        sample = data[idx]
        sample_id = sample.get("sample_id", idx)
        # sample_id is like "conv-26" — use as-is for display, prefix with "locomo-" for user isolation
        user_id = f"locomo-{sample_id}"

        print(f"\n{'='*60}")
        print(f"CONVERSATION: {sample_id}")
        print(f"{'='*60}")

        # Step 1: Clear database for this user
        async with httpx.AsyncClient(timeout=60.0) as client:
            await clear_all(client, args.api, user_id)
            print(f"  Cleared user scope: {user_id}")

            # Step 2: Ingest
            t0 = time.time()
            n_stored = await ingest_conversation(client, sample, args.api, user_id)
            print(f"  Ingested {n_stored} memories ({time.time()-t0:.0f}s)")

            # Step 3: Eval questions
            qa_items = [qa for qa in sample["qa"] if qa["category"] not in exclude]
            skipped = len(sample["qa"]) - len(qa_items)
            print(f"  Evaluating {len(qa_items)} questions (skipped {skipped} cat-5)")

            conv_results = []
            for qi, qa in enumerate(qa_items):
                try:
                    memories = await recall_memories(client, args.api, qa["question"], user_id, k=args.k)
                    predicted = await generate_answer(oai, qa["question"], memories)
                except Exception as e:
                    print(f"    ERROR q{qi}: {e}")
                    memories = []
                    predicted = "(error)"

                conv_results.append({
                    "sample_id": sample_id,
                    "question_idx": qi,
                    "category": qa["category"],
                    "question": qa["question"],
                    "gold_answer": qa["answer"],
                    "predicted_answer": predicted,
                    "evidence": qa.get("evidence", []),
                    "num_memories_retrieved": len(memories),
                })

            all_results.extend(conv_results)
            elapsed = time.time() - start
            print(f"  Done: {len(conv_results)} answers ({elapsed:.0f}s total)")

    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE: {len(all_results)} answers in {time.time()-start:.0f}s")
    print(f"{'='*60}")

    # Step 4: Judge all results
    print(f"\nJudging {len(all_results)} answers with {JUDGE_MODEL}...")
    judge_sem = asyncio.Semaphore(20)
    judge_start = time.time()

    judged = []
    batch_size = 50
    for batch_start in range(0, len(all_results), batch_size):
        batch = all_results[batch_start:batch_start + batch_size]
        batch_results = await asyncio.gather(*[judge_one(oai, judge_sem, r) for r in batch])
        judged.extend(batch_results)
        done = batch_start + len(batch)
        if done % 100 == 0 or done == len(all_results):
            print(f"  Judged {done}/{len(all_results)} ({time.time()-judge_start:.0f}s)")

    # Step 5: Report
    CATEGORY_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain", 5: "Adversarial"}
    cat_correct: dict[int, int] = {}
    cat_total: dict[int, int] = {}
    conv_correct: dict[str, int] = {}
    conv_total: dict[str, int] = {}

    for r in judged:
        cat = r["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if r["is_correct"]:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

        sid = str(r["sample_id"])
        conv_total[sid] = conv_total.get(sid, 0) + 1
        if r["is_correct"]:
            conv_correct[sid] = conv_correct.get(sid, 0) + 1

    print(f"\n{'='*60}")
    print("LoCoMo J-Score Results (Genesys)")
    print(f"{'='*60}")
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'J-Score':>10}")
    print("-" * 60)

    overall_correct = 0
    overall_total = 0
    for cat in sorted(cat_total.keys()):
        c = cat_correct.get(cat, 0)
        t = cat_total[cat]
        s = c / t * 100 if t > 0 else 0
        print(f"{CATEGORY_NAMES.get(cat, f'Cat {cat}'):<20} {c:>8} {t:>8} {s:>9.1f}%")
        overall_correct += c
        overall_total += t

    overall_score = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_correct:>8} {overall_total:>8} {overall_score:>9.1f}%")
    print("=" * 60)

    print(f"\n{'Per-Conversation Breakdown':^60}")
    print("=" * 60)
    print(f"{'Conversation':<20} {'Correct':>8} {'Total':>8} {'J-Score':>10}")
    print("-" * 60)
    for sid in sorted(conv_total.keys()):
        c = conv_correct.get(sid, 0)
        t = conv_total[sid]
        s = c / t * 100 if t > 0 else 0
        print(f"{sid:<20} {c:>8} {t:>8} {s:>9.1f}%")
    print("=" * 60)

    print(f"\nAnswer model: {ANSWER_MODEL}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Retrieval k: {args.k}")

    # Save
    out_path = args.output or str(Path(__file__).parent / "locomo_judged.json")
    per_conv = {
        sid: {
            "correct": conv_correct.get(sid, 0),
            "total": conv_total[sid],
            "j_score": round(conv_correct.get(sid, 0) / conv_total[sid] * 100, 1),
        }
        for sid in sorted(conv_total.keys())
    }
    output_data = {
        "metadata": {
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
            "retrieval_k": args.k,
            "total_questions": overall_total,
            "overall_j_score": round(overall_score, 1),
            "per_category": {
                CATEGORY_NAMES.get(cat, f"cat_{cat}"): {
                    "correct": cat_correct.get(cat, 0),
                    "total": cat_total[cat],
                    "j_score": round(cat_correct.get(cat, 0) / cat_total[cat] * 100, 1),
                }
                for cat in sorted(cat_total.keys())
            },
            "per_conversation": per_conv,
        },
        "results": judged,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
