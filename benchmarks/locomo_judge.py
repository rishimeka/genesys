"""Judge LoCoMo eval results using LLM-as-Judge (Claude Haiku).

Reads eval results, judges each answer as CORRECT/WRONG, reports J-scores per category.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"
RESULTS_FILE = Path(__file__).parent / "locomo_eval_results.json"

CATEGORY_NAMES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
    5: "Adversarial",
}

JUDGE_PROMPT = """You will be given the following data:
(1) a question (posed by one user to another user)
(2) a 'gold' (ground truth) answer
(3) a generated answer which you will score as CORRECT/WRONG

The gold answer will usually be a concise and short answer that includes the referenced topic. Be generous with grading — as long as the generated answer touches on the same topic as the gold answer, it should be counted as CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

Respond with a JSON object with two keys: "reasoning" (brief explanation) and "label" (either "CORRECT" or "WRONG")."""

# Limit concurrency to avoid rate limits
SEMAPHORE_LIMIT = 10


async def judge_one(
    sem: asyncio.Semaphore,
    client: anthropic.AsyncAnthropic,
    result: dict,
) -> dict:
    """Judge a single QA result."""
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=result["question"],
            gold_answer=result["gold_answer"],
            predicted_answer=result["predicted_answer"],
        )

        for attempt in range(5):
            try:
                resp = await client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                try:
                    parsed = json.loads(raw)
                    verdict = parsed.get("label", "WRONG").upper()
                except json.JSONDecodeError:
                    verdict = "CORRECT" if "CORRECT" in raw.upper() and "WRONG" not in raw.upper() else "WRONG"
                is_correct = verdict == "CORRECT"
                break
            except Exception as e:
                if "rate" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                print(f"  JUDGE ERROR: {e}")
                verdict = "ERROR"
                is_correct = False
                break

        return {
            **result,
            "verdict": verdict,
            "is_correct": is_correct,
        }


async def main():
    parser = argparse.ArgumentParser(description="Judge LoCoMo eval results")
    parser.add_argument("--input", type=str, default=None, help="Eval results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Judged results output file")
    parser.add_argument("--concurrency", type=int, default=SEMAPHORE_LIMIT, help="Max concurrent judge calls")
    args = parser.parse_args()

    input_path = args.input or str(RESULTS_FILE)
    with open(input_path) as f:
        results = json.load(f)

    print(f"Judging {len(results)} answers with {JUDGE_MODEL} (concurrency={args.concurrency})")
    start_time = time.time()

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    # Process in batches with progress reporting
    judged = []
    batch_size = args.concurrency
    for batch_start in range(0, len(results), batch_size):
        batch = results[batch_start:batch_start + batch_size]
        batch_tasks = [judge_one(sem, client, r) for r in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        judged.extend(batch_results)
        done = batch_start + len(batch)
        if done % 50 == 0 or done == len(results):
            elapsed = time.time() - start_time
            print(f"  Judged {done}/{len(results)} ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"Judging complete in {elapsed:.0f}s")

    # Compute J-scores per category
    cat_correct: dict[int, int] = {}
    cat_total: dict[int, int] = {}

    # Per-conversation tracking
    conv_correct: dict[str, int] = {}
    conv_total: dict[str, int] = {}

    for r in judged:
        cat = r["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if r["is_correct"]:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

        sid = str(r.get("sample_id", "unknown"))
        conv_total[sid] = conv_total.get(sid, 0) + 1
        if r["is_correct"]:
            conv_correct[sid] = conv_correct.get(sid, 0) + 1

    # Print report
    print("\n" + "=" * 60)
    print("LoCoMo J-Score Results (Genesys)")
    print("=" * 60)
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'J-Score':>10}")
    print("-" * 60)

    overall_correct = 0
    overall_total = 0

    for cat in sorted(cat_total.keys()):
        correct = cat_correct.get(cat, 0)
        total = cat_total[cat]
        score = correct / total * 100 if total > 0 else 0
        name = CATEGORY_NAMES.get(cat, f"Category {cat}")
        print(f"{name:<20} {correct:>8} {total:>8} {score:>9.1f}%")
        overall_correct += correct
        overall_total += total

    overall_score = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_correct:>8} {overall_total:>8} {overall_score:>9.1f}%")
    print("=" * 60)

    # Per-conversation breakdown
    print(f"\n{'Per-Conversation Breakdown':^60}")
    print("=" * 60)
    print(f"{'Conversation':<20} {'Correct':>8} {'Total':>8} {'J-Score':>10}")
    print("-" * 60)
    for sid in sorted(conv_total.keys()):
        correct = conv_correct.get(sid, 0)
        total = conv_total[sid]
        score = correct / total * 100 if total > 0 else 0
        print(f"{'conv-' + sid:<20} {correct:>8} {total:>8} {score:>9.1f}%")
    print("=" * 60)

    print(f"\nAnswer model: gpt-4o-mini")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Retrieval k: inferred from results")

    # Save judged results
    out_path = args.output or str(Path(__file__).parent / "locomo_judged.json")
    per_conversation = {
        f"conv-{sid}": {
            "correct": conv_correct.get(sid, 0),
            "total": conv_total[sid],
            "j_score": round(conv_correct.get(sid, 0) / conv_total[sid] * 100, 1) if conv_total[sid] > 0 else 0,
        }
        for sid in sorted(conv_total.keys())
    }
    output_data = {
        "metadata": {
            "answer_model": "gpt-4o-mini",
            "judge_model": JUDGE_MODEL,
            "total_questions": overall_total,
            "overall_j_score": round(overall_score, 1),
            "per_category": {
                CATEGORY_NAMES.get(cat, f"cat_{cat}"): {
                    "correct": cat_correct.get(cat, 0),
                    "total": cat_total[cat],
                    "j_score": round(cat_correct.get(cat, 0) / cat_total[cat] * 100, 1) if cat_total[cat] > 0 else 0,
                }
                for cat in sorted(cat_total.keys())
            },
            "per_conversation": per_conversation,
        },
        "results": judged,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
