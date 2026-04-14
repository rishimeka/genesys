"""Ingest LoCoMo conversations into Genesys via the REST API.

Each conversation is isolated into its own user scope (locomo-{sample_id})
so retrieval doesn't cross-contaminate between conversations.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

DEFAULT_API = "http://localhost:8000"
DATA_FILE = Path(__file__).parent / "locomo10.json"


def parse_locomo_datetime(dt_str: str) -> str:
    """Convert LoCoMo datetime strings to ISO 8601."""
    dt_str = dt_str.strip()
    try:
        dt = datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        try:
            dt = datetime.strptime(dt_str, "%I:%M %p on %d %B %Y")
        except ValueError:
            print(f"  WARNING: Could not parse datetime '{dt_str}', using epoch")
            return "2023-01-01T00:00:00+00:00"
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


async def clear_user(client: httpx.AsyncClient, api_url: str, user_id: str):
    """Clear all memories for a user scope before ingestion."""
    try:
        resp = await client.post(
            f"{api_url}/api/admin/clear-user",
            headers={"X-User-Id": user_id},
        )
        resp.raise_for_status()
        print(f"  Cleared existing memories for {user_id}")
    except Exception as e:
        print(f"  WARNING: Could not clear user {user_id}: {e}")


async def ingest_conversation(
    client: httpx.AsyncClient,
    sample: dict,
    sample_idx: int,
    api_url: str,
) -> dict:
    """Ingest one LoCoMo conversation into Genesys with per-conversation user isolation."""
    conv = sample["conversation"]
    sample_id = sample.get("sample_id", sample_idx)
    user_id = f"locomo-{sample_id}"
    headers = {"X-User-Id": user_id}

    # Clear any existing data for this conversation's user scope
    await clear_user(client, api_url, user_id)

    # Collect sessions in order
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    total_stored = 0
    node_ids = []

    for session_key in session_keys:
        session_num = int(session_key.split("_")[1])
        dt_key = f"{session_key}_date_time"
        timestamp = parse_locomo_datetime(conv.get(dt_key, ""))
        turns = conv[session_key]

        for turn in turns:
            speaker = turn["speaker"]
            text = turn["text"]
            dia_id = turn.get("dia_id", "")

            # Include date in content so it gets embedded for temporal queries
            date_str = timestamp[:10] if timestamp else ""
            content = f"[{date_str}] [{speaker}] (session {session_num}): {text}"

            payload = {
                "content": content,
                "source_session": f"locomo_{sample_id}_session_{session_num}",
                "created_at": timestamp,
            }

            if node_ids:
                payload["related_to"] = [node_ids[-1]]

            try:
                resp = await client.post(
                    f"{api_url}/api/memories",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()
                node_id = result.get("node_id") or result.get("id")
                if node_id:
                    node_ids.append(str(node_id))
                total_stored += 1
            except Exception as e:
                print(f"  ERROR storing {dia_id}: {e}")

        print(f"  Session {session_num}: {len(turns)} turns ingested")

    return {"sample_id": sample_id, "user_id": user_id, "turns_stored": total_stored, "node_ids": node_ids}


async def main():
    parser = argparse.ArgumentParser(description="Ingest LoCoMo conversations into Genesys")
    parser.add_argument("--api", default=DEFAULT_API, help="Genesys API URL")
    parser.add_argument("--samples", type=str, default="all", help="Comma-separated sample indices (0-9) or 'all'")
    parser.add_argument("--output", type=str, default=None, help="Output file for ingestion metadata")
    args = parser.parse_args()

    with open(DATA_FILE) as f:
        data = json.load(f)

    if args.samples == "all":
        indices = list(range(len(data)))
    else:
        indices = [int(x) for x in args.samples.split(",")]

    print(f"Ingesting {len(indices)} conversations into {args.api} (per-conversation isolation)")

    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx in indices:
            sample = data[idx]
            sample_id = sample.get("sample_id", idx)
            print(f"\nConversation {idx} (sample_id={sample_id}, user_id=locomo-{sample_id}):")
            result = await ingest_conversation(client, sample, idx, args.api)
            results.append(result)
            print(f"  Total: {result['turns_stored']} memories stored")

    # Save metadata for eval script
    out_path = args.output or str(Path(__file__).parent / "locomo_ingestion.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nIngestion metadata saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
