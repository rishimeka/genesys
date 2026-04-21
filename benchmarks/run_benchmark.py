#!/usr/bin/env python3
"""Benchmark runner: compares Genesys causal memory vs flat vector baseline.

Usage:
    python benchmarks/run_benchmark.py              # Full run (needs API keys)
    python benchmarks/run_benchmark.py --dry-run    # Parse scenarios only, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path (must precede benchmarks imports)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic  # noqa: E402

from benchmarks.baseline_flat import FlatVectorMemory  # noqa: E402
from genesys_memory.engine.scoring import cosine_similarity  # noqa: E402
from genesys_memory.mcp.tools import MCPToolHandler  # noqa: E402
from genesys_memory.models.edge import MemoryEdge  # noqa: E402
from genesys_memory.models.enums import CAUSAL_EDGE_TYPES, EdgeType, MemoryStatus  # noqa: E402
from genesys_memory.models.node import MemoryNode  # noqa: E402
from genesys_memory.retrieval.embedding import OpenAIEmbeddingProvider  # noqa: E402
from genesys_memory.storage.base import LLMProvider  # noqa: E402

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


class InMemoryGraphProvider:
    """Lightweight in-memory graph for benchmarking — no FalkorDB needed."""

    def __init__(self):
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: list[MemoryEdge] = []

    async def initialize(self, user_id: str) -> None:
        pass

    async def destroy(self, user_id: str) -> None:
        self.nodes.clear()
        self.edges.clear()

    async def create_node(self, node: MemoryNode) -> str:
        nid = str(node.id)
        self.nodes[nid] = node
        return nid

    async def get_node(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    async def update_node(self, node_id: str, updates: dict) -> None:
        node = self.nodes.get(node_id)
        if node:
            for k, v in updates.items():
                setattr(node, k, v)

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if str(e.source_id) != node_id and str(e.target_id) != node_id]

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        return [n for n in self.nodes.values() if n.status == status][:limit]

    async def create_edge(self, edge: MemoryEdge) -> str:
        self.edges.append(edge)
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        result = []
        for e in self.edges:
            src, tgt = str(e.source_id), str(e.target_id)
            match = False
            if direction in ("out", "both") and src == node_id:
                match = True
            if direction in ("in", "both") and tgt == node_id:
                match = True
            if match and (edge_type is None or e.type == edge_type):
                result.append(e)
        return result

    async def delete_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if str(e.id) != edge_id]

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        return any(
            str(e.source_id) == source_id and str(e.target_id) == target_id and e.type == edge_type
            for e in self.edges
        )

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None) -> list[MemoryNode]:
        visited = set()
        queue = [(start_id, 0)]
        result = []
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                result.append(node)
            if d < depth:
                for e in self.edges:
                    src, tgt = str(e.source_id), str(e.target_id)
                    if edge_types and e.type not in edge_types:
                        continue
                    if src == nid and tgt not in visited:
                        queue.append((tgt, d + 1))
                    if tgt == nid and src not in visited:
                        queue.append((src, d + 1))
        return result

    async def get_causal_chain(self, node_id: str, direction: str) -> list[MemoryNode]:
        visited = set()
        queue = [node_id]
        result = []
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            for e in self.edges:
                if e.type not in CAUSAL_EDGE_TYPES:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                next_id = None
                if direction == "upstream" and tgt == nid and src not in visited:
                    next_id = src
                elif direction == "downstream" and src == nid and tgt not in visited:
                    next_id = tgt
                if next_id:
                    queue.append(next_id)
                    node = self.nodes.get(next_id)
                    if node:
                        result.append(node)
        return result

    async def get_causal_weight(self, node_id: str) -> int:
        count = 0
        for e in self.edges:
            if e.type in CAUSAL_EDGE_TYPES:
                if str(e.source_id) == node_id or str(e.target_id) == node_id:
                    count += 1
        return count

    async def is_orphan(self, node_id: str) -> bool:
        for e in self.edges:
            if str(e.source_id) == node_id or str(e.target_id) == node_id:
                return False
        return True

    async def get_orphans(self) -> list[MemoryNode]:
        connected = set()
        for e in self.edges:
            connected.add(str(e.source_id))
            connected.add(str(e.target_id))
        return [n for nid, n in self.nodes.items() if nid not in connected]

    async def vector_search(
        self, embedding: list[float], k: int = 10, status_filter: list[MemoryStatus] | None = None
    ) -> list[tuple[MemoryNode, float]]:
        scored = []
        for node in self.nodes.values():
            if status_filter and node.status not in status_filter:
                continue
            if node.embedding:
                sim = cosine_similarity(embedding, node.embedding)
                scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10) -> list[MemoryNode]:
        query_lower = query.lower()
        results = []
        for node in self.nodes.values():
            if query_lower in (node.content_full or node.content_summary).lower():
                results.append(node)
        return results[:k]

    async def get_stats(self) -> dict:
        max_cw = 0
        for nid in self.nodes:
            cw = 0
            for e in self.edges:
                if e.type in CAUSAL_EDGE_TYPES:
                    if str(e.source_id) == nid or str(e.target_id) == nid:
                        cw += 1
            if cw > max_cw:
                max_cw = cw
        return {"nodes": len(self.nodes), "edges": len(self.edges), "max_causal_weight": max_cw}

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()


@dataclass
class BenchmarkScenario:
    name: str
    description: str
    conversation_history: list[dict]
    questions: list[dict]


@dataclass
class QuestionResult:
    question: str
    ground_truth: str
    category: str
    genesys_answer: str = ""
    baseline_answer: str = ""
    genesys_scores: dict = field(default_factory=dict)
    baseline_scores: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    scenario_name: str
    question_results: list[QuestionResult] = field(default_factory=list)
    genesys_avg: dict = field(default_factory=dict)
    baseline_avg: dict = field(default_factory=dict)
    improvement: dict = field(default_factory=dict)


JUDGE_DIMENSIONS = ["factual", "causal", "completeness", "temporal", "outdated"]

JUDGE_PROMPT = """You are evaluating memory system recall quality. Score each dimension 1-5.

Question: {question}
Ground truth: {ground_truth}
System answer (retrieved memories): {answer}

Score each dimension (1=terrible, 5=perfect):
- factual: Are the facts in the answer correct?
- causal: Does the answer capture cause-effect relationships?
- completeness: Does the answer cover all relevant information?
- temporal: Does the answer respect temporal ordering and recency?
- outdated: Does the answer avoid presenting outdated information as current?

Respond with ONLY a JSON object: {{"factual": N, "causal": N, "completeness": N, "temporal": N, "outdated": N}}"""


class LLMJudge:
    """Uses Anthropic Claude to score recall quality on 5 dimensions."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def score(self, question: str, ground_truth: str, answer: str) -> dict[str, int]:
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {d: 1 for d in JUDGE_DIMENSIONS}

    @staticmethod
    def format_prompt(question: str, ground_truth: str, answer: str) -> str:
        """Return the formatted judge prompt (for testing)."""
        return JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )


def load_scenario(path: Path) -> BenchmarkScenario:
    """Load a benchmark scenario from a JSON file."""
    data = json.loads(path.read_text())
    return BenchmarkScenario(
        name=data["name"],
        description=data["description"],
        conversation_history=data["conversation_history"],
        questions=data["questions"],
    )


def load_all_scenarios() -> list[BenchmarkScenario]:
    """Load all scenario files from the scenarios directory."""
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        scenarios.append(load_scenario(path))
    return scenarios


class BenchmarkRunner:
    """Runs side-by-side comparison of Genesys vs flat vector baseline."""

    def __init__(
        self,
        genesys: MCPToolHandler,
        baseline: FlatVectorMemory,
        judge: LLMJudge,
        llm: "LLMProvider | None" = None,
    ):
        self.genesys = genesys
        self.baseline = baseline
        self.judge = judge
        self.llm = llm

    async def _run_background_processing(self, node_id: str, content: str) -> None:
        """Run the same background processing that BackgroundWorkers._on_memory_created does,
        but inline (no Redis event bus needed). This is the key to building real causal structure."""
        if not self.llm:
            return

        graph = self.genesys.graph
        node = await graph.get_node(node_id)
        if not node:
            return

        # 1. Extract entities
        try:
            entities = await self.llm.extract_entities(content)
            if entities:
                await graph.update_node(node_id, {"entity_refs": entities})
        except Exception:
            pass

        # 2. Classify category
        try:
            category = await self.llm.classify_category(content)
            if category:
                await graph.update_node(node_id, {"category": category})
        except Exception:
            pass

        # 3. Infer causal edges (the core differentiator)
        try:
            recent = []
            for status in (MemoryStatus.ACTIVE, MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC, MemoryStatus.CORE):
                recent.extend(await graph.get_nodes_by_status(status, limit=50))
            existing = [(str(n.id), n.content_summary) for n in recent if str(n.id) != node_id]
            if existing:
                edges = await self.llm.infer_causal_edges(content, existing[:30])
                for target_id, edge_type, confidence in edges:
                    try:
                        edge = MemoryEdge(
                            source_id=node.id,
                            target_id=__import__("uuid").UUID(target_id),
                            type=edge_type,
                            weight=confidence,
                        )
                        await graph.create_edge(edge)
                    except (ValueError, KeyError):
                        pass
        except Exception:
            pass

        # 4. Core promotion check
        try:
            node = await graph.get_node(node_id)
            if node:
                from genesys_memory.core_memory.promoter import evaluate_core_promotion, promote_to_core
                should_promote, reason = await evaluate_core_promotion(node, graph)
                if should_promote and reason:
                    await promote_to_core(node_id, reason, graph)
        except Exception:
            pass

    async def _run_forgetting_sweep(self) -> int:
        """Run the forgetting engine to prune irrelevant orphan memories.
        Also sets decay_score to near-zero for orphan noise nodes."""
        graph = self.genesys.graph
        embeddings = self.genesys.embeddings

        # First, recalculate decay scores for all active nodes (without query context,
        # so relevance falls back to recency-based scoring)
        stats = await graph.get_stats()
        max_cw = stats.get("max_causal_weight", 1)

        for node in list(graph.nodes.values()) if hasattr(graph, 'nodes') else []:
            try:
                from genesys_memory.engine.scoring import calculate_decay_score
                score = await calculate_decay_score(
                    node, None, None, graph, embeddings, max_cw
                )
                await graph.update_node(str(node.id), {"decay_score": score})
            except Exception:
                pass

        # Now run the actual forgetting sweep
        from genesys_memory.engine.forgetting import sweep_for_forgetting
        pruned = await sweep_for_forgetting(graph)
        return len(pruned)

    async def _recall_with_scoring(self, query: str, k: int = 10) -> list[dict]:
        """Recall with three-force scoring re-ranking instead of raw cosine similarity."""
        graph = self.genesys.graph
        embeddings = self.genesys.embeddings

        # Get query embedding and entities
        query_embedding = await embeddings.embed(query)

        # Get more candidates than k, then re-rank by decay_score
        candidates = await graph.vector_search(query_embedding, k=k * 3)

        stats = await graph.get_stats()
        max_cw = stats.get("max_causal_weight", 1)

        # Extract entities from query for keyword overlap
        query_entities = None
        if self.llm:
            try:
                query_entities = await self.llm.extract_entities(query)
            except Exception:
                pass

        # Re-score each candidate with full three-force formula
        scored_results = []
        for node, vec_sim in candidates:
            try:
                from genesys_memory.engine.scoring import calculate_decay_score
                score = await calculate_decay_score(
                    node, query_embedding, query_entities, graph, embeddings, max_cw
                )
            except Exception:
                score = vec_sim

            # Also get causal context
            causal_basis = []
            try:
                upstream = await graph.get_causal_chain(str(node.id), "upstream")
                causal_basis = [
                    {"id": str(n.id), "summary": n.content_summary}
                    for n in upstream[:5]
                ]
            except Exception:
                pass

            scored_results.append({
                "id": str(node.id),
                "content": node.content_full or node.content_summary,
                "summary": node.content_summary,
                "status": node.status.value,
                "decay_score": round(score, 4),
                "vec_sim": round(vec_sim, 4),
                "causal_basis": causal_basis,
            })

        # Sort by three-force decay_score, not raw vector similarity
        scored_results.sort(key=lambda m: m["decay_score"], reverse=True)
        return scored_results[:k]

    async def ingest(self, scenario: BenchmarkScenario) -> None:
        """Ingest conversation history into both systems.

        For Genesys: stores memories, creates temporal edges, runs full background
        processing (entity extraction, category classification, LLM causal inference,
        core promotion). This builds the real causal graph structure.
        """
        prev_id: str | None = None
        total = len(scenario.conversation_history)
        for i, turn in enumerate(scenario.conversation_history):
            content = turn["content"]
            related = [prev_id] if prev_id else None
            result = await self.genesys.memory_store(
                content, source_session="benchmark", related_to=related
            )
            curr_id = result["node_id"]

            # Run background processing inline (entity extraction, causal inference, etc.)
            await self._run_background_processing(curr_id, content)

            if (i + 1) % 10 == 0:
                print(f"  Ingested {i + 1}/{total} turns...")

            prev_id = curr_id
            await self.baseline.store(content)

    async def query_both(self, question: str, k: int = 10) -> tuple[str, str]:
        """Query both systems and return formatted answers."""
        # Genesys: use three-force scoring re-ranking
        genesys_memories = await self._recall_with_scoring(question, k=k)
        genesys_answer = "\n".join(
            f"- {m.get('content', m.get('summary', ''))}" for m in genesys_memories
        )

        # Baseline: raw cosine similarity
        baseline_results = await self.baseline.recall(question, k=k)
        baseline_answer = "\n".join(
            f"- {m['content']}" for m in baseline_results
        )

        return genesys_answer or "(no results)", baseline_answer or "(no results)"

    async def run_scenario(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run a complete benchmark scenario."""
        print(f"  Phase 1: Ingesting {len(scenario.conversation_history)} turns with background processing...")
        await self.ingest(scenario)

        # Run forgetting sweep to prune irrelevant orphan memories
        print("  Phase 2: Running forgetting sweep...")
        pruned_count = await self._run_forgetting_sweep()
        graph = self.genesys.graph
        remaining = len(graph.nodes) if hasattr(graph, 'nodes') else 0
        print(f"  Pruned {pruned_count} memories, {remaining} remaining")

        # Query phase
        print(f"  Phase 3: Running {len(scenario.questions)} queries...")
        result = BenchmarkResult(scenario_name=scenario.name)

        for q in scenario.questions:
            genesys_ans, baseline_ans = await self.query_both(q["question"])

            genesys_scores = self.judge.score(q["question"], q["ground_truth"], genesys_ans)
            baseline_scores = self.judge.score(q["question"], q["ground_truth"], baseline_ans)

            qr = QuestionResult(
                question=q["question"],
                ground_truth=q["ground_truth"],
                category=q["category"],
                genesys_answer=genesys_ans,
                baseline_answer=baseline_ans,
                genesys_scores=genesys_scores,
                baseline_scores=baseline_scores,
            )
            result.question_results.append(qr)

        # Calculate averages
        for dim in JUDGE_DIMENSIONS:
            g_scores = [qr.genesys_scores.get(dim, 0) for qr in result.question_results]
            b_scores = [qr.baseline_scores.get(dim, 0) for qr in result.question_results]
            result.genesys_avg[dim] = sum(g_scores) / max(len(g_scores), 1)
            result.baseline_avg[dim] = sum(b_scores) / max(len(b_scores), 1)
            if result.baseline_avg[dim] > 0:
                result.improvement[dim] = (
                    (result.genesys_avg[dim] - result.baseline_avg[dim])
                    / result.baseline_avg[dim]
                    * 100
                )
            else:
                result.improvement[dim] = 0.0

        return result

    async def run_all(self, scenarios: list[BenchmarkScenario]) -> list[BenchmarkResult]:
        """Run all scenarios sequentially."""
        results = []
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Running scenario: {scenario.name}")
            print(f"{'='*60}")
            result = await self.run_scenario(scenario)
            results.append(result)
            # Clear both systems between scenarios
            self.baseline.clear()
            if hasattr(self.genesys.graph, 'clear'):
                self.genesys.graph.clear()
        return results


def generate_report(results: list[BenchmarkResult]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# Genesys Benchmark Results",
        "",
        f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        "",
    ]

    # Overall summary
    all_g = {d: [] for d in JUDGE_DIMENSIONS}
    all_b = {d: [] for d in JUDGE_DIMENSIONS}

    for r in results:
        for dim in JUDGE_DIMENSIONS:
            all_g[dim].append(r.genesys_avg.get(dim, 0))
            all_b[dim].append(r.baseline_avg.get(dim, 0))

    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Dimension | Genesys | Baseline | Improvement |")
    lines.append("|-----------|---------|----------|-------------|")

    for dim in JUDGE_DIMENSIONS:
        g = sum(all_g[dim]) / max(len(all_g[dim]), 1)
        b = sum(all_b[dim]) / max(len(all_b[dim]), 1)
        imp = ((g - b) / b * 100) if b > 0 else 0
        lines.append(f"| {dim} | {g:.2f} | {b:.2f} | {imp:+.1f}% |")

    lines.append("")

    # Per-scenario details
    for r in results:
        lines.append(f"## {r.scenario_name}")
        lines.append("")
        lines.append("| Dimension | Genesys | Baseline | Improvement |")
        lines.append("|-----------|---------|----------|-------------|")
        for dim in JUDGE_DIMENSIONS:
            g = r.genesys_avg.get(dim, 0)
            b = r.baseline_avg.get(dim, 0)
            imp = r.improvement.get(dim, 0)
            lines.append(f"| {dim} | {g:.2f} | {b:.2f} | {imp:+.1f}% |")
        lines.append("")

        # Question details
        lines.append("<details>")
        lines.append(f"<summary>Question details ({len(r.question_results)} questions)</summary>")
        lines.append("")
        for i, qr in enumerate(r.question_results, 1):
            lines.append(f"### Q{i}: {qr.question}")
            lines.append(f"**Category**: {qr.category}")
            lines.append("")
            lines.append(f"**Genesys scores**: {qr.genesys_scores}")
            lines.append(f"**Baseline scores**: {qr.baseline_scores}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def dry_run() -> None:
    """Parse all scenarios and print summary without making API calls."""
    scenarios = load_all_scenarios()
    print(f"Loaded {len(scenarios)} scenarios:\n")
    for s in scenarios:
        print(f"  {s.name}: {s.description}")
        print(f"    Turns: {len(s.conversation_history)}, Questions: {len(s.questions)}")
        categories = set(q["category"] for q in s.questions)
        print(f"    Categories: {', '.join(sorted(categories))}")
        print()

    total_q = sum(len(s.questions) for s in scenarios)
    print(f"Total: {total_q} questions across {len(scenarios)} scenarios")
    print("\nDry run complete. All scenarios parsed successfully.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Genesys benchmark runner")
    parser.add_argument("--dry-run", action="store_true", help="Parse scenarios without running benchmarks")
    parser.add_argument("--scenario", type=str, help="Run a specific scenario by name")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # Validate API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for embeddings", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY required for LLM judge", file=sys.stderr)
        sys.exit(1)

    # Setup providers
    from unittest.mock import AsyncMock
    from genesys_memory.engine.llm_provider import AnthropicLLMProvider

    embeddings = OpenAIEmbeddingProvider(api_key=os.environ["OPENAI_API_KEY"])
    llm = AnthropicLLMProvider(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Genesys with in-memory graph (supports real vector search + causal edges)
    graph = InMemoryGraphProvider()
    mock_cache = AsyncMock()

    genesys = MCPToolHandler(
        graph=graph,
        embeddings=embeddings,
        cache=mock_cache,
    )
    baseline = FlatVectorMemory(embeddings)
    judge = LLMJudge(anthropic.Anthropic())

    # Load scenarios
    scenarios = load_all_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"Error: scenario '{args.scenario}' not found", file=sys.stderr)
            sys.exit(1)

    runner = BenchmarkRunner(genesys, baseline, judge, llm=llm)
    results = await runner.run_all(scenarios)

    # Generate and save report
    report = generate_report(results)
    report_path = Path(__file__).parent / "RESULTS.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
