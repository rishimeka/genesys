"""Tests for the benchmark framework."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from benchmarks.baseline_flat import FlatMemoryEntry, FlatVectorMemory
from benchmarks.run_benchmark import (
    BenchmarkResult,
    BenchmarkScenario,
    JUDGE_DIMENSIONS,
    LLMJudge,
    QuestionResult,
    generate_report,
    load_all_scenarios,
    load_scenario,
)

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "scenarios"


# --- Helpers ---

def _mock_embeddings(dim: int = 8) -> AsyncMock:
    """Create a mock EmbeddingProvider that returns deterministic embeddings."""
    emb = AsyncMock()
    emb.dimension = dim

    call_count = 0

    async def _embed(text: str) -> list[float]:
        nonlocal call_count
        # Deterministic: hash the text for reproducibility
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(dim).tolist()
        call_count += 1
        return vec

    emb.embed = _embed
    emb.embed_batch = AsyncMock(return_value=[])
    return emb


# --- FlatVectorMemory tests ---

class TestFlatVectorMemory:
    @pytest.mark.asyncio
    async def test_store_returns_id(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        entry_id = await fvm.store("hello world")
        assert isinstance(entry_id, str)
        assert len(fvm.memories) == 1

    @pytest.mark.asyncio
    async def test_store_multiple(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        await fvm.store("first")
        await fvm.store("second")
        await fvm.store("third")
        assert len(fvm.memories) == 3

    @pytest.mark.asyncio
    async def test_recall_returns_list(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        await fvm.store("The cat sat on the mat")
        await fvm.store("Dogs are great pets")
        results = await fvm.recall("cat", k=5)
        assert isinstance(results, list)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_recall_empty(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        results = await fvm.recall("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_respects_k(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        for i in range(10):
            await fvm.store(f"memory number {i}")
        results = await fvm.recall("memory", k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_recall_has_required_fields(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        await fvm.store("test content")
        results = await fvm.recall("test")
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "score" in r
        assert "created_at" in r
        assert r["content"] == "test content"

    @pytest.mark.asyncio
    async def test_recall_sorted_by_similarity(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        await fvm.store("alpha")
        await fvm.store("beta")
        await fvm.store("gamma")
        results = await fvm.recall("query", k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_clear(self):
        emb = _mock_embeddings()
        fvm = FlatVectorMemory(emb)
        await fvm.store("one")
        await fvm.store("two")
        fvm.clear()
        assert len(fvm.memories) == 0


# --- Scenario loading tests ---

class TestScenarioLoading:
    def test_load_all_scenarios(self):
        scenarios = load_all_scenarios()
        assert len(scenarios) == 4
        names = {s.name for s in scenarios}
        assert names == {"causal_reasoning", "temporal_awareness", "structural_importance", "outdated_info"}

    def test_scenario_has_required_fields(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            assert isinstance(s, BenchmarkScenario)
            assert s.name
            assert s.description
            assert len(s.conversation_history) > 0
            assert len(s.questions) >= 10

    def test_conversation_turn_format(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            for turn in s.conversation_history:
                assert "turn" in turn
                assert "week" in turn
                assert "content" in turn
                assert isinstance(turn["content"], str)
                assert len(turn["content"]) > 0

    def test_question_format(self):
        scenarios = load_all_scenarios()
        for s in scenarios:
            for q in s.questions:
                assert "question" in q
                assert "ground_truth" in q
                assert "category" in q
                assert "requires" in q

    def test_load_single_scenario(self):
        path = SCENARIOS_DIR / "causal_reasoning.json"
        s = load_scenario(path)
        assert s.name == "causal_reasoning"
        assert len(s.questions) == 13

    def test_scenario_json_valid(self):
        """Verify all scenario JSON files are valid."""
        for path in SCENARIOS_DIR.glob("*.json"):
            data = json.loads(path.read_text())
            assert "name" in data
            assert "conversation_history" in data
            assert "questions" in data


# --- Judge prompt formatting tests ---

class TestLLMJudge:
    def test_format_prompt_contains_question(self):
        prompt = LLMJudge.format_prompt(
            question="What happened?",
            ground_truth="Something important",
            answer="Retrieved memory content",
        )
        assert "What happened?" in prompt
        assert "Something important" in prompt
        assert "Retrieved memory content" in prompt

    def test_format_prompt_contains_all_dimensions(self):
        prompt = LLMJudge.format_prompt("q", "gt", "a")
        for dim in JUDGE_DIMENSIONS:
            assert dim in prompt

    def test_judge_dimensions_complete(self):
        assert set(JUDGE_DIMENSIONS) == {"factual", "causal", "completeness", "temporal", "outdated"}


# --- Report generation tests ---

class TestReportGeneration:
    def _make_result(self, name: str = "test_scenario") -> BenchmarkResult:
        qr = QuestionResult(
            question="Test question?",
            ground_truth="Test ground truth",
            category="test",
            genesys_answer="genesys answer",
            baseline_answer="baseline answer",
            genesys_scores={"factual": 4, "causal": 5, "completeness": 4, "temporal": 3, "outdated": 4},
            baseline_scores={"factual": 3, "causal": 2, "completeness": 3, "temporal": 3, "outdated": 3},
        )
        result = BenchmarkResult(
            scenario_name=name,
            question_results=[qr],
            genesys_avg={"factual": 4.0, "causal": 5.0, "completeness": 4.0, "temporal": 3.0, "outdated": 4.0},
            baseline_avg={"factual": 3.0, "causal": 2.0, "completeness": 3.0, "temporal": 3.0, "outdated": 3.0},
            improvement={"factual": 33.3, "causal": 150.0, "completeness": 33.3, "temporal": 0.0, "outdated": 33.3},
        )
        return result

    def test_report_is_markdown(self):
        result = self._make_result()
        report = generate_report([result])
        assert report.startswith("# Genesys Benchmark Results")

    def test_report_contains_scenario(self):
        result = self._make_result("my_scenario")
        report = generate_report([result])
        assert "my_scenario" in report

    def test_report_contains_dimensions(self):
        result = self._make_result()
        report = generate_report([result])
        for dim in JUDGE_DIMENSIONS:
            assert dim in report

    def test_report_contains_tables(self):
        result = self._make_result()
        report = generate_report([result])
        assert "| Dimension |" in report
        assert "| Genesys |" in report or "Genesys" in report

    def test_report_multiple_scenarios(self):
        results = [self._make_result("scenario_a"), self._make_result("scenario_b")]
        report = generate_report(results)
        assert "scenario_a" in report
        assert "scenario_b" in report

    def test_report_empty_results(self):
        report = generate_report([])
        assert "# Genesys Benchmark Results" in report
