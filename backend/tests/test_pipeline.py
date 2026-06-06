"""Unit tests for the custom ResearchPipeline runner."""

from __future__ import annotations

from app.graph.pipeline import ResearchPipeline


class TestPipeline:
    async def test_runs_nodes_in_order_over_shared_state(self):
        async def a(state):
            state.setdefault("order", []).append("a")

        async def b(state):
            state["order"].append("b")

        pipeline = ResearchPipeline().add("a", a).add("b", b)
        state = await pipeline.run({})
        assert state["order"] == ["a", "b"]

    async def test_failure_is_isolated_and_recorded(self):
        async def boom(state):
            raise ValueError("kaboom")

        async def after(state):
            state["after_ran"] = True

        pipeline = ResearchPipeline().add("boom", boom).add("after", after)
        state = await pipeline.run({})
        # the failing node is recorded, but the next node still runs
        assert state["after_ran"] is True
        assert any("boom: kaboom" in e for e in state["errors"])

    async def test_retry_succeeds_on_second_attempt(self):
        calls = {"n": 0}

        async def flaky(state):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            state["ok"] = True

        pipeline = ResearchPipeline().add("flaky", flaky, retries=1)
        state = await pipeline.run({})
        assert state["ok"] is True
        assert calls["n"] == 2
        assert state["errors"] == []

    async def test_retry_exhausted_records_error(self):
        calls = {"n": 0}

        async def always_fail(state):
            calls["n"] += 1
            raise RuntimeError("nope")

        pipeline = ResearchPipeline().add("always", always_fail, retries=2)
        state = await pipeline.run({})
        assert calls["n"] == 3  # 1 initial + 2 retries
        assert any("always: nope" in e for e in state["errors"])
