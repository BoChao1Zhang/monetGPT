import os
import tempfile
import unittest
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import build_agent_graph
from agent.session import create_session
import agent.nodes as nodes


class _StubInferenceEngine:
    def __init__(self, quality_scores):
        self._quality_scores = list(quality_scores)

    def query_structured(self, image_path, system_prompt, user_prompt, temperature=0.2, original_image_path=""):
        if original_image_path:
            return "quality"
        return "planner"

    def extract_json_array_from_response(self, response):
        return [
            {
                "id": 1,
                "stage_type": "global",
                "operation_category": "white-balance-tone-contrast",
                "description": "Slightly increase exposure",
                "adjustments": {"Exposure": 5},
                "local_specs": [],
            }
        ]

    def extract_json_from_response(self, response):
        score = self._quality_scores.pop(0) if self._quality_scores else 0.9
        return {
            "score": score,
            "assessment": f"stub quality score={score}",
        }


class _StubPipeline:
    def execute_single_stage(self, adjustments, src_path, output_path, is_local=False):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Keep this e2e smoke deterministic and fast: just copy source as output.
        with open(src_path, "rb") as rf, open(output_path, "wb") as wf:
            wf.write(rf.read())


class AgentE2ETests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "agent_config.yaml"
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "planner": {
                "max_sub_goals": 3,
                "temperature": 0.2,
                "max_retries_per_sub_goal": 2,
            },
            "quality": {
                "min_pass_score": 0.6,
                "temperature": 0.1,
                "auto_approve_threshold": 0.85,
            },
            "context_folding": {
                "max_history_tokens": 2000,
                "keep_recent_turns": 3,
            },
            "session": {
                "base_dir": str(self.sessions_dir),
                "checkpoint_backend": "memory",
                "sqlite_path": str(self.sessions_dir / "checkpoints.db"),
            },
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        sample = Path("assets/test/a1329-IMG_3540.png")
        if not sample.exists():
            raise RuntimeError("Missing test image assets/test/a1329-IMG_3540.png")
        self.image_path = str(sample)

        self._orig_get_engine = nodes._get_inference_engine
        self._orig_get_pipeline = nodes._get_image_pipeline

    def tearDown(self):
        nodes._get_inference_engine = self._orig_get_engine
        nodes._get_image_pipeline = self._orig_get_pipeline
        self.temp_dir.cleanup()

    def _build_graph_state(self, quality_scores, session_id):
        nodes._get_inference_engine = lambda: _StubInferenceEngine(quality_scores)
        nodes._get_image_pipeline = lambda: _StubPipeline()

        graph = build_agent_graph(checkpointer=MemorySaver())
        state, sid = create_session(
            self.image_path,
            style="balanced",
            config_path=str(self.config_path),
            session_id=session_id,
        )
        state["user_message"] = "please make it slightly brighter"
        thread = {"configurable": {"thread_id": sid}}
        return graph, state, thread

    def test_e2e_auto_approve_flow(self):
        graph, state, thread = self._build_graph_state([0.92], "e2e_auto")
        result = graph.invoke(state, config=thread)

        self.assertEqual(result.get("turn_id"), 1)
        self.assertEqual(len(result.get("action_history", [])), 1)
        self.assertTrue(os.path.exists(result.get("current_image_path", "")))

    def test_e2e_human_review_interrupt_then_approve(self):
        graph, state, thread = self._build_graph_state([0.7], "e2e_review")

        interrupted = False
        try:
            first = graph.invoke(state, config=thread)
            interrupted = isinstance(first, dict) and "__interrupt__" in first
        except Exception:
            interrupted = True

        self.assertTrue(interrupted, "Expected human_review interrupt in medium-score flow")

        resumed = graph.invoke(
            Command(resume={"decision": "approve", "modifications": {}}),
            config=thread,
        )

        self.assertEqual(resumed.get("turn_id"), 1)
        self.assertEqual(len(resumed.get("action_history", [])), 1)
        self.assertTrue(os.path.exists(resumed.get("current_image_path", "")))


if __name__ == "__main__":
    unittest.main()
