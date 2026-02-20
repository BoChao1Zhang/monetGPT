import unittest

from agent.nodes import (
    _apply_human_modifications,
    _resolve_rollback_uri,
    route_after_quality,
)


class AgentNodeRoutingTests(unittest.TestCase):
    def test_route_after_quality_auto_approve_to_context_fold(self):
        state = {
            "quality_score": 0.92,
            "quality_pass": True,
            "sub_goals": [{"retry_count": 0}],
            "agent_config": {
                "quality": {"min_pass_score": 0.6, "auto_approve_threshold": 0.85},
                "planner": {"max_retries_per_sub_goal": 2},
            },
        }
        self.assertEqual(route_after_quality(state), "context_fold")

    def test_route_after_quality_replan_when_below_threshold_and_retry_available(self):
        state = {
            "quality_score": 0.45,
            "quality_pass": False,
            "sub_goals": [{"retry_count": 1}],
            "agent_config": {
                "quality": {"min_pass_score": 0.6, "auto_approve_threshold": 0.85},
                "planner": {"max_retries_per_sub_goal": 2},
            },
        }
        self.assertEqual(route_after_quality(state), "planner")

    def test_route_after_quality_to_human_review_when_retry_exhausted(self):
        state = {
            "quality_score": 0.4,
            "quality_pass": False,
            "sub_goals": [{"retry_count": 2}],
            "agent_config": {
                "quality": {"min_pass_score": 0.6, "auto_approve_threshold": 0.85},
                "planner": {"max_retries_per_sub_goal": 2},
            },
        }
        self.assertEqual(route_after_quality(state), "human_review")

    def test_route_after_quality_to_human_review_when_quality_replan_budget_exhausted(self):
        state = {
            "quality_score": 0.4,
            "quality_pass": False,
            "replan_attempts": 2,
            "sub_goals": [{"retry_count": 0}],
            "agent_config": {
                "quality": {"min_pass_score": 0.6, "auto_approve_threshold": 0.85},
                "planner": {"max_retries_per_sub_goal": 2},
            },
        }
        self.assertEqual(route_after_quality(state), "human_review")


class AgentModificationTests(unittest.TestCase):
    def setUp(self):
        self.original = "sessions/s1/original.tif"
        self.step0 = "sessions/s1/turn0_step0.tif"
        self.step1 = "sessions/s1/turn0_step1.tif"
        self.state = {
            "turn_id": 0,
            "original_image_path": self.original,
            "current_image_path": self.step1,
            "current_sub_goal_idx": 2,
            "sub_goals": [
                {
                    "id": 1,
                    "stage_type": "global",
                    "operation_category": "white-balance-tone-contrast",
                    "description": "step0",
                    "adjustments": {"Exposure": 5},
                    "local_specs": [],
                    "status": "completed",
                    "retry_count": 0,
                },
                {
                    "id": 2,
                    "stage_type": "global",
                    "operation_category": "color-temperature",
                    "description": "step1",
                    "adjustments": {"Temperature": 10},
                    "local_specs": [],
                    "status": "completed",
                    "retry_count": 0,
                },
            ],
            "asset_graph": [
                {
                    "uri": self.original,
                    "parent_uri": "",
                    "transform_summary": "Original image",
                    "turn_id": 0,
                    "step_id": -1,
                },
                {
                    "uri": self.step0,
                    "parent_uri": self.original,
                    "transform_summary": "T0S0",
                    "turn_id": 0,
                    "step_id": 0,
                },
                {
                    "uri": self.step1,
                    "parent_uri": self.step0,
                    "transform_summary": "T0S1",
                    "turn_id": 0,
                    "step_id": 1,
                },
            ],
        }

    def test_apply_human_modifications_resets_from_first_changed_step(self):
        modifications = {"sub_goal_id": 2, "adjustments": {"Temperature": -5}}
        result = _apply_human_modifications(self.state, modifications)

        self.assertTrue(result["changed"])
        self.assertEqual(result["start_idx"], 1)
        self.assertEqual(result["resume_image"], self.step0)
        self.assertEqual(result["sub_goals"][1]["adjustments"], {"Temperature": -5})
        self.assertEqual(result["sub_goals"][1]["status"], "pending")
        # stale step1 node is removed from this turn
        uris = [n.get("uri") for n in result["asset_graph"]]
        self.assertNotIn(self.step1, uris)

    def test_resolve_rollback_uri_by_turn_and_step(self):
        rollback_uri, target = _resolve_rollback_uri(self.state, {"turn_id": 0, "step_id": 0})
        self.assertEqual(rollback_uri, self.step0)
        self.assertEqual(target["asset_uri"], self.step0)


if __name__ == "__main__":
    unittest.main()
