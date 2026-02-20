import unittest

from agent.context_folding import fold_context_for_prompt, prune_asset_graph


class ContextFoldingTests(unittest.TestCase):
    def setUp(self):
        self.original = "sessions/s1/original.tif"
        self.step0 = "sessions/s1/turn0_step0.tif"
        self.step1 = "sessions/s1/turn0_step1.tif"
        self.asset_graph = [
            {"uri": self.original, "parent_uri": "", "transform_summary": "Original image", "turn_id": 0},
            {"uri": self.step0, "parent_uri": self.original, "transform_summary": "T0S0", "turn_id": 0},
            {"uri": self.step1, "parent_uri": self.step0, "transform_summary": "T0S1", "turn_id": 0},
        ]

    def test_prune_asset_graph_keeps_ancestor_chain(self):
        pruned = prune_asset_graph(self.asset_graph, self.step1)
        self.assertEqual([n["uri"] for n in pruned], [self.original, self.step0, self.step1])

    def test_fold_context_respects_approx_token_budget(self):
        action_history = [
            {"turn_id": 0, "intent": "make it bright", "plan_summary": "exp+10", "outcome": "completed"},
            {"turn_id": 1, "intent": "make it warmer", "plan_summary": "temp+20", "outcome": "completed"},
            {"turn_id": 2, "intent": "reduce highlights", "plan_summary": "highlights-15", "outcome": "completed"},
        ]
        folded = fold_context_for_prompt(
            self.asset_graph,
            action_history,
            current_image_path=self.step1,
            keep_recent=2,
            max_history_tokens=18,
        )

        # Budgeted output should keep content but stay concise.
        self.assertTrue(len(folded.split()) <= 18)
        self.assertTrue("Turn 1" in folded or "Turn 2" in folded)


if __name__ == "__main__":
    unittest.main()
