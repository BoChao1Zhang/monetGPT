"""
Context Folding — 3-layer memory compression for multi-turn stability.

Layer 1 - AssetContext:  Image version DAG (uri, parent, transform)
Layer 2 - ToolContext:   Ephemeral tool params (cleared each fold)
Layer 3 - ActionContext: Persistent high-level summaries (accumulated)

Folding strategy:
- Recent 3 turns: full ActionRecord retained
- Older turns: compressed to single-line "T{n}:{outcome}"
- AssetGraph: only current_image_path ancestor chain + original
- Token budget: ~2000 tokens for planner context
"""

from __future__ import annotations

from typing import List

def fold_context_for_prompt(
    asset_graph: List[dict],
    action_history: List[dict],
    current_image_path: str = "",
    keep_recent: int = 3,
    max_history_tokens: int = 2000,
) -> str:
    """
    Generate compressed history string for planner prompt.

    Args:
        asset_graph: List of AssetNode dicts from state.
        action_history: List of ActionRecord dicts from state.
        current_image_path: Current image path for ancestor chain pruning.
        keep_recent: Number of recent turns to keep in full detail.

    Returns:
        Compressed context string (~2000 tokens max).
    """
    lines = []

    # --- Asset lineage (Layer 1) ---
    if asset_graph:
        ancestor_chain = _get_ancestor_chain(asset_graph, current_image_path)
        if ancestor_chain:
            lines.append("### Image Lineage")
            for node in ancestor_chain:
                summary = node.get("transform_summary", "original")
                lines.append(f"  {summary}")

    # --- Action history (Layer 3) ---
    if action_history:
        lines.append("### Edit History")
        n = len(action_history)
        for i, record in enumerate(action_history):
            turn_id = record.get("turn_id", i)
            if i < n - keep_recent:
                # Compress old turns
                outcome = record.get("outcome", "?")
                lines.append(f"  T{turn_id}:{outcome}")
            else:
                # Keep recent turns in full
                intent = record.get("intent", "")
                plan = record.get("plan_summary", "")
                outcome = record.get("outcome", "")
                lines.append(f"  Turn {turn_id}: {intent}")
                if plan:
                    lines.append(f"    Plan: {plan}")
                lines.append(f"    Outcome: {outcome}")

    if not lines:
        return ""

    return _trim_to_token_budget(lines, max_history_tokens)


def prune_asset_graph(
    asset_graph: List[dict],
    current_image_path: str,
) -> List[dict]:
    """
    Return only the ancestor chain of current_image_path + the original.
    Used to keep state size bounded.
    """
    return _get_ancestor_chain(asset_graph, current_image_path)


def _get_ancestor_chain(
    asset_graph: List[dict],
    target_uri: str,
) -> List[dict]:
    """Walk backwards from target_uri to the root (original image)."""
    if not asset_graph:
        return []

    # Build uri → node lookup
    by_uri = {}
    for node in asset_graph:
        uri = node.get("uri", "")
        if uri:
            by_uri[uri] = node

    if not target_uri:
        # Keep only original + latest when current target is missing.
        if len(asset_graph) <= 2:
            return list(asset_graph)
        return [asset_graph[0], asset_graph[-1]]

    if target_uri not in by_uri:
        if len(asset_graph) <= 2:
            return list(asset_graph)
        return [asset_graph[0], asset_graph[-1]]

    chain = []
    current = target_uri
    visited = set()

    while current and current in by_uri and current not in visited:
        visited.add(current)
        node = by_uri[current]
        chain.append(node)
        current = node.get("parent_uri", "")

    chain.reverse()
    return chain


def _trim_to_token_budget(lines: List[str], max_tokens: int) -> str:
    """
    Trim folded context to an approximate token budget.

    We use a lightweight approximation: whitespace-delimited words.
    """
    if max_tokens <= 0:
        return ""

    kept: List[str] = []
    used = 0
    # Keep the latest lines first; these contain the most recent actionable history.
    for line in reversed(lines):
        token_cost = max(1, len(line.split()))
        if used + token_cost > max_tokens:
            continue
        kept.append(line)
        used += token_cost

    kept.reverse()
    return "\n".join(kept)
