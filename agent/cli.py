"""
Interactive CLI for MonetAgent multi-turn image editing.

Usage:
    python -m agent.cli --image path/to/image.tif --style balanced
"""

from __future__ import annotations

import argparse
import json

from langgraph.types import Command

from .graph import build_agent_graph
from .session import create_checkpointer, create_session, load_agent_config


def main():
    parser = argparse.ArgumentParser(description="MonetAgent Interactive CLI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--style", default="balanced", choices=["balanced", "vibrant", "retro"])
    parser.add_argument("--config", default="configs/agent_config.yaml", help="Agent config path")
    args = parser.parse_args()

    # Build graph
    agent_config = load_agent_config(args.config)
    checkpointer = create_checkpointer(agent_config)
    graph = build_agent_graph(checkpointer=checkpointer, config=agent_config)

    # Create session
    initial_state, session_id = create_session(
        args.image, style=args.style, config_path=args.config,
    )
    thread_config = {"configurable": {"thread_id": session_id}}

    print(f"Session: {session_id}")
    print(f"Image: {initial_state['original_image_path']}")
    print(f"Style: {args.style}")
    print("Type your editing request (or 'quit' to exit):\n")

    turn = 0
    while True:
        try:
            user_input = input(f"[Turn {turn}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # Set user message for this turn
        initial_state["user_message"] = user_input

        # Invoke graph
        try:
            result = graph.invoke(initial_state, config=thread_config)
        except Exception as e:
            # Check for interrupt (human review)
            state = graph.get_state(thread_config)
            if state and state.next and "human_review" in state.next:
                result = _handle_human_review(graph, thread_config, state)
            else:
                print(f"Error: {e}")
                continue

        # Display result
        if result:
            current = result.get("current_image_path", "")
            score = result.get("quality_score", 0)
            assessment = result.get("quality_assessment", "")
            print(f"\nResult: {current}")
            print(f"Quality: {score:.2f}")
            if assessment:
                print(f"Assessment: {assessment}")
            print()

        # Prepare for next turn (carry forward the state)
        initial_state = dict(result) if result else initial_state
        turn += 1


def _handle_human_review(graph, thread_config, state):
    """Handle human-in-the-loop interrupt at the CLI."""
    # Display review info
    values = state.values
    print("\n--- Human Review ---")
    print(f"Current image: {values.get('current_image_path', '')}")
    print(f"Quality score: {values.get('quality_score', 0):.2f}")
    print(f"Assessment: {values.get('quality_assessment', '')}")
    print("\nSub-goals:")
    for sg in values.get("sub_goals", []):
        print(f"  #{sg.get('id', '?')}: [{sg.get('status', '?')}] {sg.get('description', '')}")

    print("\nOptions: [a]pprove / [m]odify / [r]eplan / [b]rollback")
    choice = input("Decision: ").strip().lower()

    decision_map = {"a": "approve", "m": "modify", "r": "replan", "b": "rollback"}
    decision = decision_map.get(choice, "approve")
    resume_payload = {"decision": decision, "modifications": {}}

    if decision in ("modify", "rollback"):
        print("Enter JSON payload (blank for {}):")
        raw = input("> ").strip()
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    if decision == "modify":
                        resume_payload["modifications"] = payload
                    else:
                        resume_payload["rollback_target"] = payload
            except json.JSONDecodeError:
                print("Invalid JSON payload. Continuing with empty payload.")

    # Resume the graph with the human decision
    result = graph.invoke(
        Command(resume=resume_payload),
        config=thread_config,
    )
    return result


if __name__ == "__main__":
    main()
