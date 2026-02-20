"""
MonetAgent Gradio Demo — Interactive multi-turn image editing with hierarchical agent.

Layout:
  Left column:  Original + Current image comparison
  Middle column: Chat history + input
  Right column:  JSON adjustment history + sub-goal status
  Bottom:        Intermediate image gallery (asset_graph)
  Review panel:  Quality score + assessment + approve/modify/rollback/replan buttons
"""

from __future__ import annotations

import json
import os
import gradio as gr
from langgraph.types import Command

from agent.graph import build_agent_graph
from agent.nodes import warmup_local_editing_models
from agent.session import create_checkpointer, create_session, load_agent_config


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_graph = None
_thread_config = None
_current_state = None
_waiting_for_review = False
_agent_config = None
_config_path = "configs/agent_config.yaml"


def _get_graph():
    global _graph, _agent_config
    if _graph is None:
        _agent_config = load_agent_config(_config_path)
        checkpointer = create_checkpointer(_agent_config)
        _graph = build_agent_graph(checkpointer=checkpointer, config=_agent_config)
    return _graph


# ---------------------------------------------------------------------------
# Core handlers
# ---------------------------------------------------------------------------

def init_session(image_path: str, style: str):
    """Initialize a new editing session."""
    global _thread_config, _current_state, _waiting_for_review

    if not image_path:
        return None, None, "Please upload an image first.", "[]", []

    state, session_id = create_session(image_path, style=style, config_path=_config_path)
    _thread_config = {"configurable": {"thread_id": session_id}}
    _current_state = state
    _waiting_for_review = False

    warmup_ok, warmup_msg = warmup_local_editing_models()
    if warmup_ok:
        status = f"Session created: {session_id} | local models prewarmed"
    else:
        status = f"Session created: {session_id} | {warmup_msg}"

    return (
        image_path,                             # original image
        image_path,                             # current image (same initially)
        status,                                 # status message
        "[]",                                   # adjustments JSON
        [],                                     # gallery images
    )


def send_message(user_message: str, chat_history: list):
    """Process a user editing request through the agent pipeline."""
    global _current_state, _waiting_for_review

    if not _current_state:
        chat_history.append({"role": "assistant", "content": "Please initialize a session first."})
        return chat_history, None, "No session", "[]", []

    if not user_message.strip():
        return chat_history, None, "", "[]", []

    # Add user message to chat
    chat_history.append({"role": "user", "content": user_message})
    _current_state["user_message"] = user_message

    graph = _get_graph()

    try:
        result = graph.invoke(_current_state, config=_thread_config)
        _current_state = dict(result)
        _waiting_for_review = False
    except Exception as e:
        # Check for interrupt (human review needed)
        state = graph.get_state(_thread_config)
        if state and state.next and "human_review" in state.next:
            _waiting_for_review = True
            values = state.values
            _current_state = dict(values)

            review_msg = (
                f"**Review needed** (Quality: {values.get('quality_score', 0):.2f})\n\n"
                f"{values.get('quality_assessment', '')}\n\n"
                "Use the review panel below to approve, modify, replan, or rollback."
            )
            chat_history.append({"role": "assistant", "content": review_msg})

            return (
                chat_history,
                values.get("current_image_path"),
                f"Score: {values.get('quality_score', 0):.2f} — Waiting for review",
                _format_adjustments(values),
                _get_gallery_images(values),
            )
        else:
            chat_history.append({"role": "assistant", "content": f"Error: {e}"})
            return chat_history, None, str(e), "[]", []

    # Success path
    current_img = _current_state.get("current_image_path", "")
    score = _current_state.get("quality_score", 0)
    assessment = _current_state.get("quality_assessment", "")

    response = f"Edit applied. Quality: {score:.2f}\n\n{assessment}"
    chat_history.append({"role": "assistant", "content": response})

    return (
        chat_history,
        current_img,
        f"Turn {_current_state.get('turn_id', 0)} — Score: {score:.2f}",
        _format_adjustments(_current_state),
        _get_gallery_images(_current_state),
    )


def handle_review(decision: str, review_json: str):
    """Handle human review decision (approve/modify/replan/rollback)."""
    global _current_state, _waiting_for_review

    if not _waiting_for_review:
        return "No review pending.", None, "", "[]", []

    extra_payload = {}
    if review_json and review_json.strip():
        try:
            parsed = json.loads(review_json)
            if isinstance(parsed, dict):
                extra_payload = parsed
        except json.JSONDecodeError:
            return "Invalid JSON in review payload.", None, "", "[]", []

    resume_payload = {"decision": decision, "modifications": {}}
    if decision == "modify":
        resume_payload["modifications"] = extra_payload
    elif decision == "rollback":
        resume_payload["rollback_target"] = extra_payload

    graph = _get_graph()

    try:
        result = graph.invoke(
            Command(resume=resume_payload),
            config=_thread_config,
        )
        _current_state = dict(result)
        _waiting_for_review = False
        current_img = _current_state.get("current_image_path", "")
        status = f"Turn {_current_state.get('turn_id', 0)} — Score: {_current_state.get('quality_score', 0):.2f}"
        adjustments = _format_adjustments(_current_state)
        gallery = _get_gallery_images(_current_state)
        return f"Decision '{decision}' applied.", current_img, status, adjustments, gallery
    except Exception as e:
        return f"Error applying decision: {e}", None, str(e), "[]", []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_adjustments(state: dict) -> str:
    """Format sub-goal adjustments as JSON string for display."""
    sub_goals = state.get("sub_goals", [])
    display = []
    for sg in sub_goals:
        entry = {
            "id": sg.get("id"),
            "type": sg.get("stage_type"),
            "category": sg.get("operation_category"),
            "description": sg.get("description"),
            "status": sg.get("status"),
        }
        if sg.get("adjustments"):
            entry["adjustments"] = sg["adjustments"]
        if sg.get("local_specs"):
            entry["local_edits"] = len(sg["local_specs"])
        display.append(entry)
    return json.dumps(display, indent=2, ensure_ascii=False)


def _get_gallery_images(state: dict) -> list:
    """Get list of intermediate image paths from asset_graph."""
    images = []
    for node in state.get("asset_graph", []):
        uri = node.get("uri", "")
        if uri and os.path.exists(uri):
            images.append(uri)
    return images


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo():
    """Build and return the Gradio demo interface."""
    with gr.Blocks(title="MonetAgent — Multi-Turn Image Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MonetAgent — Hierarchical Multi-Turn Image Editor")

        with gr.Row():
            # Left: Image comparison
            with gr.Column(scale=2):
                with gr.Row():
                    original_img = gr.Image(label="Original", type="filepath", interactive=False)
                    current_img = gr.Image(label="Current", type="filepath", interactive=False)

            # Middle: Chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Editing Chat", type="messages", height=400)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Describe your editing request...",
                        label="Message",
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # Right: Status & Adjustments
            with gr.Column(scale=2):
                status_text = gr.Textbox(label="Status", interactive=False)
                adjustments_json = gr.Code(label="Adjustments", language="json")

        # Bottom: Gallery
        gallery = gr.Gallery(label="Intermediate Images", columns=6, height=150)

        # Session controls
        with gr.Accordion("Session Setup", open=True):
            with gr.Row():
                upload_img = gr.Image(label="Upload Image", type="filepath")
                style_dropdown = gr.Dropdown(
                    choices=["balanced", "vibrant", "retro"],
                    value="balanced",
                    label="Style",
                )
                init_btn = gr.Button("Start Session", variant="primary")

        # Review panel
        with gr.Accordion("Human Review", open=False):
            gr.Markdown("When quality review is triggered, use these buttons:")
            review_json = gr.Code(
                label="Review Payload JSON (modify: overrides, rollback: target)",
                language="json",
                value="{}",
            )
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                modify_btn = gr.Button("Modify", variant="secondary")
                replan_btn = gr.Button("Replan", variant="secondary")
                rollback_btn = gr.Button("Rollback", variant="stop")
            review_output = gr.Textbox(label="Review Result", interactive=False)

        # --- Event handlers ---
        init_btn.click(
            fn=init_session,
            inputs=[upload_img, style_dropdown],
            outputs=[original_img, current_img, status_text, adjustments_json, gallery],
        )

        send_btn.click(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, current_img, status_text, adjustments_json, gallery],
        )
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, current_img, status_text, adjustments_json, gallery],
        )

        approve_btn.click(
            fn=lambda payload: handle_review("approve", payload),
            inputs=[review_json],
            outputs=[review_output, current_img, status_text, adjustments_json, gallery],
        )
        modify_btn.click(
            fn=lambda payload: handle_review("modify", payload),
            inputs=[review_json],
            outputs=[review_output, current_img, status_text, adjustments_json, gallery],
        )
        replan_btn.click(
            fn=lambda payload: handle_review("replan", payload),
            inputs=[review_json],
            outputs=[review_output, current_img, status_text, adjustments_json, gallery],
        )
        rollback_btn.click(
            fn=lambda payload: handle_review("rollback", payload),
            inputs=[review_json],
            outputs=[review_output, current_img, status_text, adjustments_json, gallery],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
