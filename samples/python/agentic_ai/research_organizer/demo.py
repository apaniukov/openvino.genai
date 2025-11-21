#!/usr/bin/env python3
"""
Demo script for Research Organizer
Demonstrates natural language interaction through the orchestrator.
"""

import argparse
import asyncio
import os
import sys

from tools import init_database  # type: ignore[import]
from agents import init_llm, OrchestratorAgent
from mcp_tools.client import MCPClient, MCPServerConfig
import ui


DEMO_REQUIRED_INTENTS = {
    "add_topic",
    "list_topics",
    "add_paper",
    "list_papers_by_topic",
    "summarize_topic",
}


def _run_async(coro_factory):
    """Run an async callable in a fresh event loop if needed."""
    try:
        return asyncio.run(coro_factory())
    except RuntimeError as exc:
        message = "asyncio.run() cannot be called from a running event loop"
        if message not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_factory())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def create_mcp_clients() -> list[MCPClient]:
    """Instantiate MCP clients for the demo servers."""
    python_cmd = sys.executable or "python3"
    base_module = "mcp_tools"
    configs = [
        MCPServerConfig(
            name="research-db",
            command=python_cmd,
            args=["-m", f"{base_module}.db"],
        ),
        MCPServerConfig(
            name="research-arxiv",
            command=python_cmd,
            args=["-m", f"{base_module}.arxiv"],
            env={
                "http_proxy": os.environ.get("http_proxy", ""),
                "https_proxy": os.environ.get("https_proxy", ""),
            }
        ),
        MCPServerConfig(
            name="research-topics",
            command=python_cmd,
            args=["-m", f"{base_module}.topics"],
            env={
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            }
        ),
        MCPServerConfig(
            name="research-summary",
            command=python_cmd,
            args=["-m", f"{base_module}.summary"],
            env={
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            }
        ),
    ]
    return [MCPClient(config) for config in configs]


def close_mcp_clients(clients: list[MCPClient]) -> None:
    """Gracefully close MCP client connections."""
    for client in clients:
        try:
            _run_async(client.close)
        except Exception:
            continue


def demo(model_path, device="CPU"):
    """
    Run a demo of the research organizer using natural language.
    Demonstrates orchestrator-based interaction with the system.
    
    Args:
        model_path: path to LLM model directory
        device: Device for LLM inference
    """
    ui.print_header()
    print("\n🎬 Running Research Organizer Demo...\n")
    
    try:
        ui.print_info("Initializing LLM...")
        init_llm(model_path, device=device)
        ui.print_success("LLM initialized")
    except Exception as e:
        ui.print_warning(f"Failed to initialize LLM: {e}")
        return
    
    # Initialize database
    init_database(init_new_db=True)
    ui.print_success("Database initialized")
    
    clients: list[MCPClient] = []
    orchestrator: OrchestratorAgent | None = None

    try:
        ui.print_info("Initializing MCP clients...")
        clients = create_mcp_clients()
        orchestrator = OrchestratorAgent(clients)

        missing_intents = DEMO_REQUIRED_INTENTS - set(
            orchestrator.supported_intents
        )
        if missing_intents:
            missing_list = ", ".join(sorted(missing_intents))
            ui.print_error(
                f"Required intents are unavailable: {missing_list}."
            )
            return

        ui.print_success("Orchestrator agent ready")

        ui.print_divider()
        ui.print_info(
            "📁 Step 1: Adding research topics using natural language"
        )
        print()

        topic_queries = [
            (
                "add machine learning as a topic about algorithms that learn "
                "from data"
            ),
            (
                "add a topic called natural language processing for "
                "processing and understanding human language and a "
                "computer vision topic for teaching computers to "
                "understand images"
            ),
        ]

        for query in topic_queries:
            ui.display_user_input(query)
            intent_result = orchestrator.parse_intent(query)
            orchestrator.execute_action(intent_result)
            print()

        ui.display_user_input("show me all the topics")
        intent_result = orchestrator.parse_intent("show me all the topics")
        orchestrator.execute_action(intent_result)

        ui.print_divider()
        ui.print_info(
            "📄 Step 2: Adding a paper from ArXiv using natural language"
        )
        print()

        sample_arxiv_url = "https://arxiv.org/abs/1706.03762"
        # "Attention Is All You Need"
        paper_query = f"add this paper from arxiv: {sample_arxiv_url}"

        ui.display_user_input(paper_query)
        intent_result = orchestrator.parse_intent(paper_query)
        result = orchestrator.execute_action(intent_result)

        if result.get("success") and result.get("result", {}).get("success"):
            extracted_topics = result["result"].get("extracted_topics", [])

            ui.print_divider()
            ui.print_info(
                "📚 Step 3: Listing papers by topic using natural language"
            )
            print()

            if extracted_topics:
                first_topic = extracted_topics[0]
                list_query = f"show me papers about {first_topic}"

                ui.display_user_input(list_query)
                intent_result = orchestrator.parse_intent(list_query)
                orchestrator.execute_action(intent_result)

                ui.print_divider()
                ui.print_info(
                    "📝 Step 4: Generating topic summary using natural language"
                )
                print()

                summary_query = f"summarize the topic {first_topic}"

                ui.display_user_input(summary_query)
                intent_result = orchestrator.parse_intent(summary_query)
                orchestrator.execute_action(intent_result)

        ui.print_divider()
        ui.print_success("Demo completed!")
        ui.print_info(
            "Run 'python agent.py' to start the interactive agent.\n"
        )

    except Exception as exc:
        ui.print_error(f"Demo failed: {exc}")

    finally:
        close_mcp_clients(clients)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Research Organizer Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the LLM model directory (optional)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU', 'AUTO'],
        help='Device to run LLM inference on (default: CPU)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        demo(model_path=args.model_path, device=args.device)
    except Exception as e:
        ui.print_error(f"Demo error: {str(e)}")
        import traceback
        traceback.print_exc()
