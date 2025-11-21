"""
Orchestrator Agent for Research Organizer
Parses user intent and routes to appropriate tools and agents.
"""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import openvino_genai as ov_genai

import ui
from agents.llm import get_llm
from mcp_tools.client import MCPClient
from .summarizer_agent import SummarizerAgent


class OrchestratorAgent:
    """
    Agent that understands user intent and orchestrates tool/agent calls.
    Uses LLM to parse natural language queries into structured actions.
    """

    INTENT_DETAILS = {
        "add_topic": {
            "description": "Add a new research topic.",
            "parameters": [
                "topic_name (required string)",
                "topic_description (optional string)",
            ],
            "tools": ["add_topic"],
        },
        "list_topics": {
            "description": "Show all stored topics.",
            "parameters": [],
            "tools": ["list_topics"],
        },
        "remove_topic": {
            "description": "Delete an existing topic.",
            "parameters": ["topic_name (required string)"],
            "tools": ["remove_topic"],
        },
        "add_paper": {
            "description": (
                "Fetch a paper from ArXiv, store it, and tag matching topics."
            ),
            "parameters": ["arxiv_url (required string)"],
            "tools": [
                "fetch_arxiv_paper",
                "add_paper",
                "list_topics",
                "extract_topics",
                "link_paper_to_topics",
            ],
        },
        "list_papers": {
            "description": "List every paper stored in the database.",
            "parameters": [],
            "tools": ["get_all_papers"],
        },
        "list_papers_by_topic": {
            "description": "Show papers tagged with a specific topic.",
            "parameters": ["topic_name (required string)"],
            "tools": ["get_papers_by_topic"],
        },
        "summarize_topic": {
            "description": "Generate a summary of papers for a topic.",
            "parameters": ["topic_name (required string)"],
            "tools": ["get_papers_by_topic", "summarize_topic"],
        },
    }

    SPECIAL_INTENT_DETAILS = {
        "help": "Show a help message with supported commands.",
        "exit": "Exit the application after confirming with the user.",
        "unclear": "Use when no action fits or the request is ambiguous.",
    }

    PARAMETER_PROPERTIES = {
        "topic_name": {
            "type": "string",
            "description": "Name of the topic",
        },
        "topic_description": {
            "type": "string",
            "description": "Description of the topic",
        },
        "arxiv_url": {
            "type": "string",
            "description": "ArXiv URL or ID",
        },
    }

    PROMPT_TEMPLATE = (
        "You are an intelligent assistant that understands user queries "
        "for a research paper management system.\n\n"
        "**User Query:** \"{user_input}\"\n\n"
        "**Available Actions:**\n"
        "{tool_descriptions}\n\n"
        "{special_descriptions}"
        "\n\n"
        "**Instructions:**\n"
        "1. Analyze the user query carefully\n"
        "2. Choose the most suitable intent from the list\n"
        "3. Extract parameters mentioned in the request\n"
        "4. Set confidence: high (very clear), medium (somewhat clear), "
        "low (ambiguous)\n"
        "5. Respond with strictly valid JSON that matches the schema\n"
        "6. Use 'unclear' only if no action fits\n"
    )

    def __init__(self, clients: Sequence[MCPClient] | None = None):
        """
        Initialize the orchestrator agent.

        Args:
            clients: Sequence of MCP clients to use for tool invocation.
        """
        self.llm = get_llm()
        if not self.llm:
            raise RuntimeError(
                "LLM not initialized. Call init_llm() first."
            )

        self.mcp_clients = list(clients or [])
        if not self.mcp_clients:
            raise ValueError("At least one MCP client is required.")

        self.tool_catalog = self._load_tool_catalog()

        self.intent_handlers = {
            "add_topic": self._handle_add_topic,
            "list_topics": self._handle_list_topics,
            "remove_topic": self._handle_remove_topic,
            "add_paper": self._handle_add_paper,
            "list_papers": self._handle_list_papers,
            "list_papers_by_topic": self._handle_list_papers_by_topic,
            "summarize_topic": self._handle_summarize_topic,
        }

        self.supported_intents = self._resolve_supported_intents()
        schema = self._build_schema(self.supported_intents)

        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = 256
        self.config.temperature = 0.1
        self.config.do_sample = False
        self.config.structured_output_config = (
            ov_genai.StructuredOutputConfig(json_schema=json.dumps(schema))
        )

    def _resolve_supported_intents(self) -> list[str]:
        supported: list[str] = []
        for intent in self.INTENT_DETAILS:
            if intent not in self.intent_handlers:
                continue
            required_tools = self.INTENT_DETAILS[intent].get("tools", [])
            missing = [
                name
                for name in required_tools
                if name not in self.tool_catalog
            ]
            if missing:
                ui.print_warning(
                    f"Disabling intent '{intent}' "
                    f"(missing MCP tools: {', '.join(missing)})."
                )
                continue
            supported.append(intent)

        supported.extend(sorted(self.SPECIAL_INTENT_DETAILS))
        return supported

    def _build_schema(self, intents: Sequence[str]) -> dict[str, Any]:
        unique_intents = list(dict.fromkeys(intents))
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": unique_intents,
                    "description": "The user's primary intent",
                },
                "parameters": {
                    "type": "object",
                    "properties": self.PARAMETER_PROPERTIES,
                    "description": "Parameters extracted from user input",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence in intent classification",
                },
            },
            "required": ["intent", "parameters", "confidence"],
        }

    def _load_tool_catalog(self) -> dict[str, dict[str, Any]]:
        catalog: dict[str, dict[str, Any]] = {}
        for client in self.mcp_clients:
            try:
                tools = self._run_async(self._list_tools_async, client)
            except Exception as exc:
                ui.print_warning(
                    "Failed to load tools from "
                    f"{self._client_label(client)}: {exc}"
                )
                continue

            if not tools:
                continue

            for tool in tools:
                if tool.name in catalog:
                    ui.print_warning(
                        f"Duplicate MCP tool '{tool.name}' detected; "
                        "keeping the first registration."
                    )
                    continue
                catalog[tool.name] = {"client": client, "tool": tool}

        if not catalog:
            raise RuntimeError(
                "No MCP tools discovered from provided clients."
            )

        return catalog

    async def _list_tools_async(self, client: MCPClient):
        async with client:
            return await client.list_tools()

    def _client_label(self, client: MCPClient) -> str:
        config = getattr(client, "_config", None)
        if config is not None and getattr(config, "name", None):
            return str(config.name)
        return client.__class__.__name__

    def _run_async(
        self,
        coroutine_fn,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async def runner():
            return await coroutine_fn(*args, **kwargs)

        try:
            return asyncio.run(runner())
        except RuntimeError as exc:
            message = (
                "asyncio.run() cannot be called from a running event loop"
            )
            if message not in str(exc):
                raise
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coroutine_fn(*args, **kwargs))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        binding = self.tool_catalog.get(tool_name)
        if not binding:
            raise RuntimeError(f"Tool '{tool_name}' is not available.")
        client: MCPClient = binding["client"]

        async def caller():
            async with client:
                return await client.call_tool(tool_name, arguments)

        return self._run_async(caller)

    def _compose_prompt(self, user_input: str) -> str:
        tool_section = self._build_tool_section()
        special_section = self._build_special_section()
        prompt = self.PROMPT_TEMPLATE.format(
            user_input=user_input,
            tool_descriptions=tool_section,
            special_descriptions=special_section,
        )
        return prompt.rstrip()

    def _build_tool_section(self) -> str:
        lines: list[str] = []
        for intent, info in self.INTENT_DETAILS.items():
            if intent not in self.supported_intents:
                continue

            params_desc = (
                ", ".join(info["parameters"]) if info["parameters"] else "none"
            )

            tool_descs: list[str] = []
            for tool_name in info.get("tools", []):
                binding = self.tool_catalog.get(tool_name)
                if not binding:
                    continue
                description = binding["tool"].description or "No description"
                tool_descs.append(f"{tool_name}: {description}")

            tools_segment = (
                "; ".join(tool_descs)
                if tool_descs
                else "No MCP tool registered"
            )

            lines.append(
                (
                    f"- {intent}: {info['description']} "
                    f"(Parameters: {params_desc}) "
                    f"(MCP tools: {tools_segment})"
                )
            )

        if not lines:
            return "- No MCP-backed actions available."

        return "\n".join(lines)

    def _build_special_section(self) -> str:
        lines: list[str] = []
        for intent, description in self.SPECIAL_INTENT_DETAILS.items():
            if intent not in self.supported_intents:
                continue
            lines.append(f"- {intent}: {description}")

        if not lines:
            return ""

        return "**Special Commands:**\n" + "\n".join(lines)

    def parse_intent(
        self,
        user_input: str
    ) -> dict[str, Any]:
        if not user_input or not user_input.strip():
            return {
                "success": False,
                "message": "Empty input",
                "intent": "unclear",
                "parameters": {},
                "confidence": "low",
            }

        prompt = self._compose_prompt(user_input)

        try:
            result = self.llm.generate([prompt], generation_config=self.config)
            response_text = result.texts[0]  # type: ignore[attr-defined]

            try:
                intent_data = json.loads(response_text)
                return {
                    "success": True,
                    "intent": intent_data.get("intent", "unclear"),
                    "parameters": intent_data.get("parameters", {}),
                    "confidence": intent_data.get("confidence", "low"),
                }
            except json.JSONDecodeError as exc:
                ui.print_error(f"Failed to parse intent: {exc}")
                return {
                    "success": False,
                    "message": f"Failed to parse LLM response: {exc}",
                    "raw_response": response_text,
                    "intent": "unclear",
                    "parameters": {},
                    "confidence": "low",
                }

        except Exception as exc:
            ui.print_error(f"Error during intent parsing: {exc}")
            return {
                "success": False,
                "message": f"Error during intent parsing: {exc}",
                "intent": "unclear",
                "parameters": {},
                "confidence": "low",
            }

    def explain_intent(
        self,
        intent_result: dict[str, Any]
    ) -> str:
        if not intent_result.get("success"):
            return "Could not understand the request."

        intent = intent_result.get("intent", "unclear")
        params = intent_result.get("parameters", {})
        confidence = intent_result.get("confidence", "low")

        explanations = {
            "add_topic": (
                "Adding topic "
                f"'{params.get('topic_name', 'unknown')}'"
            ),
            "list_topics": "Listing all topics",
            "remove_topic": (
                "Removing topic "
                f"'{params.get('topic_name', 'unknown')}'"
            ),
            "add_paper": (
                "Adding paper from "
                f"{params.get('arxiv_url', 'URL')}"
            ),
            "list_papers": "Listing all papers",
            "list_papers_by_topic": (
                "Listing papers for topic "
                f"'{params.get('topic_name', 'unknown')}'"
            ),
            "summarize_topic": (
                "Summarizing topic "
                f"'{params.get('topic_name', 'unknown')}'"
            ),
            "help": "Showing help",
            "exit": "Exiting application",
            "unclear": "Request is unclear",
        }

        explanation = explanations.get(intent, "Unknown action")

        if confidence == "low":
            explanation += " (low confidence - please verify)"

        return explanation

    def execute_action(
        self,
        intent_result: dict[str, Any]
    ) -> dict[str, Any]:
        if not intent_result.get("success"):
            return {
                "success": False,
                "message": "Cannot execute - intent parsing failed",
            }

        intent = intent_result.get("intent", "unclear")
        params = intent_result.get("parameters", {})
        confidence = intent_result.get("confidence", "low")

        if intent == "unclear":
            ui.print_warning("I'm not sure what you want to do.")
            ui.print_info(
                "Could you try rephrasing? Type 'help' for commands."
            )
            return {
                "success": False,
                "message": "Unclear intent",
            }

        if intent == "help":
            ui.print_help()
            return {
                "success": True,
                "intent": intent,
                "result": {"message": "Help displayed"},
            }

        if intent == "exit":
            ui.print_info("Exit requested.")
            return {
                "success": True,
                "intent": intent,
                "result": {"message": "Exit requested"},
            }

        if intent not in self.intent_handlers:
            ui.print_error(f"No handler found for intent: {intent}")
            return {
                "success": False,
                "message": f"No handler for intent: {intent}",
            }

        explanation = self.explain_intent(intent_result)

        if confidence == "low":
            ui.print_warning(f"I think you want to: {explanation}")
            if not ui.confirm_action("Is this correct?"):
                ui.print_info("Cancelled. Please try again.")
                return {
                    "success": False,
                    "message": "User cancelled low-confidence action",
                }
        elif confidence == "medium":
            ui.print_info(f"Understanding: {explanation}")

        handler = self.intent_handlers[intent]

        try:
            result = handler(params)
        except Exception as exc:
            ui.print_error(f"Error executing action '{intent}': {exc}")
            return {
                "success": False,
                "message": f"Execution error: {exc}",
            }

        return {
            "success": True,
            "intent": intent,
            "result": result,
        }

    def _handle_add_topic(self, params: dict[str, Any]) -> dict[str, Any]:
        topic_name = params.get("topic_name", "").strip()
        topic_desc = params.get("topic_description", "").strip()

        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Missing topic name"}

        payload = {"name": topic_name, "description": topic_desc}

        try:
            result = self._call_tool("add_topic", payload)
        except Exception as exc:
            ui.print_error(f"Failed to add topic: {exc}")
            return {"success": False, "message": str(exc)}

        if isinstance(result, dict) and result.get("success"):
            ui.print_success(f"Added topic: {topic_name}")
            return result

        message = "Unknown error"
        if isinstance(result, dict):
            message = result.get("message", message)
            ui.print_error(f"Failed to add topic: {message}")
            return result

        ui.print_error("Unexpected response from add_topic tool.")
        return {"success": False, "message": "Unexpected MCP response"}

    def _handle_list_topics(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        try:
            topics = self._call_tool("list_topics", {})
        except Exception as exc:
            ui.print_error(f"Failed to list topics: {exc}")
            return {"success": False, "message": str(exc), "topics": []}

        if not isinstance(topics, list):
            ui.print_error("Unexpected response from list_topics tool.")
            return {
                "success": False,
                "message": "Unexpected MCP response",
                "topics": [],
            }

        ui.display_topics_table(topics)
        return {"success": True, "topics": topics}

    def _handle_remove_topic(self, params: dict[str, Any]) -> dict[str, Any]:
        topic_name = params.get("topic_name", "").strip()

        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Missing topic_name"}

        if not ui.confirm_action(f"Remove topic '{topic_name}'?"):
            ui.print_info("Cancelled.")
            return {"success": False, "message": "Cancelled by user"}

        try:
            result = self._call_tool("remove_topic", {"name": topic_name})
        except Exception as exc:
            ui.print_error(f"Failed to remove topic: {exc}")
            return {"success": False, "message": str(exc)}

        if isinstance(result, dict) and result.get("success"):
            ui.print_success(
                result.get("message", f"Removed topic '{topic_name}'.")
            )
            return result

        message = "Unknown error"
        if isinstance(result, dict):
            message = result.get("message", message)
            ui.print_error(f"Failed to remove topic: {message}")
            return result

        ui.print_error("Unexpected response from remove_topic tool.")
        return {"success": False, "message": "Unexpected MCP response"}

    def _handle_list_papers(self, params: dict[str, Any]) -> dict[str, Any]:
        topic_name = params.get("topic_name", "").strip()

        if topic_name:
            return self._handle_list_papers_by_topic(
                {"topic_name": topic_name}
            )

        try:
            papers = self._call_tool("get_all_papers", {})
        except Exception as exc:
            ui.print_error(f"Failed to list papers: {exc}")
            return {"success": False, "message": str(exc)}

        if not isinstance(papers, list):
            ui.print_error("Unexpected response from get_all_papers tool.")
            return {"success": False, "message": "Unexpected MCP response"}

        ui.display_papers_table(papers)
        return {"success": True, "papers": papers}

    def _handle_list_papers_by_topic(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        topic_name = params.get("topic_name", "").strip()

        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Missing topic_name"}

        try:
            papers = self._call_tool(
                "get_papers_by_topic",
                {"topic_name": topic_name},
            )
        except Exception as exc:
            ui.print_error(f"Failed to list papers: {exc}")
            return {"success": False, "message": str(exc)}

        if not isinstance(papers, list):
            ui.print_error(
                "Unexpected response from get_papers_by_topic tool."
            )
            return {"success": False, "message": "Unexpected MCP response"}

        if not papers:
            ui.print_info(f"No papers found for topic: {topic_name}")

        ui.display_papers_table(papers, topic_name)
        return {"success": True, "papers": papers, "topic": topic_name}

    def _handle_add_paper(self, params: dict[str, Any]) -> dict[str, Any]:
        arxiv_url = params.get("arxiv_url", "").strip()

        if not arxiv_url:
            ui.print_error("ArXiv URL is required.")
            return {"success": False, "message": "Missing arxiv_url"}

        ui.print_info("Fetching paper from ArXiv...")
        try:
            paper_data = self._call_tool(
                "fetch_arxiv_paper",
                {"arxiv_url": arxiv_url},
            )
        except Exception as exc:
            ui.print_error(f"Failed to fetch paper: {exc}")
            return {"success": False, "message": str(exc)}

        if not isinstance(paper_data, dict) or not paper_data.get("success"):
            message = "Unknown error"
            if isinstance(paper_data, dict):
                message = paper_data.get("message", message)
            ui.print_error(f"Failed to fetch paper: {message}")
            return (
                paper_data
                if isinstance(paper_data, dict)
                else {"success": False, "message": message}
            )

        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")
        canonical_url = paper_data.get("arxiv_url", arxiv_url)

        ui.print_success(f"Retrieved: {title}")

        ui.print_info("Adding paper to database...")
        try:
            add_result = self._call_tool(
                "add_paper",
                {
                    "title": title,
                    "arxiv_url": canonical_url,
                    "abstract": abstract,
                },
            )
        except Exception as exc:
            ui.print_error(f"Failed to add paper: {exc}")
            return {"success": False, "message": str(exc)}

        if not isinstance(add_result, dict) or not add_result.get("success"):
            message = "Unknown error"
            if isinstance(add_result, dict):
                message = add_result.get("message", message)
            ui.print_error(f"Failed to add paper: {message}")
            return (
                add_result
                if isinstance(add_result, dict)
                else {"success": False, "message": message}
            )

        paper_id = add_result.get("paper_id")
        ui.print_success(f"Added paper to database (ID: {paper_id})")

        extracted_topics: list[str] = []
        linked_count = 0

        if "extract_topics" in self.tool_catalog:
            ui.print_info("🔍 Extracting paper topics...")
            try:
                extraction = self._call_tool(
                    "extract_topics",
                    {"title": title, "abstract": abstract},
                )
            except Exception as exc:
                ui.print_warning(f"Topic extraction failed: {exc}")
            else:
                if isinstance(extraction, dict) and extraction.get("success"):
                    extracted_topics = (
                        extraction.get("extracted_topics", []) or []
                    )
                    if extracted_topics:
                        ui.print_info(
                            f"Detected topics: {', '.join(extracted_topics)}"
                        )
                        linked_count = self._link_paper_to_topics(
                            paper_id,
                            extracted_topics,
                        )
                elif isinstance(extraction, dict):
                    ui.print_warning(
                        extraction.get("message", "Topic extraction failed.")
                    )
        else:
            ui.print_info("Topic extraction tool not available.")

        ui.display_article_card(
            title=title,
            abstract=abstract,
            topics=extracted_topics,
        )

        if extracted_topics and linked_count == 0:
            ui.print_info(
                "Extracted topics did not match existing topics."
            )
        elif not extracted_topics:
            ui.print_info("No matching topics found.")

        return {
            "success": True,
            "paper_id": paper_id,
            "paper_data": paper_data,
            "extracted_topics": extracted_topics,
        }

    def _link_paper_to_topics(
        self,
        paper_id: Any,
        topic_names: list[str],
    ) -> int:
        if paper_id is None or not topic_names:
            return 0

        if "link_paper_to_topics" not in self.tool_catalog:
            ui.print_info("Linking tool not available.")
            return 0

        try:
            topics = self._call_tool("list_topics", {})
        except Exception as exc:
            ui.print_warning(f"Unable to list topics for linking: {exc}")
            return 0

        if not isinstance(topics, list):
            ui.print_warning("Unexpected response while listing topics.")
            return 0

        if not topics:
            ui.print_info("No topics available. Paper stored without links.")
            return 0

        topic_map = {
            topic.get("name"): topic.get("id")
            for topic in topics
            if topic.get("name") is not None
        }
        topic_ids = [
            topic_map[name]
            for name in topic_names
            if name in topic_map
        ]

        if not topic_ids:
            ui.print_info("No matching topics found for extracted labels.")
            return 0

        try:
            link_result = self._call_tool(
                "link_paper_to_topics",
                {"paper_id": paper_id, "topic_ids": topic_ids},
            )
        except Exception as exc:
            ui.print_warning(f"Failed to link topics: {exc}")
            return 0

        if isinstance(link_result, dict) and link_result.get("success"):
            ui.print_success(f"Linked to {len(topic_ids)} topic(s)")
            return len(topic_ids)

        if isinstance(link_result, dict):
            ui.print_warning(
                link_result.get("message", "Failed to link topics.")
            )
        else:
            ui.print_warning("Unexpected response from linking tool.")
        return 0

    def _handle_summarize_topic(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        topic_name = params.get("topic_name", "").strip()

        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Missing topic_name"}

        try:
            papers = self._call_tool(
                "get_papers_by_topic",
                {"topic_name": topic_name},
            )
        except Exception as exc:
            ui.print_error(f"Failed to find papers: {exc}")
            return {"success": False, "message": str(exc)}

        if not isinstance(papers, list):
            ui.print_error(
                "Unexpected response from get_papers_by_topic tool."
            )
            return {"success": False, "message": "Unexpected MCP response"}

        if not papers:
            ui.print_info(f"No papers found for topic: {topic_name}")
            return {"success": False, "message": "No papers found"}

        ui.print_info(f"Generating summary for {len(papers)} paper(s)...")
        try:
            summary_result = self._call_tool(
                "summarize_topic",
                {"papers": papers, "topic_name": topic_name},
            )
        except Exception as exc:
            ui.print_error(f"Summarization error: {exc}")
            return {"success": False, "message": str(exc)}

        if isinstance(summary_result, dict) and summary_result.get("success"):
            formatted = SummarizerAgent.format_summary_for_display(
                summary_result.get("summary", {})
            )
            ui.display_markdown(formatted)
            return {
                "success": True,
                "summary": summary_result.get("summary"),
                "topic": topic_name,
            }

        if isinstance(summary_result, dict):
            ui.print_error(
                summary_result.get("message", "Summarization failed.")
            )
            return summary_result

        ui.print_error("Unexpected response from summarize_topic tool.")
        return {"success": False, "message": "Unexpected MCP response"}
