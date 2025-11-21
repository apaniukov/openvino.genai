"""
Summarizer Agent for Research Organizer
Generates intelligent summaries of papers using LLM.
"""

import json
from typing import Any

import openvino_genai as ov_genai

import ui
from agents.llm import get_llm


class SummarizerAgent:
    """
    Agent that generates comprehensive summaries of research papers.
    Uses LLM to synthesize information from multiple papers on a topic.
    """

    # JSON schema for structured summary output
    SCHEMA = {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": "Brief overview (2-3 sentences) of what these specific papers cover"
            },
            "key_findings": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Key findings from the provided papers (3-5 items)",
                "maxItems": 5
            },
            "papers_summary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Exact paper title from input"
                        },
                        "key_point": {
                            "type": "string",
                            "description": "One sentence main point (max 150 chars)"
                        }
                    },
                    "required": ["title", "key_point"]
                },
                "description": "Brief summary of each provided paper"
            }
        },
        "required": [
            "overview",
            "key_findings",
            "papers_summary"
        ]
    }

    # Prompt template
    PROMPT_TEMPLATE = """You are a research analyst. Summarize ONLY the papers listed below on topic "{topic_name}".

**Papers to Analyze ({num_papers} total):**
{papers_list}

**CRITICAL INSTRUCTIONS:**
1. Summarize ONLY the {num_papers} papers listed above - DO NOT mention or invent other papers
2. Use the EXACT titles from the list above
3. Write a brief overview (2-3 sentences) about what THESE papers cover
4. List 3-5 key findings from THESE specific papers
5. For each paper, write ONE concise sentence about its main point (max 150 characters)
6. Be factual and concise - stick to what's in the abstracts

Response format: JSON only, no additional text."""

    def __init__(self):
        """Initialize the summarizer agent."""
        self.llm = get_llm()

    def summarize_topic(
        self,
        papers: list[dict[str, Any]],
        topic_name: str = ""
    ) -> dict[str, Any]:
        """
        Generate a comprehensive summary of papers on a topic.

        Args:
            papers: List of paper dicts with 'title', 'abstract', etc.
            topic_name: Name of the topic being summarized

        Returns:
            Dictionary with structured summary
        """
        if not papers:
            return {
                "success": False,
                "message": f"No papers found for topic '{topic_name}'.",
                "summary": None
            }

        # Format papers for the prompt
        papers_list = self._format_papers(papers)

        # Create the prompt
        prompt = self.PROMPT_TEMPLATE.format(
            topic_name=topic_name or "Unknown Topic",
            num_papers=len(papers),
            papers_list=papers_list
        )

        try:
            # Generate with JSON schema constraint
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 1024  # Reduced for more concise output
            config.temperature = 0.3  # Lower for more focused output
            config.do_sample = False
            config.structured_output_config = (
                ov_genai.StructuredOutputConfig(
                    json_schema=json.dumps(self.SCHEMA)
                )
            )

            result = self.llm.generate([prompt], generation_config=config)
            response_text = result.texts[0]

            # Parse JSON response
            try:
                summary_data = json.loads(response_text)

                return {
                    "success": True,
                    "topic_name": topic_name,
                    "summary": summary_data,
                    "paper_count": len(papers)
                }

            except json.JSONDecodeError as e:
                ui.print_error(f"Failed to parse LLM summary: {e}")
                return {
                    "success": False,
                    "message": f"Failed to parse LLM response: {e}",
                    "raw_response": response_text,
                    "summary": None
                }

        except Exception as e:
            ui.print_error(f"Error during summarization: {e}")
            return {
                "success": False,
                "message": f"Error during summarization: {e}",
                "summary": None
            }

    @staticmethod
    def _format_papers(papers: list[dict[str, Any]]) -> str:
        """
        Format papers list for inclusion in prompt.

        Args:
            papers: List of paper dictionaries

        Returns:
            Formatted string of papers
        """
        formatted_lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "No abstract available")

            # Truncate very long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            formatted_lines.append(f"{i}. **{title}**")
            formatted_lines.append(f"   Abstract: {abstract}\n")

        return "\n".join(formatted_lines)

    @staticmethod
    def format_summary_for_display(
        summary_data: dict[str, Any]
    ) -> str:
        """
        Format structured summary for nice display.

        Args:
            summary_data: Summary dict from summarize_topic()

        Returns:
            Markdown-formatted string
        """
        if not summary_data:
            return "No summary available."

        lines = []

        # Overview
        if "overview" in summary_data:
            lines.append("## Overview")
            lines.append(summary_data["overview"])
            lines.append("")

        # Key Findings
        if "key_findings" in summary_data and summary_data["key_findings"]:
            lines.append("## Key Findings")
            for finding in summary_data["key_findings"]:
                lines.append(f"• {finding}")
            lines.append("")

        # Papers Summary
        if "papers_summary" in summary_data:
            lines.append("## Papers")
            for paper in summary_data["papers_summary"]:
                title = paper.get("title", "Unknown")
                key_point = paper.get("key_point", "N/A")
                lines.append(f"**{title}**")
                lines.append(f"  {key_point}")
                lines.append("")

        return "\n".join(lines)
