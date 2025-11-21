"""
Topic Agent for Research Organizer
Analyzes papers and suggests applicable topics using LLM.
"""

import json
from typing import Any

import openvino_genai as ov_genai

import ui
from tools.db import list_topics
from agents.llm import get_llm


class TopicAgent:
    """
    Agent that analyzes research papers and suggests applicable topics.
    Uses LLM to understand paper content and match it with available topics.
    """

    @staticmethod
    def get_structured_output_config(topics: list[str]) -> ov_genai.StructuredOutputConfig:
        return ov_genai.StructuredOutputConfig(
            json_schema=json.dumps({
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": topics
                        }
                    }
                },
                "required": ["topics"]
            })
        )

    # Prompt template
    PROMPT_TEMPLATE = """You are an expert research paper classifier. Your task is to analyze a research paper and suggest which topics from the available list are most applicable.

**Paper Information:**
Title: {title}

Abstract: {abstract}

**Available Topics:**
{topics_list}

**Instructions:**
1. Carefully read the paper's title and abstract
2. Consider each available topic and its description
3. Suggest ONLY the topics that are clearly relevant to this paper
4. If no topics are applicable, return an empty list
5. Focus on quality over quantity - only suggest topics you're confident about

Analyze the paper and respond with the applicable topics in JSON format."""
    
    def __init__(self):
        """Initialize the topic agent."""
        self.llm = get_llm()
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = 1024
        self.config.temperature = 0.3

    def extract_topics(
        self,
        title: str,
        abstract: str,
    ) -> dict[str, Any]:
        """
        Analyze a paper and suggest applicable topics.

        Args:
            title: Paper title
            abstract: Paper abstract
            available_topics: list of topic dicts with "name" and "description"

        Returns:
            dictionary with suggested topics and metadata
        """
        available_topics = list_topics()
        if not available_topics:
            return {
                "success": False,
                "message": "No topics available for classification",
                "suggested_topics": []
            }

        # Format topics list for the prompt
        topics_list = self._format_topics(available_topics)
        topics_names = [topic.get("name", "Unknown") for topic in available_topics]

        # Create the prompt
        prompt = self.PROMPT_TEMPLATE.format(
            title=title,
            abstract=abstract,
            topics_list=topics_list
        )
        try:
            self.config.structured_output_config = self.get_structured_output_config(topics_names)
            result = self.llm.generate([prompt], generation_config=self.config)
            response_text = result.texts[0]
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                extracted_topics = response_data.get("topics", [])
                return {
                    "success": True,
                    "extracted_topics": extracted_topics,
                }

            except json.JSONDecodeError as e:
                ui.print_error(f"[Topic Agent] Failed to parse LLM response: {e}")
                return {
                    "success": False,
                    "message": f"Failed to parse LLM response: {e}",
                    "raw_response": response_text,
                    "extracted_topics": []
                }

        except Exception as e:
            ui.print_error(f"[Topic Agent] Error during topic analysis: {e}")
            return {
                "success": False,
                "message": f"Error during topic analysis: {e}",
                "extracted_topics": []
            }
    
    @staticmethod
    def _format_topics(topics: list[dict[str, Any]]) -> str:
        """
        Format topics list for inclusion in prompt.
        
        Args:
            topics: list of topic dictionaries
        
        Returns:
            Formatted string of topics
        """
        formatted_lines = []
        for i, topic in enumerate(topics, 1):
            name = topic.get("name", "Unknown")
            description = topic.get("description", "No description")
            formatted_lines.append(
                f"{i}. **{name}**: {description}"
            )
        return "\n".join(formatted_lines)
