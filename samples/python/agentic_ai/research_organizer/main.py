#!/usr/bin/env python3
"""
Research Organizer Agent
Main orchestrator for the research assistant with MCP tool integration.
"""

import sys
import argparse

# Import tool implementations
from tools import (
    init_database, add_topic, list_topics, remove_topic,
    add_paper, link_paper_to_topics, get_papers_by_topic,
    get_all_papers, fetch_arxiv_paper,
)

# Import agents infrastructure
from agents import (
    init_llm,
    is_llm_initialized,
    TopicAgent,
    SummarizerAgent,
    OrchestratorAgent
)

# Import UI utilities
import ui


class ResearchAgent:
    """Main agent class for orchestrating research organization tasks."""
    
    def __init__(self, model_path=None, device="CPU"):
        """
        Initialize the research agent.
        
        Args:
            model_path: Optional path to LLM model directory
            device: Device for LLM inference (CPU, GPU, AUTO)
        """
        # Initialize database on startup
        init_database()
        self.running = True
        self.orchestrator = None
        
        # Initialize LLM if model path provided
        if model_path and not is_llm_initialized():
            try:
                ui.print_info(f"Initializing LLM from {model_path}...")
                init_llm(model_path, device=device)
                ui.print_success("LLM initialized successfully")
                
                # Initialize orchestrator for natural language understanding
                try:
                    # Build action handlers map
                    action_handlers = {
                        "add_topic": self.handle_add_topic_from_params,
                        "list_topics": lambda p: self.handle_list_topics(),
                        "remove_topic": self.handle_remove_topic_from_params,
                        "add_paper": self.handle_add_paper_from_params,
                        "list_papers": lambda p: self.handle_list_papers(None),
                        "list_papers_by_topic": self.handle_list_papers_by_topic,
                        "summarize_topic": self.handle_summarize_topic_from_params,
                        "help": lambda p: ui.print_help(),
                        "exit": lambda p: self.handle_exit(),
                    }
                    
                    self.orchestrator = OrchestratorAgent(action_handlers)
                    ui.print_success("Natural language mode enabled")
                except Exception as e:
                    ui.print_warning(f"Orchestrator init failed: {e}")
                    
            except Exception as e:
                ui.print_warning(
                    f"Failed to initialize LLM: {e}"
                )
                ui.print_info(
                    "Continuing without LLM features..."
                )
        else:
            ui.print_info(
                "No model path provided. "
                "LLM features will be disabled."
            )
            ui.print_info(
                "Use --model-path to enable AI features."
            )
    
    def run(self):
        """Main chat loop."""
        # Show appropriate welcome message
        if self.orchestrator:
            ui.print_welcome_with_nlp()
        else:
            ui.print_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = ui.get_user_input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Process command
                self.process_command(user_input)
                
            except KeyboardInterrupt:
                ui.print_info("\nUse 'exit' or 'quit' to exit gracefully.")
            except Exception as e:
                ui.print_error(f"Unexpected error: {str(e)}")
    
    def process_command(self, command: str):
        """
        Process user command and route to appropriate handler.
        Uses orchestrator for natural language if available,
        falls back to pattern matching.
        
        Args:
            command: User command string
        """
        # Try using orchestrator for natural language understanding
        if self.orchestrator:
            try:
                intent_result = self.orchestrator.parse_intent(command)
                
                if intent_result.get("success"):
                    # Execute the action (orchestrator has handlers configured)
                    self.orchestrator.execute_action(intent_result)
                    return
                    
            except Exception as e:
                ui.print_warning(
                    f"Natural language processing failed: {e}"
                )
                ui.print_info("Falling back to command parsing...")
        
        # Fallback to traditional command parsing
        command_lower = command.lower()
        
        # Exit commands
        if command_lower in ['exit', 'quit']:
            self.handle_exit()
        
        # Help command
        elif command_lower == 'help':
            ui.print_help()
        
        # Add topic
        elif command_lower.startswith('add topic '):
            self.handle_add_topic(command)
        
        # List topics
        elif command_lower == 'list topics':
            self.handle_list_topics()
        
        # Remove topic
        elif command_lower.startswith('remove topic '):
            self.handle_remove_topic(command)
        
        # Add paper
        elif command_lower.startswith('add paper '):
            self.handle_add_paper(command)
        
        # List papers
        elif command_lower.startswith('list papers'):
            self.handle_list_papers(command)
        
        # Summarize topic
        elif command_lower.startswith('summarize topic '):
            self.handle_summarize_topic(command)
        # Unknown command
        else:
            ui.print_warning(f"Unknown command: {command}")
            ui.print_info("Type 'help' to see available commands.")
    
    def handle_exit(self):
        """Handle exit command."""
        if ui.confirm_action("Are you sure you want to exit?"):
            ui.print_info("Goodbye! 👋")
            self.running = False
    
    def handle_add_topic(self, command: str):
        """Handle add topic command."""
        # Parse: add topic <name> [description]
        parts = command[10:].strip().split(maxsplit=1)
        
        if not parts:
            ui.print_error("Please provide a topic name.")
            ui.print_info("Usage: add topic <name> [description]")
            return
        
        name = parts[0]
        description = parts[1] if len(parts) > 1 else ""
        
        ui.print_agent_action(f"Adding topic '{name}'...")
        
        result = add_topic(name, description)
        
        if result['success']:
            ui.print_success(result['message'])
        else:
            ui.print_error(result['message'])
    
    def handle_list_topics(self):
        """Handle list topics command."""
        ui.print_agent_action("Retrieving topics...")
        
        topics = list_topics()
        ui.display_topics_table(topics)
    
    def handle_remove_topic(self, command: str):
        """Handle remove topic command."""
        # Parse: remove topic <name>
        name = command[13:].strip()
        
        if not name:
            ui.print_error("Please provide a topic name to remove.")
            return
        
        if not ui.confirm_action(f"Remove topic '{name}'?"):
            ui.print_info("Cancelled.")
            return
        
        ui.print_agent_action(f"Removing topic '{name}'...")
        
        result = remove_topic(name)
        
        if result['success']:
            ui.print_success(result['message'])
        else:
            ui.print_error(result['message'])
    
    def handle_add_paper(self, command: str):
        """Handle add paper command with automatic topic tagging."""
        # Parse: add paper <arxiv_url>
        arxiv_url = command[10:].strip()
        
        if not arxiv_url:
            ui.print_error("Please provide an ArXiv URL.")
            ui.print_info("Usage: add paper <arxiv_url>")
            return
        
        # Step 1: Fetch paper from ArXiv
        ui.print_fetching("Fetching paper from ArXiv...")
        
        paper_data = fetch_arxiv_paper(arxiv_url)
        
        if not paper_data['success']:
            ui.print_error(paper_data['message'])
            return
        
        # Step 2: Add paper to database
        ui.print_agent_action("Adding paper to database...")
        
        result = add_paper(
            paper_data['title'],
            paper_data['arxiv_url'],
            paper_data['abstract']
        )
        
        if not result['success']:
            ui.print_error(result['message'])
            return
        
        paper_id = result['paper_id']
        
        # Step 3: Classify paper to topics
        ui.print_analyzing("Analyzing paper for topic classification...")
        
        topics = list_topics()
        matched_topic_names = []
        
        if topics:
            # Use LLM-based TopicAgent if available, fallback to keyword matching
            if is_llm_initialized():
                try:
                    topic_agent = TopicAgent()
                    analysis = topic_agent.extract_topics(
                        title=paper_data['title'],
                        abstract=paper_data['abstract'],
                        available_topics=topics
                    )
                    
                    if analysis['success'] and analysis['suggested_topics']:
                        # Display suggestions with confidence levels
                        ui.print_info("\nLLM Topic Analysis:")
                        for suggestion in analysis['suggested_topics']:
                            conf = suggestion['confidence']
                            name = suggestion['topic_name']
                            reason = suggestion['reasoning']
                            
                            conf_emoji = {
                                'high': '🟢',
                                'medium': '🟡',
                                'low': '🟠'
                            }.get(conf, '⚪')
                            
                            ui.print_info(
                                f"  {conf_emoji} {name} ({conf}): {reason}"
                            )
                        
                        # Get topic IDs for high confidence matches
                        high_conf_names = topic_agent.get_high_confidence_topics(analysis)
                        
                        if high_conf_names:
                            # Find topic IDs
                            topic_map = {t['name']: t['id'] for t in topics}
                            topic_ids = [
                                topic_map[name] 
                                for name in high_conf_names 
                                if name in topic_map
                            ]
                            
                            if topic_ids:
                                link_paper_to_topics(paper_id, topic_ids)
                                matched_topic_names = high_conf_names
                        else:
                            ui.print_info(
                                "\nNo high-confidence matches. "
                                "Paper not auto-tagged."
                            )
                    
                except Exception as e:
                    ui.print_warning(f"LLM classification failed: {e}")
                    ui.print_info("Falling back to keyword matching...")
                    
                    # Fallback to keyword matching
                    classification = classify_paper(
                        paper_data['abstract'], 
                        topics
                    )
                    
                    if classification['success'] and classification['matched_topic_ids']:
                        link_paper_to_topics(
                            paper_id,
                            classification['matched_topic_ids']
                        )
                        matched_topic_names = [
                            t['name'] 
                            for t in classification['matched_topics']
                        ]
            else:
                # Use keyword-based classification
                classification = classify_paper(paper_data['abstract'], topics)
                
                if classification['success'] and classification['matched_topic_ids']:
                    link_paper_to_topics(
                        paper_id,
                        classification['matched_topic_ids']
                    )
                    matched_topic_names = [
                        t['name'] 
                        for t in classification['matched_topics']
                    ]
        
        # Step 4: Display article card with all information
        ui.display_article_card(
            title=paper_data['title'],
            abstract=paper_data['abstract'],
            topics=matched_topic_names if matched_topic_names else None
        )
        
        if not matched_topic_names and topics:
            ui.print_info("No matching topics found. Consider adding relevant topics.")
        elif not topics:
            ui.print_info("No topics available. Add topics to enable auto-tagging.")
    
    def handle_list_papers(self, command: str):
        """Handle list papers command."""
        # Parse: list papers [topic_name]
        parts = command[11:].strip()
        
        if parts:
            # List papers for specific topic
            topic_name = parts
            ui.print_agent_action(f"Retrieving papers for topic '{topic_name}'...")
            
            papers = get_papers_by_topic(topic_name)
            ui.display_papers_table(papers, topic_name)
        else:
            # List all papers
            ui.print_agent_action("Retrieving all papers...")
            
            papers = get_all_papers()
            ui.display_papers_table(papers)
    
    def handle_summarize_topic(self, command: str):
        """Handle summarize topic command."""
        # Parse: summarize topic <name>
        topic_name = command[16:].strip()
        
        if not topic_name:
            ui.print_error("Please provide a topic name.")
            return
        
        ui.print_agent_action(
            f"Generating summary for topic '{topic_name}'..."
        )
        
        # Get papers for this topic
        papers = get_papers_by_topic(topic_name)
        
        if not papers:
            ui.print_info(f"No papers found for topic '{topic_name}'.")
            return
        
        # Use LLM-based SummarizerAgent if available
        if is_llm_initialized():
            try:
                summarizer = SummarizerAgent()
                summary_result = summarizer.summarize_topic(
                    papers,
                    topic_name
                )
                
                if summary_result['success']:
                    # Format and display the structured summary
                    formatted_summary = (
                        SummarizerAgent.format_summary_for_display(
                            summary_result['summary']
                        )
                    )
                    ui.print_divider()
                    ui.display_markdown(formatted_summary)
                    ui.print_divider()
                else:
                    ui.print_error(summary_result.get('message', 'Failed'))
                
            except Exception as e:
                ui.print_warning(f"LLM summarization failed: {e}")
                ui.print_info("Falling back to simple summarization...")
                
                # Fallback to simple summarization
                summary_result = summarize_papers(papers, topic_name)
                
                if summary_result['success']:
                    ui.print_divider()
                    ui.display_markdown(summary_result['summary'])
                    ui.print_divider()
                else:
                    ui.print_error(summary_result['message'])
        else:
            # Use simple extractive summarization
            summary_result = summarize_papers(papers, topic_name)
            
            if summary_result['success']:
                ui.print_divider()
                ui.display_markdown(summary_result['summary'])
                ui.print_divider()
            else:
                ui.print_error(summary_result['message'])
    
    # ========== Parameter-based handlers for orchestrator ==========
    
    def handle_add_topic_from_params(self, params: dict):
        """Handle add topic from orchestrator parameters."""
        name = params.get("topic_name", "").strip()
        description = params.get("topic_description", "").strip()
        
        if not name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Topic name required"}
        
        ui.print_agent_action(f"Adding topic '{name}'...")
        result = add_topic(name, description)
        
        if result['success']:
            ui.print_success(result['message'])
        else:
            ui.print_error(result['message'])
        
        return result
    
    def handle_remove_topic_from_params(self, params: dict):
        """Handle remove topic from orchestrator parameters."""
        name = params.get("topic_name", "").strip()
        
        if not name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Topic name required"}
        
        if not ui.confirm_action(f"Remove topic '{name}'?"):
            ui.print_info("Cancelled.")
            return {"success": False, "message": "Cancelled by user"}
        
        ui.print_agent_action(f"Removing topic '{name}'...")
        result = remove_topic(name)
        
        if result['success']:
            ui.print_success(result['message'])
        else:
            ui.print_error(result['message'])
        
        return result
    
    def handle_add_paper_from_params(self, params: dict):
        """Handle add paper from orchestrator parameters."""
        arxiv_url = params.get("arxiv_url", "").strip()
        
        if not arxiv_url:
            ui.print_error("ArXiv URL is required.")
            return {"success": False, "message": "ArXiv URL required"}
        
        # Use existing handle_add_paper logic
        self.handle_add_paper(f"add paper {arxiv_url}")
        return {"success": True}
    
    def handle_list_papers_by_topic(self, params: dict):
        """Handle list papers by topic from orchestrator parameters."""
        topic_name = params.get("topic_name", "").strip()
        
        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Topic name required"}
        
        ui.print_agent_action(
            f"Retrieving papers for topic '{topic_name}'..."
        )
        papers = get_papers_by_topic(topic_name)
        ui.display_papers_table(papers, topic_name)
        return {"success": True, "count": len(papers)}
    
    def handle_summarize_topic_from_params(self, params: dict):
        """Handle summarize topic from orchestrator parameters."""
        topic_name = params.get("topic_name", "").strip()
        
        if not topic_name:
            ui.print_error("Topic name is required.")
            return {"success": False, "message": "Topic name required"}
        
        # Use existing handle_summarize_topic logic
        self.handle_summarize_topic(f"summarize topic {topic_name}")
        return {"success": True}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research Organizer - AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run without LLM
  python agent.py
  
  # Run with LLM
  python agent.py --model-path /path/to/model
  
  # Run with LLM on GPU
  python agent.py --model-path /path/to/model --device GPU
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the LLM model directory (OpenVINO IR format)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU', 'AUTO'],
        help='Device to run LLM inference on (default: CPU)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        agent = ResearchAgent(
            model_path=args.model_path,
            device=args.device
        )
        agent.run()
    except Exception as e:
        ui.print_error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
