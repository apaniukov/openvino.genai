"""
UI Utilities for Research Organizer
Provides rich-based console interface helpers for styled terminal output.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box


# Global console instance
console = Console()


def print_header():
    """Print the application header."""
    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         📚 Research Organizer - AI Research Assistant        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(header, style="bold cyan")


def print_welcome():
    """Print welcome message with usage instructions."""
    print_header()
    
    welcome_text = """
## Welcome to Research Organizer! 🤖

An AI-powered research assistant that helps you manage and explore research papers.

### Available Commands:
- `add topic <name> [description]` - Add a new research topic
- `list topics` - Show all topics
- `remove topic <name>` - Remove a topic
- `add paper <arxiv_url>` - Add a paper from ArXiv
- `list papers [topic]` - List all papers or papers for a specific topic
- `summarize topic <name>` - Get a summary of papers on a topic
- `help` - Show this help message
- `exit` or `quit` - Exit the application

Type your command to get started!
    """
    
    console.print(Markdown(welcome_text))


def print_welcome_with_nlp():
    """Print welcome message with natural language support."""
    print_header()
    
    welcome_text = """
## Welcome to Research Organizer! 🤖✨

An AI-powered research assistant with **natural language understanding**.

### 💬 Talk Naturally:
- "add machine learning as a topic"
- "show me papers about neural networks"
- "summarize deep learning papers"
- "add this paper: https://arxiv.org/abs/1706.03762"

### Or Use Commands:
- `add topic <name> [description]` - Add a new research topic
- `list topics` - Show all topics
- `remove topic <name>` - Remove a topic
- `add paper <arxiv_url>` - Add a paper from ArXiv
- `list papers [topic]` - List all papers or papers for a specific topic
- `summarize topic <name>` - Get a summary of papers on a topic
- `insights topic <name>` - Get insights about a topic
- `help` - Show this help message
- `exit` or `quit` - Exit the application

Type naturally or use commands - I understand both!
    """
    
    console.print(Markdown(welcome_text))


def print_help():
    """Print help information."""
    help_text = """
## Research Organizer Commands

### Topic Management
- **add topic** `<name>` `[description]` - Add a new research topic
- **list topics** - Display all available topics
- **remove topic** `<name>` - Delete a topic

### Paper Management
- **add paper** `<arxiv_url>` - Add a paper from ArXiv URL
- **list papers** - Show all papers in the library
- **list papers** `<topic>` - Show papers for a specific topic

### Analysis & Insights
- **summarize topic** `<name>` - Generate summary of papers on a topic
- **insights topic** `<name>` - Get keyword insights and trends

### Other
- **help** - Show this help message
- **exit** / **quit** - Exit the application
    """
    console.print(Markdown(help_text))


def print_success(message: str):
    """Print a success message."""
    console.print(f"✅ {message}", style="bold green")


def print_error(message: str):
    """Print an error message."""
    console.print(f"❌ {message}", style="bold red")


def print_info(message: str):
    """Print an info message."""
    console.print(f"ℹ️  {message}", style="bold blue")


def display_user_input(text: str, title: str = "💬 User Input"):
    """Render user input inside a bordered panel for clarity."""
    panel = Panel(
        f"[bold white]{text}[/bold white]",
        title=title,
        border_style="magenta",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(panel)


def display_article_card(title: str, abstract: str, topics: list = None):
    """
    Display an article card with title, abstract, and detected topics.
    
    Args:
        title: Article title
        abstract: Article abstract (will be truncated if too long)
        topics: List of topic names (optional)
    """
    # Truncate abstract if too long
    max_abstract_length = 500
    if len(abstract) > max_abstract_length:
        abstract = abstract[:max_abstract_length] + "..."
    
    # Build the content
    content_parts = []
    
    # Title
    content_parts.append(f"[bold cyan]Title:[/bold cyan]\n{title}\n")
    
    # Abstract
    content_parts.append(f"[bold cyan]Abstract:[/bold cyan]\n{abstract}")
    
    # Topics (if any)
    if topics:
        topics_str = ", ".join([f"[green]{t}[/green]" for t in topics])
        content_parts.append(f"\n[bold cyan]Topics:[/bold cyan] {topics_str}")
    
    content = "\n".join(content_parts)
    
    # Display in a panel
    panel = Panel(
        content,
        title="📄 Paper Added",
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(panel)


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"⚠️  {message}", style="bold yellow")


def print_agent_action(action: str):
    """Print an agent action message."""
    console.print(f"🤖 {action}", style="bold magenta")


def print_fetching(message: str):
    """Print a fetching/loading message."""
    console.print(f"📡 {message}", style="bold cyan")


def print_analyzing(message: str):
    """Print an analyzing message."""
    console.print(f"🔍 {message}", style="bold cyan")


def print_summary(message: str):
    """Print a summary message."""
    console.print(f"🧠 {message}", style="bold green")


def get_user_input(prompt_text: str = "> ") -> str:
    """
    Get user input with a styled prompt.
    
    Args:
        prompt_text: The prompt text to display
    
    Returns:
        User input string
    """
    return Prompt.ask(f"[bold cyan]{prompt_text}[/bold cyan]")


def confirm_action(question: str) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        question: The question to ask
    
    Returns:
        True if confirmed, False otherwise
    """
    return Confirm.ask(f"[bold yellow]{question}[/bold yellow]")


def display_topics_table(topics: list):
    """
    Display topics in a formatted table.
    
    Args:
        topics: List of topic dictionaries
    """
    if not topics:
        print_info("No topics found.")
        return
    
    table = Table(
        title="Research Topics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    
    for topic in topics:
        description = topic.get('description', '') or ''
        if description and len(description) > 60:
            description = description[:60] + "..."

        table.add_row(
            str(topic.get('id', '')),
            topic.get('name', ''),
            description
        )
    
    console.print(table)


def display_papers_table(papers: list, topic_name: str = ""):
    """
    Display papers in a formatted table.
    
    Args:
        papers: List of paper dictionaries
        topic_name: Optional topic name for the title
    """
    if not papers:
        if topic_name:
            print_info(f"No papers found for topic '{topic_name}'.")
        else:
            print_info("No papers found.")
        return
    
    title = f"Papers - {topic_name}" if topic_name else "All Papers"
    
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("ID", style="cyan", justify="right", width=5)
    table.add_column("Title", style="green", width=50)
    table.add_column("ArXiv URL", style="blue", width=30)
    
    for paper in papers:
        title_text = paper.get('title', '')[:47] + "..." \
            if len(paper.get('title', '')) > 50 else paper.get('title', '')
        
        table.add_row(
            str(paper.get('id', '')),
            title_text,
            paper.get('arxiv_url', '')
        )
    
    console.print(table)


def display_markdown(text: str):
    """
    Display text as formatted markdown.
    
    Args:
        text: Markdown text to display
    """
    console.print(Markdown(text))


def display_panel(content: str, title: str = "", style: str = "cyan"):
    """
    Display content in a bordered panel.
    
    Args:
        content: Content to display
        title: Optional panel title
        style: Panel border style
    """
    panel = Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED
    )
    console.print(panel)


def clear_screen():
    """Clear the console screen."""
    console.clear()


def print_divider():
    """Print a visual divider."""
    console.print("─" * 70, style="dim")
