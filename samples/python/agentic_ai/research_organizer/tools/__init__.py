"""
Tools Package - Pure Implementations
Contains standalone tool implementations without MCP dependencies.
"""

from .db import (
    init_database,
    add_topic,
    list_topics,
    remove_topic,
    add_paper,
    link_paper_to_topics,
    get_papers_by_topic,
    get_all_papers,
    get_topic_by_name
)
from .arxiv import fetch_arxiv_paper, extract_arxiv_id

__all__ = [
    # Database operations
    'init_database',
    'add_topic',
    'list_topics',
    'remove_topic',
    'add_paper',
    'link_paper_to_topics',
    'get_papers_by_topic',
    'get_all_papers',
    'get_topic_by_name',
    # ArXiv operations
    'fetch_arxiv_paper',
    'extract_arxiv_id',
]
