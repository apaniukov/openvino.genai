"""
Database Operations for Research Organizer
Pure SQLite CRUD operations for topics, papers, and paper_topics.
No external dependencies - can be used standalone.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "research.db"


def init_database(init_new_db: bool = False):
    """Initialize the SQLite database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if init_new_db and DB_PATH.exists():
        DB_PATH.unlink()  # Remove existing database for fresh start
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create topics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create papers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            arxiv_url TEXT UNIQUE NOT NULL,
            abstract TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create paper_topics junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_topics (
            paper_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            PRIMARY KEY (paper_id, topic_id),
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()


def add_topic(name: str, description: str = "") -> Dict:
    """Add a new topic to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO topics (name, description) VALUES (?, ?)",
            (name, description)
        )
        topic_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "topic_id": topic_id,
            "message": f"Added topic '{name}'."
        }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": f"Topic '{name}' already exists."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error adding topic: {str(e)}"
        }


def list_topics() -> List[Dict]:
    """List all topics in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, name, description, created_at FROM topics ORDER BY id"
    )
    topics = []
    
    for row in cursor.fetchall():
        topics.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3]
        })
    
    conn.close()
    return topics


def remove_topic(name: str) -> Dict:
    """Remove a topic from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM topics WHERE name = ?", (name,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return {
                "success": True,
                "message": f"Removed topic '{name}'."
            }
        else:
            return {
                "success": False,
                "message": f"Topic '{name}' not found."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error removing topic: {str(e)}"
        }


def add_paper(title: str, arxiv_url: str, abstract: str) -> Dict:
    """Add a new paper to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO papers (title, arxiv_url, abstract) VALUES (?, ?, ?)",
            (title, arxiv_url, abstract)
        )
        paper_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "paper_id": paper_id,
            "message": f"Added paper '{title}'."
        }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": f"Paper with URL '{arxiv_url}' already exists."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error adding paper: {str(e)}"
        }


def link_paper_to_topics(paper_id: int, topic_ids: List[int]) -> Dict:
    """Associate a paper with multiple topics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for topic_id in topic_ids:
            cursor.execute(
                """INSERT OR IGNORE INTO paper_topics 
                   (paper_id, topic_id) VALUES (?, ?)""",
                (paper_id, topic_id)
            )
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Linked paper to {len(topic_ids)} topic(s)."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error linking paper to topics: {str(e)}"
        }


def get_papers_by_topic(topic_name: str) -> List[Dict]:
    """Get all papers associated with a specific topic."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.id, p.title, p.arxiv_url, p.abstract, p.added_at
        FROM papers p
        JOIN paper_topics pt ON p.id = pt.paper_id
        JOIN topics t ON pt.topic_id = t.id
        WHERE t.name = ?
        ORDER BY p.added_at DESC
    """, (topic_name,))
    
    papers = []
    for row in cursor.fetchall():
        papers.append({
            "id": row[0],
            "title": row[1],
            "arxiv_url": row[2],
            "abstract": row[3],
            "added_at": row[4]
        })
    
    conn.close()
    return papers


def get_all_papers() -> List[Dict]:
    """Get all papers in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT id, title, arxiv_url, abstract, added_at 
           FROM papers ORDER BY added_at DESC"""
    )
    
    papers = []
    for row in cursor.fetchall():
        papers.append({
            "id": row[0],
            "title": row[1],
            "arxiv_url": row[2],
            "abstract": row[3],
            "added_at": row[4]
        })
    
    conn.close()
    return papers


def get_topic_by_name(name: str) -> Optional[Dict]:
    """Get a topic by name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, name, description, created_at FROM topics WHERE name = ?",
        (name,)
    )
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3]
        }
    return None
