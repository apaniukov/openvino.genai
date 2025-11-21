"""
ArXiv Paper Fetching
Retrieves paper metadata and abstract from ArXiv API.
No external dependencies beyond requests and standard library.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional
import requests


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract ArXiv ID from URL."""
    # Match patterns like: https://arxiv.org/abs/2412.01234
    # or arxiv.org/abs/2412.01234v1
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If the input is just an ID
    if re.match(r'^\d+\.\d+$', url):
        return url

    return None


def fetch_arxiv_paper(arxiv_url: str) -> Dict:
    """
    Fetch paper metadata and abstract from ArXiv API.
    
    Args:
        arxiv_url: ArXiv URL or ID 
                   (e.g., https://arxiv.org/abs/2412.01234 or 2412.01234)
    
    Returns:
        Dictionary with paper information or error message
    """
    arxiv_id = extract_arxiv_id(arxiv_url)
    
    if not arxiv_id:
        return {
            "success": False,
            "message": f"Invalid ArXiv URL or ID: {arxiv_url}"
        }
    
    # ArXiv API endpoint
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Find the entry (paper)
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            return {
                "success": False,
                "message": f"Paper not found: {arxiv_id}"
            }
        
        # Extract metadata
        title = entry.find('atom:title', ns)
        abstract = entry.find('atom:summary', ns)
        published = entry.find('atom:published', ns)
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        
        # Clean up text (remove extra whitespace and newlines)
        if title is not None:
            title_text = ' '.join(title.text.split())
        else:
            title_text = "Unknown Title"
            
        if abstract is not None:
            abstract_text = ' '.join(abstract.text.split())
        else:
            abstract_text = ""
        
        # Construct the canonical ArXiv URL
        canonical_url = f"https://arxiv.org/abs/{arxiv_id}"
        
        return {
            "success": True,
            "arxiv_id": arxiv_id,
            "title": title_text,
            "abstract": abstract_text,
            "arxiv_url": canonical_url,
            "authors": authors,
            "published": published.text if published is not None else None
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "message": f"Error fetching paper from ArXiv: {str(e)}"
        }
    except ET.ParseError as e:
        return {
            "success": False,
            "message": f"Error parsing ArXiv response: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }
