# Research Organizer Agent 📚

A console-based, chat-driven AI research assistant that helps you manage and explore research papers by topic. Built with Python's `rich` library for a beautiful terminal interface and Model Context Protocol (MCP) for modular tool architecture.

## Features

✨ **Conversational Console Interface**
- Styled terminal chat using `rich` library
- Colorized messages for different agent actions
- Interactive prompts and confirmations

🏷️ **Topic Management**
- Add, list, and remove research topics
- Store topics locally in SQLite database

📄 **Paper Ingestion (ArXiv)**
- Add papers via ArXiv URL (e.g., `https://arxiv.org/abs/2412.01234`)
- Automatically fetch title and abstract
- No PDF download - metadata only

🤖 **Automatic Topic Tagging**
- Papers are automatically classified to relevant topics
- Uses keyword similarity matching
- Based on abstract content analysis

📊 **Summarization & Insights**
- Summarize all papers on a specific topic
- Generate keyword insights and trends
- Extractive summarization from abstracts

🔧 **MCP-Based Architecture**
- Each functional module is an MCP tool
- Modular and extensible design
- Database, ArXiv, classification, and summarization tools

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Navigate to the project directory:
```bash
cd samples/python/agentic_ai/research_organizer
```

2. Install dependencies:
```bash
pip install rich requests fastmcp
```

3. Run the application:
```bash
python agent.py
```

## Usage

### Starting the Application

```bash
python agent.py
```

You'll be greeted with a welcome screen showing available commands.

### Available Commands

#### Topic Management

**Add a topic:**
```
> add topic scalable alignment
```

**Add a topic with description:**
```
> add topic machine learning An area focusing on algorithms that learn from data
```

**List all topics:**
```
> list topics
```

**Remove a topic:**
```
> remove topic scalable alignment
```

#### Paper Management

**Add a paper from ArXiv:**
```
> add paper https://arxiv.org/abs/2412.01234
```

The agent will:
1. Fetch the paper metadata from ArXiv
2. Store title and abstract in the database
3. Automatically classify it to relevant topics

**List all papers:**
```
> list papers
```

**List papers for a specific topic:**
```
> list papers scalable alignment
```

#### Analysis & Insights

**Summarize papers on a topic:**
```
> summarize topic scalable alignment
```

**Get keyword insights:**
```
> insights topic machine learning
```

#### Other Commands

**Show help:**
```
> help
```

**Exit the application:**
```
> exit
```

## Example Session

```
> add topic scalable alignment
✅ Added topic 'scalable alignment'.

> add topic ai governance
✅ Added topic 'ai governance'.

> add paper https://arxiv.org/abs/2412.01234
📡 Fetching paper from ArXiv...
✅ Retrieved: Scalable Oversight in Large Language Models
🤖 Adding paper to database...
🔍 Analyzing paper for topic classification...
✅ Tagged under topics: scalable alignment, ai governance
✅ Paper added successfully!

> summarize topic scalable alignment
🤖 Generating summary for topic 'scalable alignment'...
──────────────────────────────────────────────────────────────────────

Summary of 1 paper(s) on 'scalable alignment':

1. **Scalable Oversight in Large Language Models**
   This paper presents novel techniques for efficiently monitoring
   and aligning large language models...

──────────────────────────────────────────────────────────────────────

> insights topic scalable alignment
🤖 Generating insights for topic 'scalable alignment'...
──────────────────────────────────────────────────────────────────────
╭─ 📊 Insights: scalable alignment ─────────────────────────────────╮
│ Topic: scalable alignment                                         │
│ Total papers: 1                                                   │
│                                                                   │
│ Most frequent terms:                                              │
│   - alignment: 12 occurrences                                     │
│   - models: 8 occurrences                                         │
│   - scalable: 6 occurrences                                       │
╰───────────────────────────────────────────────────────────────────╯
──────────────────────────────────────────────────────────────────────
```

## Architecture

### Project Structure

```
research_organizer/
│
├── agent.py              # Main orchestration + chat loop
├── ui.py                 # Rich-based UI helpers
├── mcp_db.py             # MCP tool for database operations
├── mcp_arxiv.py          # MCP tool for fetching paper abstracts
├── mcp_topics.py         # MCP tool for topic classification
├── mcp_summary.py        # MCP tool for summarization
├── data/                 # Contains local SQLite database
│   └── research.db       # Auto-generated on first run
└── README.md
```

### Database Schema

**Topics Table:**
```sql
CREATE TABLE topics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Papers Table:**
```sql
CREATE TABLE papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  arxiv_url TEXT UNIQUE NOT NULL,
  abstract TEXT,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Paper-Topics Junction Table:**
```sql
CREATE TABLE paper_topics (
  paper_id INTEGER NOT NULL,
  topic_id INTEGER NOT NULL,
  PRIMARY KEY (paper_id, topic_id),
  FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
  FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);
```

### MCP Tools

Each functional module is implemented as an MCP (Model Context Protocol) tool:

1. **mcp_db.py** - Database CRUD operations
   - Add/list/remove topics
   - Add papers
   - Link papers to topics
   - Query papers by topic

2. **mcp_arxiv.py** - ArXiv integration
   - Fetch paper metadata from ArXiv API
   - Parse XML responses
   - Extract title, abstract, authors

3. **mcp_topics.py** - Topic classification
   - Keyword-based similarity matching
   - Automatic topic tagging
   - Configurable similarity threshold

4. **mcp_summary.py** - Summarization
   - Generate paper summaries by topic
   - Extract key insights
   - Keyword frequency analysis

## Tech Stack

- **Python 3.8+** - Core language
- **rich** - Terminal UI and formatting
- **sqlite3** - Local database (built-in)
- **requests** - HTTP client for ArXiv API
- **fastmcp** - Model Context Protocol framework
- **xml.etree.ElementTree** - XML parsing for ArXiv responses

## Future Enhancements

Potential improvements:

- 🔗 Integration with LLM APIs for better summarization
- 🧠 Embedding-based topic classification (using sentence-transformers)
- 📈 Visualization of research trends over time
- 🔍 Full-text search across abstracts
- 📦 Export capabilities (PDF reports, CSV exports)
- 🌐 Support for additional paper sources (PubMed, Semantic Scholar)
- 💾 Cloud database support for multi-device sync

## License

This project is part of the OpenVINO GenAI samples.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For questions or issues, please open an issue in the OpenVINO GenAI repository.

---

**Happy Researching! 🚀📚**
