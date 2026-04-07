# LLMRAGcreatetool

> For Chinese documentation, see [README_zh.md](./README_zh.md)

A production-ready RAG knowledge base processing toolkit. Covers the full pipeline from **PDF parsing → smart chunking → entity extraction → classification → vector indexing**.

## Features

- 🧩 **Multi-format support**: PDF / Word / Markdown / HTML / scanned images
- 📊 **Dual table processing**: Rule extraction + LLM validation
- 🖼️ **Multimodal understanding**: MiniMax API for chart/image description generation
- 🧠 **Smart chunking**: Fixed / recursive / semantic / heading-aware strategies
- 🔍 **Hybrid retrieval**: Vector + BM25 + Knowledge Graph
- ⚡ **Async concurrency**: Batch processing for high throughput


## Architecture

```
Input Layer
  PDF · Word · Markdown · HTML · scanned images
       ↓
Extraction Layer (open-source tools)
  PyMuPDF · pdfplumber · EasyOCR · MarkItDown
       ↓
Processing Layer (LLM + rules)
  Smart chunking · entity extraction · quality scoring
  MiniMax API · sentence-transformers
       ↓
Indexing Layer (open-source databases)
  Qdrant · Milvus · Neo4j (optional)
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt

# Optional: OCR support
pip install easyocr pytesseract
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# Optional: vector database
pip install qdrant-client
```

### Basic Usage

```python
from ragtools.pipeline import RAGPipeline
from ragtools.config import Config

# Load config
config = Config.from_yaml("config.yaml")

# Initialize pipeline
pipeline = RAGPipeline(config)

# Process single document
chunks = await pipeline.process("document.pdf")

# Batch process directory
all_chunks = await pipeline.process_corpus("./knowledge_base/")

# Query
results = await pipeline.query("Your question here?")
```

### CLI Usage

```bash
# Extract document
python -m src extract document.pdf

# Full pipeline
python -m src process ./docs/ --output ./output/

# Query
python -m src query --question "Your question"
```
## Tool Comparison

| Task | Our Approach | Alternative |
|------|------------|-------------|
| PDF text | PyMuPDF | pdfminer, pdfplumber |
| OCR | EasyOCR + Tesseract | PaddleOCR, TrOCR |
| Tables | pdfplumber + LLM validation | Camelot, Tabula |
| Charts | MiniMax multimodal API | GPT-4V |
| Entity extraction | MiniMax API (function calling) | spaCy, transformers NER |
| Vector store | Qdrant | Milvus, Chroma, FAISS |
| Graph DB | Neo4j (optional) | NebulaGraph |
| Embedding | HuggingFace sentence-transformers | OpenAI, MiniMax |

## Project Structure

```
LLMRAGcreatetool/
├── config.yaml              # Global configuration
├── requirements.txt
├── README.md
├── README_zh.md            # Chinese documentation
├── src/
│   ├── extractors/         # Document parsers
│   │   ├── pdf_extractor.py   # PDF multi-layer parsing
│   │   ├── docx_extractor.py # Word document parsing
│   │   ├── md_extractor.py   # Markdown semantic parsing
│   │   └── ocr_extractor.py   # EasyOCR + Tesseract
│   ├── processors/         # Knowledge processors
│   │   ├── chunker.py        # 4 chunking strategies
│   │   ├── classifier.py     # Cascade classifier
│   │   └── quality.py        # LLM self-evaluation
│   ├── integrations/       # External tool integration
│   │   ├── minimax_api.py   # MiniMax API wrapper
│   │   └── embedding_model.py # HF + MiniMax dual backend
│   ├── indexers/           # Index storage
│   │   └── vector_store.py   # Qdrant + Chroma
│   ├── pipeline.py          # Main pipeline
│   └── config.py            # Configuration
├── examples/
│   └── demo_pipeline.py
└── tests/
```

## Tech Stack

| Layer | Technologies |
|-------|------------|
| PDF parsing | PyMuPDF, pdfplumber, Nougat |
| OCR | EasyOCR, Tesseract |
| Embedding | sentence-transformers (BAAI/bge-large-zh-v1.5) |
| Vector DB | Qdrant, Milvus, Chroma |
| LLM | MiniMax API (multimodal + function calling) |
| Config | Pydantic, YAML |

## License

MIT
