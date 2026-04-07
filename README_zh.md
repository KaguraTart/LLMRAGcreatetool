# LLMRAGcreatetool

> 英文文档请见：[README.md](./README.md)

生产级 RAG 知识库处理工具链，覆盖从 **PDF 解析 → 智能分块 → 实体抽取 → 分类 → 向量索引** 的完整流程。

本文档由技术调研文章 [《RAG 知识库知识处理工作流》](https://github.com/KaguraTart/blog) 驱动实现，重点探索：

1. **LLM 在知识处理各环节的适用性**：理解 vs 提取的边界
2. **MiniMax 多模态 API**：图像理解 + 文本生成的融合
3. **全开源工具栈**：最小化商业依赖

## 核心特性

- 🧩 **多格式支持**：PDF / Word / Markdown / HTML / 图片/扫描件
- 📊 **表格双重处理**：规则提取 + LLM 校验
- 🖼️ **多模态理解**：MiniMax 图像描述生成
- 🧠 **智能分块**：按语义/章节/层级自适应分块
- 🔍 **混合检索**：向量 + BM25 + 知识图谱
- ⚡ **异步并发**：批量处理，吞吐量高

## 架构概览

```
输入文档层
  PDF · Word · Markdown · HTML · 图片/扫描件
          ↓
解析层（开源工具）
  PyMuPDF · pdfplumber · EasyOCR · MarkItDown
          ↓
处理层（LLM + 规则）
  智能分块 · 实体抽取 · 关系抽取 · 质量评分
  MiniMax API · Sentence-Transformers
          ↓
索引层（开源数据库）
  Qdrant · Milvus · Neo4j (可选)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 可选：OCR 支持
pip install easyocr pytesseract
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# 可选：向量数据库
pip install qdrant-client
```

### 基础使用

```python
from ragtools.pipeline import RAGPipeline
from ragtools.config import Config

# 加载配置
config = Config.from_yaml("config.yaml")

# 初始化流程
pipeline = RAGPipeline(config)

# 处理文档
chunks = await pipeline.process("document.pdf")

# 批量处理目录
all_chunks = await pipeline.process_corpus("./knowledge_base/")

# 检索
results = await pipeline.query("相关问题是什么？")
```

### CLI 使用

```bash
# 解析 PDF
python -m src extract document.pdf

# 完整工作流
python -m src process ./docs/ --output ./output/

# 交互式检索
python -m src query --question "你的问题"
```

## 工具对比

| 环节 | 本项目方案 | 备选方案 |
|------|----------|---------|
| PDF 文字 | PyMuPDF | pdfminer, pdfplumber |
| OCR | EasyOCR + Tesseract | PaddleOCR, TrOCR |
| 表格 | pdfplumber + LLM 校验 | Camelot, Tabula |
| 图表理解 | MiniMax 多模态 API | GPT-4V |
| 实体抽取 | MiniMax API (函数调用) | spaCy, transformers NER |
| 向量存储 | Qdrant | Milvus, Chroma, FAISS |
| 图数据库 | Neo4j (可选) | NebulaGraph |
| Embedding | HuggingFace sentence-transformers | OpenAI, MiniMax |

## 项目结构

```
LLMRAGcreatetool/
├── config.yaml              # 全局配置
├── requirements.txt          # 依赖清单
├── README.md                 # 英文文档
├── README_zh.md             # 中文文档
│
├── src/
│   ├── extractors/         # 文档解析层
│   │   ├── pdf_extractor.py   # PDF多层解析(PyMuPDF/pdfplumber/OCR)
│   │   ├── docx_extractor.py # Word文档解析
│   │   ├── md_extractor.py   # Markdown语义解析
│   │   └── ocr_extractor.py  # OCR双引擎(EasyOCR+Tesseract)
│   │
│   ├── processors/         # 知识处理层
│   │   ├── chunker.py        # 四种分块策略(固定/递归/语义/层级)
│   │   ├── classifier.py     # 级联分类器(规则→Embedding→LLM)
│   │   └── quality.py        # LLM自评质量+规则降级
│   │
│   ├── integrations/       # 外部工具集成
│   │   ├── minimax_api.py   # MiniMax API封装(文本/多模态/函数调用)
│   │   └── embedding_model.py # HuggingFace Embedding+MiniMax双支持
│   │
│   ├── indexers/           # 索引存储层
│   │   └── vector_store.py   # Qdrant+Chroma双支持
│   │
│   ├── pipeline.py        # 主工作流(完整串联)
│   └── config.py           # 配置管理
│
├── examples/
│   └── demo_pipeline.py
│
└── tests/
```

## License

MIT
