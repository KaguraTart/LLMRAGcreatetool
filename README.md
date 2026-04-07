# LLM RAGtools

> 基于开源工具构建的 RAG 知识库处理工具链。覆盖 PDF 解析 → 智能分块 → 实体抽取 → 分类 → 向量索引全流程。

## 项目背景

本工具基于 [RAG 知识库知识处理工作流](https://github.com/KaguraTart/blog) 技术调研文章实现，重点探索：

1. **LLM 在知识处理各环节的适用性**：理解 vs 提取的边界
2. **Claude Code CLI / Gemini CLI 的能力边界**：作为 Agent 的使用方式
3. **MiniMax 多模态 API**：图像理解 + 文本生成的融合
4. **全开源工具栈**：最小化商业依赖

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                   输入文档层                       │
│  PDF · Word · Markdown · HTML · 图片/扫描件      │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  解析层（开源工具）                                │
│  PyMuPDF · pdfplumber · EasyOCR · MarkItDown    │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  处理层（LLM + 规则）                             │
│  智能分块 · 实体抽取 · 关系抽取 · 质量评分        │
│  MiniMax API · Sentence-Transformers             │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  索引层（开源数据库）                              │
│  Qdrant · Milvus · Neo4j (可选)                  │
└─────────────────────────────────────────────────┘
```

## 核心特性

- 🧩 **多格式支持**：PDF / Word / Markdown / HTML / 图片
- 📊 **表格双重处理**：规则提取 + LLM 校验
- 🖼️ **多模态理解**：MiniMax 图像描述生成
- 🧠 **智能分块**：按语义/章节/层级自适应分块
- 🔍 **混合检索**：向量 + BM25 + 知识图谱
- ⚡ **异步并发**：批量处理，吞吐量高
- 🛠️ **CLI Agent**：Claude Code CLI / Gemini CLI 集成（可选）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 可选：OCR 支持
pip install easyocr pytesseract
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# 可选：向量数据库
pip install qdrant-client

# 可选：知识图谱
pip install neo4j
```

### 基础使用

```python
from ragtools.pipeline import RAGPipeline
from ragtools.config import Config

# 加载配置
config = Config.from_yaml("config.yaml")

# 初始化流程
pipeline = RAGPipeline(config)

# 处理单个文档
chunks = await pipeline.process("document.pdf")

# 批量处理目录
all_chunks = await pipeline.process_corpus("./knowledge_base/")

# 检索
results = await pipeline.query("相关问题是什么？")
```

### CLI 使用

```bash
# 解析 PDF
python -m ragtools.cli extract document.pdf

# 完整工作流
python -m ragtools.cli process ./docs/ --output ./output/

# 交互式检索
python -m ragtools.cli query --question "你的问题"
```

## Claude Code CLI 集成说明

**关于 Claude Code CLI：**

Claude Code CLI（`claude-code` npm 包）是 Anthropic 官方发布的终端工具，**不是开源软件**。本项目通过 `subprocess` 调用它来执行复杂的多步推理任务：

- PDF 质量审查（跨段落逻辑一致性检查）
- 复杂表格语义理解
- 文档结构推理（当规则失效时）

```bash
# 安装 Claude Code CLI（需要 ANTHROPIC_API_KEY）
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY="your-key-here"
```

Claude Code CLI 作为 Agent 使用时，本质上是通过官方 API 调用 Claude 模型。它适合：
- ✅ 多步推理的复杂分析任务
- ✅ 需要浏览文件系统、搜索代码的分析
- ✅ 作为 fallback 处理规则无法处理的边缘情况

## 工具对比

| 环节 | 本项目方案 | 备选方案 |
|------|----------|---------|
| PDF 文字 | PyMuPDF | pdfminer, pdfplumber |
| OCR | EasyOCR + Tesseract | PaddleOCR, TrOCR |
| 表格 | pdfplumber + LLM 校验 | Camelot, Tabula |
| 图表理解 | MiniMax 多模态 API | GPT-4V, Claude Vision |
| 实体抽取 | MiniMax API (函数调用) | spaCy, transformers NER |
| 向量存储 | Qdrant | Milvus, Chroma, FAISS |
| 图数据库 | Neo4j (可选) | NebulaGraph |
| Embedding | HuggingFace sentence-transformers | OpenAI, MiniMax |

## 项目结构

```
LLM RAGtools/
├── config.yaml              # 全局配置
├── requirements.txt
├── README.md
├── src/
│   ├── extractors/         # 文档解析器
│   │   ├── pdf_extractor.py
│   │   ├── docx_extractor.py
│   │   ├── md_extractor.py
│   │   └── ocr_extractor.py
│   ├── processors/         # 知识处理器
│   │   ├── chunker.py      # 智能分块
│   │   ├── ner.py          # 实体抽取（MiniMax API）
│   │   ├── classifier.py   # 知识分类
│   │   └── quality.py      # 质量评分
│   ├── indexers/          # 索引存储
│   │   ├── vector_store.py # 向量索引
│   │   └── kg_indexer.py  # 知识图谱
│   ├── integrations/       # 外部工具集成
│   │   ├── minimax_api.py # MiniMax API 封装
│   │   ├── claude_cli.py  # Claude Code CLI
│   │   └── gemini_cli.py  # Gemini CLI
│   ├── pipeline.py         # 主工作流
│   └── config.py           # 配置管理
├── tests/
│   └── test_pipeline.py
└── examples/
    └── demo_pipeline.py
```

## License

MIT
