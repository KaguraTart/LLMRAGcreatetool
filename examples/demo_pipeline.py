"""
RAGtools 使用示例

本文件展示 LLM RAGtools 的各种使用方式
"""

import asyncio
from pathlib import Path

from src.config import Config
from src.pipeline import RAGPipeline


async def demo_basic():
    """基础使用：处理单个 PDF"""
    print("=" * 60)
    print("Demo 1: 基础使用")
    print("=" * 60)
    
    # 加载配置
    config = Config.from_yaml("config.yaml")
    
    # 初始化流程
    pipeline = RAGPipeline(config)
    
    # 处理文档
    pdf_path = "./examples/sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"示例文件不存在: {pdf_path}")
        print("跳过基础演示")
        return
    
    chunks = await pipeline.process(pdf_path)
    
    print(f"\n结果: {len(chunks)} chunks")
    for chunk in chunks[:3]:
        print(f"  [{chunk.chunk_id}] {chunk.content[:100]}...")


async def demo_corpus():
    """批量处理目录"""
    print("\n" + "=" * 60)
    print("Demo 2: 批量处理目录")
    print("=" * 60)
    
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)
    
    # 处理整个目录
    chunks = await pipeline.process_corpus(
        "./knowledge_base/",
        max_workers=4,
    )
    
    stats = pipeline.get_stats()
    print(f"\n统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demo_query():
    """检索"""
    print("\n" + "=" * 60)
    print("Demo 3: 检索")
    print("=" * 60)
    
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)
    
    results = await pipeline.query(
        question="什么是 RAG？",
        k=5,
    )
    
    print(f"\n找到 {len(results)} 条结果:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']:.3f}")
        print(r["content"][:200])


async def demo_custom_pipeline():
    """
    Demo 4: 自定义 Pipeline
    
    展示如何单独使用各模块
    """
    print("\n" + "=" * 60)
    print("Demo 4: 自定义 Pipeline")
    print("=" * 60)
    
    from src.config import Config
    from src.extractors import PDFExtractor
    from src.processors.chunker import ChunkBuilder
    from src.processors.classifier import CascadeClassifier
    from src.integrations.embedding_model import EmbeddingModel
    from src.integrations.minimax_api import MiniMaxClient
    
    config = Config.from_yaml("config.yaml")
    
    # 单独使用 PDF 解析器
    pdf_extractor = PDFExtractor()
    result = pdf_extractor.extract("examples/sample.pdf")
    print(f"PDF 解析: {result.total_pages} 页")
    
    # 单独使用分块器
    chunker = ChunkBuilder(strategy="recursive", chunk_size=500)
    chunks = chunker.chunk_text(result.full_text)
    print(f"分块: {len(chunks)} chunks")
    
    # 单独使用 Embedding
    embed_model = EmbeddingModel(
        model_name=config.embedding.model_name,
        device="cpu"  # 没用 GPU
    )
    if chunks:
        embeddings = embed_model.encode([chunks[0].content])
        print(f"Embedding 维度: {embeddings.shape}")


async def main():
    """运行所有演示"""
    try:
        await demo_basic()
        # await demo_corpus()
        # await demo_query()
        # await demo_custom_pipeline()
        
        print("\n" + "=" * 60)
        print("所有演示完成!")
        print("=" * 60)
    
    except FileNotFoundError as e:
        print(f"配置文件未找到: {e}")
        print("请确保 config.yaml 存在")


if __name__ == "__main__":
    asyncio.run(main())
