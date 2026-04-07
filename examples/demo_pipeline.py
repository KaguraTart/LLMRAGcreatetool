"""
RAGtools Usage Examples

This file demonstrates various ways to use LLM RAGtools
"""

import asyncio
from pathlib import Path

from src.config import Config
from src.pipeline import RAGPipeline


async def demo_basic():
    """Basic usage: process a single PDF"""
    print("=" * 60)
    print("Demo 1: Basic Usage")
    print("=" * 60)
    
    # Load config
    config = Config.from_yaml("config.yaml")
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Process document
    pdf_path = "./examples/sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Sample file not found: {pdf_path}")
        print("Skipping basic demo")
        return
    
    chunks = await pipeline.process(pdf_path)
    
    print(f"\nResult: {len(chunks)} chunks")
    for chunk in chunks[:3]:
        print(f"  [{chunk.chunk_id}] {chunk.content[:100]}...")


async def demo_corpus():
    """Batch process directory"""
    print("\n" + "=" * 60)
    print("Demo 2: Batch Process Directory")
    print("=" * 60)
    
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)
    
    # Process entire directory
    chunks = await pipeline.process_corpus(
        "./knowledge_base/",
        max_workers=4,
    )
    
    stats = pipeline.get_stats()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demo_query():
    """Query/Retrieval"""
    print("\n" + "=" * 60)
    print("Demo 3: Query/Retrieval")
    print("=" * 60)
    
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)
    
    results = await pipeline.query(
        question="What is RAG?",
        k=5,
    )
    
    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r['score']:.3f}")
        print(r["content"][:200])


async def demo_custom_pipeline():
    """
    Demo 4: Custom Pipeline
    
    Shows how to use each module individually
    """
    print("\n" + "=" * 60)
    print("Demo 4: Custom Pipeline")
    print("=" * 60)
    
    from src.config import Config
    from src.extractors import PDFExtractor
    from src.processors.chunker import ChunkBuilder
    from src.processors.classifier import CascadeClassifier
    from src.integrations.embedding_model import EmbeddingModel
    from src.integrations.minimax_api import MiniMaxClient
    
    config = Config.from_yaml("config.yaml")
    
    # Use PDF extractor individually
    pdf_extractor = PDFExtractor()
    result = pdf_extractor.extract("examples/sample.pdf")
    print(f"PDF parsed: {result.total_pages} pages")
    
    # Use chunker individually
    chunker = ChunkBuilder(strategy="recursive", chunk_size=500)
    chunks = chunker.chunk_text(result.full_text)
    print(f"Chunked: {len(chunks)} chunks")
    
    # Use Embedding individually
    embed_model = EmbeddingModel(
        model_name=config.embedding.model_name,
        device="cpu"  # Not using GPU
    )
    if chunks:
        embeddings = embed_model.encode([chunks[0].content])
        print(f"Embedding dimension: {embeddings.shape}")


async def main():
    """Run all demos"""
    try:
        await demo_basic()
        # await demo_corpus()
        # await demo_query()
        # await demo_custom_pipeline()
        
        print("\n" + "=" * 60)
        print("All demos complete!")
        print("=" * 60)
    
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Please make sure config.yaml exists")


if __name__ == "__main__":
    asyncio.run(main())
