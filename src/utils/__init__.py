"""
RAGtools Command Line Interface
"""

import asyncio
import sys
import logging
from pathlib import Path

from .config import Config
from .pipeline import RAGPipeline


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_extract(args):
    """Extract document content"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    chunks = await pipeline.process(args.file)
    
    print(f"\nExtraction complete: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")


async def cmd_process(args):
    """Process directory"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    chunks = await pipeline.process_corpus(
        args.directory,
        max_workers=args.workers,
    )
    
    stats = pipeline.get_stats()
    
    print(f"\nProcessing complete:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Failed documents: {stats['failed_documents']}")


async def cmd_query(args):
    """Search/Query"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    results = await pipeline.query(
        question=args.question,
        k=args.k,
        category_filter=args.category,
    )
    
    print(f"\nFound {len(results)} relevant results:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] (score={r['score']:.3f})")
        print(r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM RAGtools CLI")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    sub = parser.add_subparsers(dest="command", required=True)
    
    # extract command
    extract = sub.add_parser("extract", help="Extract single document")
    extract.add_argument("file", help="File path")
    
    # process command
    process = sub.add_parser("process", help="Process entire directory")
    process.add_argument("directory", help="Directory path")
    process.add_argument("-w", "--workers", type=int, default=4, help="Concurrency count")
    
    # query command
    query = sub.add_parser("query", help="Search/Query")
    query.add_argument("-q", "--question", required=True, help="Query question")
    query.add_argument("-k", type=int, default=5, help="Number of results to return")
    query.add_argument("--category", help="Filter by category")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        asyncio.run(cmd_extract(args))
    elif args.command == "process":
        asyncio.run(cmd_process(args))
    elif args.command == "query":
        asyncio.run(cmd_query(args))


if __name__ == "__main__":
    main()
