"""
RAGtools 命令行接口
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
    """提取文档内容"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    chunks = await pipeline.process(args.file)
    
    print(f"\n提取完成: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
    
    if len(chunks) > 5:
        print(f"\n... 还有 {len(chunks) - 5} 个 chunks")


async def cmd_process(args):
    """处理目录"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    chunks = await pipeline.process_corpus(
        args.directory,
        max_workers=args.workers,
    )
    
    stats = pipeline.get_stats()
    
    print(f"\n处理完成:")
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  总 chunks: {stats['total_chunks']}")
    print(f"  总实体数: {stats['total_entities']}")
    print(f"  失败文档: {stats['failed_documents']}")


async def cmd_query(args):
    """检索"""
    setup_logging(args.log)
    
    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    
    results = await pipeline.query(
        question=args.question,
        k=args.k,
        category_filter=args.category,
    )
    
    print(f"\n找到 {len(results)} 条相关结果:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] (score={r['score']:.3f})")
        print(r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM RAGtools CLI")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    sub = parser.add_subparsers(dest="command", required=True)
    
    # extract 命令
    extract = sub.add_parser("extract", help="提取单个文档")
    extract.add_argument("file", help="文件路径")
    
    # process 命令
    process = sub.add_parser("process", help="处理整个目录")
    process.add_argument("directory", help="目录路径")
    process.add_argument("-w", "--workers", type=int, default=4, help="并发数")
    
    # query 命令
    query = sub.add_parser("query", help="检索")
    query.add_argument("-q", "--question", required=True, help="查询问题")
    query.add_argument("-k", type=int, default=5, help="返回数量")
    query.add_argument("--category", help="按分类过滤")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        asyncio.run(cmd_extract(args))
    elif args.command == "process":
        asyncio.run(cmd_process(args))
    elif args.command == "query":
        asyncio.run(cmd_query(args))


if __name__ == "__main__":
    main()
