"""RAGtools Command Line Interface."""

import asyncio
import logging


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_extract(args):
    """Extract command; imports are local to avoid package import cycles."""
    setup_logging(args.log)
    from ..config import Config
    from ..pipeline import RAGPipeline

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
    """Process command; imports are local to avoid package import cycles."""
    setup_logging(args.log)
    from ..config import Config
    from ..pipeline import RAGPipeline

    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)

    await pipeline.process_corpus(args.directory, max_workers=args.workers)
    stats = pipeline.get_stats()

    print("\nProcessing complete:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Failed documents: {stats['failed_documents']}")


async def cmd_query(args):
    """Query command; imports are local to avoid package import cycles."""
    setup_logging(args.log)
    from ..config import Config
    from ..pipeline import RAGPipeline

    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)

    results = await pipeline.query(question=args.question, k=args.k, category_filter=args.category)
    print(f"\nFound {len(results)} relevant results:")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] (score={r['score']:.3f})")
        print(r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"])


async def cmd_provider(args):
    setup_logging(args.log)
    from ..config import Config
    from ..pipeline import RAGPipeline

    config = Config.from_yaml(args.config)
    pipeline = RAGPipeline(config)
    registry = pipeline.provider_registry

    if registry is None:
        print("Provider registry unavailable")
        return

    if args.provider_cmd == "list":
        active = registry.active_name
        for name in registry.list():
            marker = "*" if name == active else " "
            print(f"{marker} {name}")
    elif args.provider_cmd == "active":
        print(registry.active_name or "")
    elif args.provider_cmd == "switch":
        registry.switch(args.name)
        print(f"Active provider switched to: {registry.active_name}")
    elif args.provider_cmd == "check":
        ok = registry.check(args.name)
        print(f"{args.name}: {'healthy' if ok else 'unavailable'}")
    elif args.provider_cmd == "check-all":
        results = registry.check_all()
        for name, ok in results.items():
            print(f"{name}: {'healthy' if ok else 'unavailable'}")


async def cmd_qa_ask(args):
    setup_logging(args.log)
    from ..config import Config
    from ..pipeline import RAGPipeline

    config = Config.from_yaml(args.config)
    if args.mode:
        config.qa.retrieval_mode = args.mode
    if args.rerank is not None:
        config.qa.rerank_enabled = args.rerank

    pipeline = RAGPipeline(config)
    result = await pipeline.query_and_answer(
        question=args.question,
        k=args.k,
        category_filter=args.category,
        retrieval_mode=args.mode,
        rerank=args.rerank,
    )

    print("\n=== QA Result ===")
    print(f"Provider      : {result['provider']}")
    print(f"Intent        : {result['intent']}")
    print(f"Retrieval Mode: {result['retrieval_mode']}")
    print(f"Results       : {result['retrieval']['results_count']}")
    if result.get("quality"):
        q = result["quality"]
        print(f"Quality       : {q['score']:.3f} (passed={q['passed']})")

    print("\nAnswer:")
    print(result.get("answer", ""))

    retrieval_results = result.get("retrieval", {}).get("results", [])
    if retrieval_results:
        print("\nSources:")
        for i, item in enumerate(retrieval_results[: args.k], 1):
            print(f"[{i}] score={float(item.get('score', 0.0)):.3f} id={item.get('id', '')}")

    if result.get("warnings"):
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"- {w}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM RAGtools CLI")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract single document")
    extract.add_argument("file", help="File path")

    process = sub.add_parser("process", help="Process entire directory")
    process.add_argument("directory", help="Directory path")
    process.add_argument("-w", "--workers", type=int, default=4, help="Concurrency count")

    query = sub.add_parser("query", help="Search/Query")
    query.add_argument("-q", "--question", required=True, help="Query question")
    query.add_argument("-k", type=int, default=5, help="Number of results to return")
    query.add_argument("--category", help="Filter by category")

    provider = sub.add_parser("provider", help="Provider operations")
    provider_sub = provider.add_subparsers(dest="provider_cmd", required=True)
    provider_sub.add_parser("list", help="List providers")
    provider_sub.add_parser("active", help="Show active provider")
    provider_switch = provider_sub.add_parser("switch", help="Switch active provider")
    provider_switch.add_argument("name", help="Provider name")
    provider_check = provider_sub.add_parser("check", help="Check one provider")
    provider_check.add_argument("name", help="Provider name")
    provider_sub.add_parser("check-all", help="Check all providers")

    qa = sub.add_parser("qa", help="Question answering")
    qa_sub = qa.add_subparsers(dest="qa_cmd", required=True)
    qa_ask = qa_sub.add_parser("ask", help="Ask with retrieval + answer generation")
    qa_ask.add_argument("-q", "--question", required=True, help="Query question")
    qa_ask.add_argument("-k", type=int, default=5, help="Number of context results")
    qa_ask.add_argument("--category", help="Filter by category")
    qa_ask.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="Retrieval mode override")
    qa_ask.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable rerank override",
    )

    args = parser.parse_args()

    if args.command == "extract":
        asyncio.run(cmd_extract(args))
    elif args.command == "process":
        asyncio.run(cmd_process(args))
    elif args.command == "query":
        asyncio.run(cmd_query(args))
    elif args.command == "provider":
        asyncio.run(cmd_provider(args))
    elif args.command == "qa" and args.qa_cmd == "ask":
        asyncio.run(cmd_qa_ask(args))


if __name__ == "__main__":
    main()
