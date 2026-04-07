"""Demo: multi-provider runtime + query_and_answer."""

import asyncio

from src.config import Config
from src.pipeline import RAGPipeline


async def main():
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)

    registry = pipeline.provider_registry
    if registry:
        print("Providers:", registry.list())
        print("Active:", registry.active_name)

        results = registry.check_all()
        print("Health:", results)

        # Example: switch if openai is available and registered
        if "openai" in registry.list() and registry.active_name != "openai":
            try:
                registry.switch("openai")
                print("Switched active provider to:", registry.active_name)
            except Exception as e:
                print("Switch failed:", e)

    q = "What is RAG and when should hybrid retrieval be used?"
    result = await pipeline.query_and_answer(question=q, k=5, retrieval_mode="hybrid")

    print("\n=== Query & Answer ===")
    print("Provider:", result.get("provider"))
    print("Mode:", result.get("retrieval_mode"))
    print("Answer:\n", result.get("answer", ""))

    quality = result.get("quality")
    if quality:
        print("Quality score:", quality.get("score"), "passed=", quality.get("passed"))


if __name__ == "__main__":
    asyncio.run(main())
