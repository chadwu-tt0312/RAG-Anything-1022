"""Direct Azure OpenAI connection test.

Purpose:
- Verify Azure OpenAI service connectivity
- Validate environment variable configuration
- Test LLM (chat completion) functionality
- Test Embedding functionality
- Confirm API version and deployment names are correct

Usage:
    python eval/integration/azure_openai_direct_test.py

Required Environment Variables:
    LLM:
    - LLM_BINDING_HOST: Azure OpenAI endpoint (e.g., https://{resource}.openai.azure.com/)
    - LLM_BINDING_API_KEY: Azure OpenAI API key
    - AZURE_OPENAI_API_VERSION: API version (e.g., 2024-12-01-preview)
    - AZURE_OPENAI_DEPLOYMENT: Chat model deployment name (e.g., gpt-4o-mini)

    Embedding:
    - AZURE_EMBEDDING_BINDING_HOST: Azure OpenAI endpoint for embedding (can reuse LLM host)
    - AZURE_EMBEDDING_BINDING_API_KEY: Azure OpenAI API key for embedding
    - AZURE_EMBEDDING_API_VERSION: API version for embedding
    - AZURE_EMBEDDING_DEPLOYMENT: Embedding model deployment name
    - AZURE_EMBEDDING_DIM: Embedding dimension (1536 for text-embedding-3-small)
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure the project root is in path for dotenv loading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_env() -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=".env", override=False)
        print("[INFO] Loaded .env file")
    except ImportError:
        print("[WARN] python-dotenv not installed, relying on OS environment")


def get_env(name: str, fallback: str | None = None) -> str:
    """Get environment variable with optional fallback."""
    value = os.getenv(name)
    if value:
        return value
    if fallback:
        fallback_value = os.getenv(fallback)
        if fallback_value:
            print(f"[INFO] Using {fallback} as fallback for {name}")
            return fallback_value
    raise RuntimeError(f"Missing required environment variable: {name}")


def print_config_summary() -> None:
    """Print configuration summary (without sensitive data)."""
    print("\n" + "=" * 60)
    print("Azure OpenAI Configuration Summary")
    print("=" * 60)

    # LLM Config
    llm_host = os.getenv("LLM_BINDING_HOST", "(not set)")
    llm_api_key = os.getenv("LLM_BINDING_API_KEY", "")
    llm_key_preview = (
        f"{llm_api_key[:8]}...{llm_api_key[-4:]}"
        if len(llm_api_key) > 12
        else "(not set or too short)"
    )
    llm_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "(not set)")
    llm_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "(not set)")

    print(f"\nLLM Configuration:")
    print(f"  Host:       {llm_host}")
    print(f"  API Key:    {llm_key_preview}")
    print(f"  Deployment: {llm_deployment}")
    print(f"  API Version: {llm_api_version}")

    # Embedding Config
    emb_host = os.getenv("AZURE_EMBEDDING_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "(not set)"))
    emb_api_key = os.getenv("AZURE_EMBEDDING_BINDING_API_KEY", os.getenv("LLM_BINDING_API_KEY", ""))
    emb_key_preview = (
        f"{emb_api_key[:8]}...{emb_api_key[-4:]}"
        if len(emb_api_key) > 12
        else "(not set or too short)"
    )
    emb_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "(not set)")
    emb_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "(not set)")
    emb_dim = os.getenv("AZURE_EMBEDDING_DIM", "(not set)")

    print(f"\nEmbedding Configuration:")
    print(f"  Host:       {emb_host}")
    print(f"  API Key:    {emb_key_preview}")
    print(f"  Deployment: {emb_deployment}")
    print(f"  API Version: {emb_api_version}")
    print(f"  Dimension:  {emb_dim}")
    print("=" * 60 + "\n")


async def test_llm() -> bool:
    """Test Azure OpenAI LLM (chat completion) connectivity."""
    print("[TEST] LLM Chat Completion...")

    try:
        from openai import AsyncAzureOpenAI

        endpoint = get_env("LLM_BINDING_HOST")
        api_key = get_env("LLM_BINDING_API_KEY")
        deployment = get_env("AZURE_OPENAI_DEPLOYMENT")
        api_version = get_env("AZURE_OPENAI_API_VERSION")

        # Remove trailing slash if present
        endpoint = endpoint.rstrip("/")

        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        async with client:
            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Reply briefly."},
                    {
                        "role": "user",
                        "content": "Say 'Azure OpenAI connection successful!' in one sentence.",
                    },
                ],
                max_tokens=50,
                temperature=0.7,
            )

        content = response.choices[0].message.content
        print(f"[OK] LLM Response: {content}")
        print(f"     Model: {response.model}")
        print(
            f"     Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total tokens"
        )
        return True

    except Exception as e:
        print(f"[FAIL] LLM Test Error: {type(e).__name__}: {e}")
        return False


async def test_embedding() -> bool:
    """Test Azure OpenAI Embedding connectivity."""
    print("\n[TEST] Embedding...")

    try:
        from openai import AsyncAzureOpenAI

        # Use embedding-specific env vars, fallback to LLM vars
        endpoint = get_env("AZURE_EMBEDDING_BINDING_HOST", "LLM_BINDING_HOST")
        api_key = get_env("AZURE_EMBEDDING_BINDING_API_KEY", "LLM_BINDING_API_KEY")
        deployment = get_env("AZURE_EMBEDDING_DEPLOYMENT")
        api_version = get_env("AZURE_EMBEDDING_API_VERSION")
        expected_dim = int(get_env("AZURE_EMBEDDING_DIM"))

        # Remove trailing slash if present
        endpoint = endpoint.rstrip("/")

        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        test_texts = ["Hello, world!", "Azure OpenAI embedding test."]

        async with client:
            response = await client.embeddings.create(
                model=deployment,
                input=test_texts,
            )

        embedding_dim = len(response.data[0].embedding)
        print(f"[OK] Embedding Response:")
        print(f"     Model: {response.model}")
        print(f"     Dimension: {embedding_dim} (expected: {expected_dim})")
        print(f"     Vectors: {len(response.data)} embeddings generated")
        print(f"     Usage: {response.usage.prompt_tokens} prompt tokens")

        if embedding_dim != expected_dim:
            print(f"[WARN] Dimension mismatch! Got {embedding_dim}, expected {expected_dim}")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Embedding Test Error: {type(e).__name__}: {e}")
        return False


async def main() -> int:
    """Run all tests."""
    load_env()
    print_config_summary()

    print("Starting Azure OpenAI Connection Tests...")
    print("-" * 60)

    llm_ok = await test_llm()
    emb_ok = await test_embedding()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  LLM (Chat Completion): {'[PASS]' if llm_ok else '[FAIL]'}")
    print(f"  Embedding:             {'[PASS]' if emb_ok else '[FAIL]'}")
    print("=" * 60)

    if llm_ok and emb_ok:
        print("\n[SUCCESS] All Azure OpenAI tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
