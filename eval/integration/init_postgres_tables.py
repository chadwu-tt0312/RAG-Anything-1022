"""Initialize PostgreSQL tables for LightRAG storage backends.

This script creates the necessary tables for LightRAG PostgreSQL storage:
- PGKVStorage
- PGVectorStorage  
- PGDocStatusStorage

Usage:
    python eval/integration/init_postgres_tables.py

Environment variables required:
    POSTGRES_HOST
    POSTGRES_PORT
    POSTGRES_USER
    POSTGRES_PASSWORD
    POSTGRES_DATABASE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _get_postgres_database() -> str:
    """取得 PostgreSQL 資料庫名稱，支援 POSTGRES_DATABASE 和 POSTGRES_DB"""
    db_name = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    if not db_name:
        raise RuntimeError("Missing env var: POSTGRES_DATABASE or POSTGRES_DB")
    return db_name


def get_connection():
    """Create PostgreSQL connection"""
    host = _require_env("POSTGRES_HOST")
    port = _require_env("POSTGRES_PORT")
    user = _require_env("POSTGRES_USER")
    password = _require_env("POSTGRES_PASSWORD")
    database = _get_postgres_database()

    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn


def check_extension(conn, extension_name: str) -> bool:
    """Check if PostgreSQL extension exists"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = %s)",
            (extension_name,),
        )
        return cur.fetchone()[0]


def create_extension(conn, extension_name: str):
    """Create PostgreSQL extension if it doesn't exist"""
    if check_extension(conn, extension_name):
        print(f"✓ Extension '{extension_name}' already exists")
        return

    with conn.cursor() as cur:
        cur.execute(f"CREATE EXTENSION IF NOT EXISTS {extension_name}")
    print(f"✓ Created extension '{extension_name}'")


def check_table(conn, table_name: str) -> bool:
    """Check if table exists"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
            """,
            (table_name,),
        )
        return cur.fetchone()[0]


def create_tables(conn, embedding_dim: int = 1536):
    """Create LightRAG PostgreSQL storage tables"""

    # Create pgvector extension
    create_extension(conn, "vector")

    with conn.cursor() as cur:
        # PGKVStorage table
        if check_table(conn, "lightrag_kv_storage"):
            print("✓ Table 'lightrag_kv_storage' already exists")
        else:
            cur.execute("""
                CREATE TABLE lightrag_kv_storage (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX idx_lightrag_kv_storage_value_gin 
                ON lightrag_kv_storage USING GIN (value)
            """)
            print("✓ Created table 'lightrag_kv_storage'")

        # PGVectorStorage tables
        vector_tables = [
            ("lightrag_vector_storage_chunks", "chunks"),
            ("lightrag_vector_storage_entities", "entities"),
            ("lightrag_vector_storage_relationships", "relationships"),
        ]

        for table_name, description in vector_tables:
            if check_table(conn, table_name):
                print(f"✓ Table '{table_name}' already exists")
            else:
                cur.execute(f"""
                    CREATE TABLE {table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({embedding_dim}),
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Create HNSW index for vector similarity search
                cur.execute(f"""
                    CREATE INDEX idx_{table_name}_embedding 
                    ON {table_name} 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64)
                """)
                # Create GIN index for metadata
                cur.execute(f"""
                    CREATE INDEX idx_{table_name}_metadata_gin 
                    ON {table_name} USING GIN (metadata)
                """)
                print(f"✓ Created table '{table_name}' ({description})")

        # PGDocStatusStorage table
        if check_table(conn, "lightrag_doc_status_storage"):
            print("✓ Table 'lightrag_doc_status_storage' already exists")
        else:
            cur.execute("""
                CREATE TABLE lightrag_doc_status_storage (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX idx_lightrag_doc_status_storage_value_gin 
                ON lightrag_doc_status_storage USING GIN (value)
            """)
            print("✓ Created table 'lightrag_doc_status_storage'")

    print("\n✅ All LightRAG PostgreSQL tables initialized successfully!")


def list_tables(conn):
    """List all LightRAG-related tables"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'lightrag_%'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        if tables:
            print("\n📋 Existing LightRAG tables:")
            for (table_name,) in tables:
                print(f"  - {table_name}")
        else:
            print("\n⚠️  No LightRAG tables found")


def main():
    """Main function"""
    try:
        print("🔌 Connecting to PostgreSQL...")
        conn = get_connection()
        print("✓ Connected successfully\n")

        # Get embedding dimension from env or use default
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        print(f"📐 Using embedding dimension: {embedding_dim}\n")

        # List existing tables
        list_tables(conn)

        # Create tables
        print("\n🔨 Creating tables...")
        create_tables(conn, embedding_dim)

        # List tables again
        list_tables(conn)

        conn.close()
        print("\n✅ Done!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

