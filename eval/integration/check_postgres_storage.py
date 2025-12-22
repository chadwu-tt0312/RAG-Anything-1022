"""檢查 PostgreSQL storage 的實際狀態

這個腳本用於診斷 PostgreSQL storage 是否正確設定和使用。

Usage:
    export POSTGRES_HOST=docker-postgres-1
    export POSTGRES_PORT=5432
    export POSTGRES_USER=rag_user
    export POSTGRES_PASSWORD=rag_password
    export POSTGRES_DATABASE=rag_db
    python eval/integration/check_postgres_storage.py
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
    from psycopg2.extras import RealDictCursor
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
    return conn


def check_tables(conn):
    """檢查 LightRAG 相關表格是否存在"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'lightrag_%'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        return [t[0] for t in tables]


def check_doc_status(conn, doc_id: str = None):
    """檢查 doc_status_storage 中的記錄"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if doc_id:
            cur.execute(
                "SELECT key, value, created_at, updated_at FROM lightrag_doc_status_storage WHERE key = %s",
                (doc_id,)
            )
        else:
            cur.execute(
                "SELECT key, value, created_at, updated_at FROM lightrag_doc_status_storage ORDER BY created_at DESC LIMIT 10"
            )
        return cur.fetchall()


def check_kv_storage(conn, limit: int = 10):
    """檢查 kv_storage 中的記錄"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT key, created_at, updated_at FROM lightrag_kv_storage ORDER BY created_at DESC LIMIT {limit}"
        )
        return cur.fetchall()


def check_vector_storage(conn, table_name: str, limit: int = 10):
    """檢查 vector_storage 中的記錄"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT id, created_at, updated_at FROM {table_name} ORDER BY created_at DESC LIMIT {limit}"
        )
        return cur.fetchall()


def main():
    """Main function"""
    try:
        print("🔌 Connecting to PostgreSQL...")
        conn = get_connection()
        print("✓ Connected successfully\n")

        # Check tables
        print("📋 Checking LightRAG tables...")
        tables = check_tables(conn)
        if tables:
            print(f"✓ Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
        else:
            print("⚠️  No LightRAG tables found!")
            print("   Run: python eval/integration/init_postgres_tables.py")
            return
        print()

        # Check doc_status_storage
        if "lightrag_doc_status_storage" in tables:
            print("📄 Checking doc_status_storage...")
            doc_statuses = check_doc_status(conn)
            if doc_statuses:
                print(f"✓ Found {len(doc_statuses)} document status records:")
                for status in doc_statuses:
                    print(f"  - Key: {status['key']}")
                    print(f"    Created: {status['created_at']}")
                    print(f"    Value: {status['value']}")
            else:
                print("  ⚠️  No document status records found")
            print()

        # Check kv_storage
        if "lightrag_kv_storage" in tables:
            print("🗄️  Checking kv_storage...")
            kv_records = check_kv_storage(conn)
            if kv_records:
                print(f"✓ Found {len(kv_records)} KV records (showing first 10):")
                for record in kv_records:
                    print(f"  - Key: {record['key']}")
                    print(f"    Created: {record['created_at']}")
            else:
                print("  ⚠️  No KV records found")
            print()

        # Check vector storage
        vector_tables = [
            "lightrag_vector_storage_chunks",
            "lightrag_vector_storage_entities",
            "lightrag_vector_storage_relationships",
        ]
        for table_name in vector_tables:
            if table_name in tables:
                print(f"🔢 Checking {table_name}...")
                vector_records = check_vector_storage(conn, table_name)
                if vector_records:
                    print(f"✓ Found {len(vector_records)} vector records (showing first 10):")
                    for record in vector_records:
                        print(f"  - ID: {record['id']}")
                        print(f"    Created: {record['created_at']}")
                else:
                    print(f"  ⚠️  No vector records found in {table_name}")
                print()

        # Check specific doc_id if provided
        doc_id = os.getenv("CHECK_DOC_ID")
        if doc_id:
            print(f"🔍 Checking specific doc_id: {doc_id}")
            if "lightrag_doc_status_storage" in tables:
                status = check_doc_status(conn, doc_id)
                if status:
                    print(f"✓ Found doc_status for {doc_id}:")
                    for s in status:
                        print(f"  Value: {s['value']}")
                else:
                    print(f"  ⚠️  No doc_status found for {doc_id}")
            print()

        conn.close()
        print("✅ Done!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

