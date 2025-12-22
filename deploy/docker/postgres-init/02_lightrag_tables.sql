-- LightRAG PostgreSQL Storage Tables Initialization
-- This script creates the necessary tables for LightRAG PostgreSQL storage backends
-- Tables are created with IF NOT EXISTS to allow safe re-execution

-- Ensure pgvector extension is available (should already be created by 01_init.sql)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- PGKVStorage: Key-Value Storage Table
-- ============================================
-- Stores all key-value pairs (documents, chunks, entities, relations, cache, etc.)
CREATE TABLE IF NOT EXISTS lightrag_kv_storage (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster JSONB queries
CREATE INDEX IF NOT EXISTS idx_lightrag_kv_storage_value_gin ON lightrag_kv_storage USING GIN (value);

-- ============================================
-- PGVectorStorage: Vector Database Tables
-- ============================================
-- Table for chunk vectors
CREATE TABLE IF NOT EXISTS lightrag_vector_storage_chunks (
    id TEXT PRIMARY KEY,
    embedding vector(1536),  -- Default dimension, adjust if needed
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for entity vectors
CREATE TABLE IF NOT EXISTS lightrag_vector_storage_entities (
    id TEXT PRIMARY KEY,
    embedding vector(1536),  -- Default dimension, adjust if needed
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for relationship vectors
CREATE TABLE IF NOT EXISTS lightrag_vector_storage_relationships (
    id TEXT PRIMARY KEY,
    embedding vector(1536),  -- Default dimension, adjust if needed
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity search indexes (HNSW for fast approximate nearest neighbor search)
-- Note: These indexes require pgvector extension
CREATE INDEX IF NOT EXISTS idx_lightrag_vector_chunks_embedding ON lightrag_vector_storage_chunks 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_lightrag_vector_entities_embedding ON lightrag_vector_storage_entities 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_lightrag_vector_relationships_embedding ON lightrag_vector_storage_relationships 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- GIN indexes for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_lightrag_vector_chunks_metadata_gin ON lightrag_vector_storage_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_lightrag_vector_entities_metadata_gin ON lightrag_vector_storage_entities USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_lightrag_vector_relationships_metadata_gin ON lightrag_vector_storage_relationships USING GIN (metadata);

-- ============================================
-- PGDocStatusStorage: Document Status Table
-- ============================================
-- Tracks document processing status
CREATE TABLE IF NOT EXISTS lightrag_doc_status_storage (
    key TEXT PRIMARY KEY,  -- Document ID
    value JSONB NOT NULL,  -- Status information
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster JSONB queries
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_storage_value_gin ON lightrag_doc_status_storage USING GIN (value);

-- ============================================
-- Comments for documentation
-- ============================================
COMMENT ON TABLE lightrag_kv_storage IS 'LightRAG Key-Value Storage: Stores documents, chunks, entities, relations, and cache data';
COMMENT ON TABLE lightrag_vector_storage_chunks IS 'LightRAG Vector Storage: Chunk embeddings for semantic search';
COMMENT ON TABLE lightrag_vector_storage_entities IS 'LightRAG Vector Storage: Entity embeddings for semantic search';
COMMENT ON TABLE lightrag_vector_storage_relationships IS 'LightRAG Vector Storage: Relationship embeddings for semantic search';
COMMENT ON TABLE lightrag_doc_status_storage IS 'LightRAG Document Status Storage: Tracks document processing status';

