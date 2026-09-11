-- Runs once, on first container start.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,
    title       TEXT        NOT NULL DEFAULT '',
    meta        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT         NOT NULL,          -- position within the document
    content      TEXT        NOT NULL,
    -- char offsets back into the parent document, so a citation can point at an exact span
    char_start   INT         NOT NULL,
    char_end     INT         NOT NULL,
    section      TEXT        NOT NULL DEFAULT '',
    token_count  INT         NOT NULL DEFAULT 0,
    embedding    VECTOR(384),                   -- default profile: bge-small-en-v1.5
                                                -- swap EMBED_MODEL and the column is rebuilt automatically
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal)
);

-- Sparse (BM25-ish) side of hybrid retrieval. Postgres FTS keeps it to one datastore.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

-- Dense side. HNSW over cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
