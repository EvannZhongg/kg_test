-- ===============================================================
-- 0. 清理旧表 (注意顺序)
-- ===============================================================
DROP TABLE IF EXISTS open_graph_provenance;
DROP TABLE IF EXISTS open_graph_edges;
DROP TABLE IF EXISTS open_graph_nodes;
DROP TABLE IF EXISTS open_graph_chunks; -- 新增
-- 清理限定抽取 (Limited Extraction) 的表
DROP TABLE IF EXISTS limited_graph_provenance;
DROP TABLE IF EXISTS limited_graph_edges;
DROP TABLE IF EXISTS limited_graph_nodes;
DROP TABLE IF EXISTS limited_graph_chunks;

-- 清理开放抽取 (Open Extraction) 的表
DROP TABLE IF EXISTS open_graph_provenance;
DROP TABLE IF EXISTS open_graph_edges;
DROP TABLE IF EXISTS open_graph_nodes;
DROP TABLE IF EXISTS open_graph_chunks;
-- 1. 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- ===============================================================
-- 2. 创建分块表 (Chunks) - 新增
-- 存储文档分块的原文和向量，避免在 provenance 中重复存储
-- ===============================================================
CREATE TABLE open_graph_chunks (
    id VARCHAR(255) PRIMARY KEY,          -- 唯一ID (格式: "docHash_index")
    project_id BIGINT NOT NULL,
    file_id BIGINT NOT NULL,
    file_name VARCHAR(512),
    chunk_index INT NOT NULL,
    text_content TEXT,                    -- 分块原文
    embedding vector(1024),               -- 分块向量 (维度可根据模型调整，如 1536, 768)

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_chunks_project ON open_graph_chunks(project_id);

-- ===============================================================
-- 3. 创建节点表 (Nodes)
-- 增加 embedding 字段
-- ===============================================================
CREATE TABLE open_graph_nodes (
    id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(255) NOT NULL,        -- 哈希ID
    project_id BIGINT NOT NULL,

    label VARCHAR(512) NOT NULL,
    entity_type VARCHAR(255) DEFAULT 'Unknown',
    description TEXT,

    embedding vector(1024),               -- 【新增】实体向量

    -- 图计算字段
    weight INTEGER DEFAULT 1,
    degree INTEGER DEFAULT 0,
    pagerank_score DOUBLE PRECISION DEFAULT 0.0,
    community_id INTEGER DEFAULT -1,

    source_chunk_ids TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_node_id UNIQUE (node_id),
    CONSTRAINT uk_project_node_label UNIQUE (project_id, label)
);

CREATE INDEX idx_open_nodes_project ON open_graph_nodes(project_id);

-- ===============================================================
-- 4. 创建边表 (Edges)
-- 增加 embedding 字段
-- ===============================================================
CREATE TABLE open_graph_edges (
    id BIGSERIAL PRIMARY KEY,
    edge_id VARCHAR(255) NOT NULL,        -- 哈希ID
    project_id BIGINT NOT NULL,

    source_id BIGINT NOT NULL REFERENCES open_graph_nodes(id) ON DELETE CASCADE,
    target_id BIGINT NOT NULL REFERENCES open_graph_nodes(id) ON DELETE CASCADE,
    relation VARCHAR(255) NOT NULL,

    description TEXT,
    weight DOUBLE PRECISION DEFAULT 1.0,

    embedding vector(1024),               -- 【新增】关系向量

    source_chunk_ids TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_edge_id UNIQUE (edge_id),
    CONSTRAINT uk_project_edge UNIQUE (project_id, source_id, target_id, relation)
);

CREATE INDEX idx_open_edges_project ON open_graph_edges(project_id);

-- ===============================================================
-- 5. 创建溯源表 (Provenance)
-- 关联 chunk_id，不再存储重复文本
-- ===============================================================
CREATE TABLE open_graph_provenance (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL,
    task_id VARCHAR(64) NOT NULL,

    chunk_id VARCHAR(255) NOT NULL REFERENCES open_graph_chunks(id) ON DELETE CASCADE,

    node_id BIGINT REFERENCES open_graph_nodes(id) ON DELETE CASCADE,
    edge_id BIGINT REFERENCES open_graph_edges(id) ON DELETE CASCADE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_provenance_chunk ON open_graph_provenance(chunk_id);



-- ===============================================================
-- 6. 限定抽取 - 分块表 (Limited Chunks)
-- ===============================================================
CREATE TABLE limited_graph_chunks (
    id VARCHAR(255) PRIMARY KEY,          -- 格式: "fileHash_projectId_index"
    project_id BIGINT NOT NULL,
    file_id BIGINT NOT NULL,              -- 对应 material_id
    file_name VARCHAR(512),
    chunk_index INT NOT NULL,
    text_content TEXT,
    embedding vector(1024),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_limited_chunks_project ON limited_graph_chunks(project_id);

-- ===============================================================
-- 7. 限定抽取 - 节点表 (Limited Nodes)
-- ===============================================================
CREATE TABLE limited_graph_nodes (
    id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(255) NOT NULL,        -- 哈希ID
    project_id BIGINT NOT NULL,

    label VARCHAR(512) NOT NULL,
    entity_type VARCHAR(255) DEFAULT 'Unknown',
    description TEXT,

    embedding vector(1024),

    weight INTEGER DEFAULT 1,             -- 实体出现频次 (权重)
    degree INTEGER DEFAULT 0,             -- 【新增】节点度 (连接的边数)

    source_chunk_ids TEXT[],              -- 溯源 Chunk ID 列表

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_limited_node_id UNIQUE (node_id)
);
CREATE INDEX idx_limited_nodes_project ON limited_graph_nodes(project_id);

-- ===============================================================
-- 8. 限定抽取 - 边表 (Limited Edges)
-- ===============================================================
CREATE TABLE limited_graph_edges (
    id BIGSERIAL PRIMARY KEY,
    edge_id VARCHAR(255) NOT NULL,
    project_id BIGINT NOT NULL,

    source_id BIGINT NOT NULL REFERENCES limited_graph_nodes(id) ON DELETE CASCADE,
    target_id BIGINT NOT NULL REFERENCES limited_graph_nodes(id) ON DELETE CASCADE,

    relation VARCHAR(255) NOT NULL,       -- 对应 relationship.label
    relation_type VARCHAR(255),           -- 【新增】对应 relationship.type

    description TEXT,                     -- 预留
    weight DOUBLE PRECISION DEFAULT 1.0,

    embedding vector(1024),
    source_chunk_ids TEXT[],

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_limited_edge_id UNIQUE (edge_id)
);
CREATE INDEX idx_limited_edges_project ON limited_graph_edges(project_id);

-- ===============================================================
-- 9. 限定抽取 - 溯源表 (Limited Provenance)
-- ===============================================================
CREATE TABLE limited_graph_provenance (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL,
    task_id VARCHAR(64) NOT NULL,

    chunk_id VARCHAR(255) NOT NULL REFERENCES limited_graph_chunks(id) ON DELETE CASCADE,
    node_id BIGINT REFERENCES limited_graph_nodes(id) ON DELETE CASCADE,
    edge_id BIGINT REFERENCES limited_graph_edges(id) ON DELETE CASCADE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_limited_provenance_chunk ON limited_graph_provenance(chunk_id);