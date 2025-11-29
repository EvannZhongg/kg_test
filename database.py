import asyncio
import hashlib
import logging
import os
from typing import List, Dict, Any, Optional
import asyncpg
from config import config
from dotenv import load_dotenv

# 确保加载环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)

env = os.getenv("FLASK_ENV", "default")
cfg = config.get(env, config['default'])

logger = logging.getLogger(__name__)


class DatabaseClient:
    _instance = None
    _pools = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseClient, cls).__new__(cls)
        return cls._instance

    @classmethod
    async def get_pool(cls):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        if current_loop in cls._pools:
            pool = cls._pools[current_loop]
            if not pool._closed:
                return pool
            else:
                del cls._pools[current_loop]

        try:
            pool = await asyncpg.create_pool(
                host=cfg.PG_HOST,
                port=cfg.PG_PORT,
                user=cfg.PG_USER,
                password=cfg.PG_PASSWORD,
                database=cfg.PG_DB,
                min_size=cfg.PG_MIN_SIZE,
                max_size=cfg.PG_MAX_SIZE,
                loop=current_loop
            )
            cls._pools[current_loop] = pool
            return pool
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    @classmethod
    async def close_current_pool(cls):
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop in cls._pools:
                pool = cls._pools[current_loop]
                await pool.close()
                del cls._pools[current_loop]
        except Exception as e:
            logger.error(f"关闭连接池失败: {e}")

    # === ID 生成工具 ===
    @staticmethod
    def generate_hash_id(content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    @classmethod
    def generate_file_hash(cls, file_content: str) -> str:
        return cls.generate_hash_id(file_content)

    @classmethod
    def generate_chunk_id(cls, file_hash: str, chunk_index: int, project_id: int) -> str:
        return f"{file_hash}_{project_id}_{chunk_index}"

    @classmethod
    def generate_node_id(cls, project_id: int, label: str) -> str:
        raw = f"{project_id}_{label.strip()}"
        return cls.generate_hash_id(raw)

    @classmethod
    def generate_edge_id(cls, project_id: int, src_label: str, tgt_label: str, relation: str) -> str:
        raw = f"{project_id}_{src_label.strip()}_{tgt_label.strip()}_{relation.strip()}"
        return cls.generate_hash_id(raw)

    # === 核心 Upsert 逻辑 (增量更新) ===

    async def upsert_chunk(self, project_id: int, file_id: int, chunk_id: str,
                           chunk_index: int, text: str, embedding: List[float],
                           file_name: str = None):  # <--- [修改1] 新增 file_name 参数
        embedding_str = str(embedding) if embedding else None

        sql = """
        INSERT INTO open_graph_chunks (id, project_id, file_id, chunk_index, text_content, embedding, file_name)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            text_content = EXCLUDED.text_content,
            embedding = EXCLUDED.embedding,
            file_name = EXCLUDED.file_name;
        """

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, chunk_id, project_id, file_id, chunk_index, text, embedding_str, file_name)

    async def get_nodes_map(self, project_id: int, labels: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取已存在实体的描述和权重，返回 {label: {'description': ..., 'weight': ...}}"""
        if not labels:
            return {}

        sql = """
        SELECT label, description, weight
        FROM open_graph_nodes 
        WHERE project_id = $1 AND label = ANY($2)
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, project_id, labels)
            return {row['label']: {'description': row['description'], 'weight': row['weight']} for row in rows}

    async def get_edges_map(self, project_id: int, edge_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取已存在关系的描述和出现次数，返回 {edge_id: {'description': ..., 'count': ...}}"""
        if not edge_ids:
            return {}

        sql = """
        SELECT edge_id, description, COALESCE(array_length(source_chunk_ids, 1), 0) as count
        FROM open_graph_edges 
        WHERE project_id = $1 AND edge_id = ANY($2)
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, project_id, edge_ids)
            return {row['edge_id']: {'description': row['description'], 'count': row['count']} for row in rows}

    # === 修改：Upsert 逻辑 (改为直接更新描述) ===

    async def upsert_node(self, project_id: int, label: str, entity_type: str,
                          description: str, source_chunk_id: str, embedding: List[float] = None) -> int:
        node_id = self.generate_node_id(project_id, label)
        embedding_str = str(embedding) if embedding else None

        sql = """
        INSERT INTO open_graph_nodes (project_id, label, node_id, entity_type, description, source_chunk_ids, weight, embedding)
        VALUES ($1, $2, $3, $4, $5, ARRAY[$6], 1, $7)
        ON CONFLICT (node_id) DO UPDATE SET
            weight = open_graph_nodes.weight + 1,
            source_chunk_ids = array_append(open_graph_nodes.source_chunk_ids, $6),
            description = EXCLUDED.description,
            embedding = EXCLUDED.embedding,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id;
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(sql, project_id, label, node_id, entity_type, description, source_chunk_id,
                                       embedding_str)

    async def upsert_edge(self, project_id: int, src_internal_id: int, tgt_internal_id: int,
                          src_label: str, tgt_label: str, relation: str,
                          description: str, weight: float, source_chunk_id: str, embedding: List[float] = None) -> \
    tuple[int, bool]:
        edge_id = self.generate_edge_id(project_id, src_label, tgt_label, relation)
        embedding_str = str(embedding) if embedding else None

        # 【核心修改】weight 计算方式改为累加：open_graph_edges.weight + $7 (传入的增量，通常为1)
        sql = """
        INSERT INTO open_graph_edges (project_id, source_id, target_id, relation, edge_id, description, weight, source_chunk_ids, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, ARRAY[$8], $9)
        ON CONFLICT (edge_id) DO UPDATE SET
            weight = open_graph_edges.weight + $7,
            source_chunk_ids = array_append(open_graph_edges.source_chunk_ids, $8),
            description = EXCLUDED.description,
            embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
        RETURNING id, (xmax = 0) AS is_insert;
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                sql, project_id, src_internal_id, tgt_internal_id, relation, edge_id, description, weight,
                source_chunk_id, embedding_str
            )
            return row['id'], row['is_insert']

    async def increment_nodes_degree(self, node_ids: List[int]):
        """批量增加节点的度 (Degree + 1)"""
        if not node_ids:
            return
        sql = """
        UPDATE open_graph_nodes 
        SET degree = degree + 1 
        WHERE id = ANY($1::bigint[])
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, node_ids)

    async def insert_provenance(self, project_id: int, task_id: str, chunk_id: str,
                                node_internal_id: Optional[int] = None,
                                edge_internal_id: Optional[int] = None):
        sql = """
        INSERT INTO open_graph_provenance (project_id, task_id, chunk_id, node_id, edge_id)
        VALUES ($1, $2, $3, $4, $5)
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(sql, project_id, task_id, chunk_id, node_internal_id, edge_internal_id)

db_client = DatabaseClient()