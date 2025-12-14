import asyncio
import uuid
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from triplet_extraction_service import TaskFile
from database import db_client
from prompt import PROMPTS
from extract_triplets_from_docx import (
    read_txt_text, chunk_text, make_client, call_llm,
    parse_lightrag_output
)
from embedding_service import embedding_service

logger = logging.getLogger(__name__)


@dataclass
class OpenIETaskResult:
    task_id: str
    status: str
    total_files: int
    processed_files: int
    failed_files: int
    total_triples: int
    processing_time: str
    results: List[TaskFile]
    errors: List[Dict[str, str]]
    created_at: str
    updated_at: str
    project_id: int = 0

    def to_dict(self):
        return asdict(self)


class OpenIEExtractionService:
    def __init__(self, output_base_dir: str = "output", max_workers: int = 3):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: Dict[str, OpenIETaskResult] = {}
        self.task_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_futures: Dict[str, Any] = {}
        self.cancelled_tasks = set()

    def create_task(self, files: List[Dict[str, Any]], provider: str, project_id: int,
                    model: Optional[str] = None, api_key: Optional[str] = None,
                    base_url: Optional[str] = None) -> str:
        # (保持不变)
        if project_id <= 0: raise ValueError("OpenIE 任务必须提供有效的 project_id")
        valid_files = []
        for file_item in files:
            if isinstance(file_item, dict):
                file_path, material_id = file_item.get('url'), file_item.get('material_id')
                # 提取原始文件名
                original_file_name = file_item.get('file_name') or Path(file_path).name
                if not material_id: raise ValueError(f"OpenIE 模式下文件必须包含 material_id: {file_path}")
            else:
                raise ValueError("OpenIE 模式不支持纯路径字符串")
            path = Path(file_path)
            if not path.exists(): raise FileNotFoundError(f"文件不存在: {file_path}")
            valid_files.append({
                'path': str(path),
                'material_id': material_id,
                'file_name': original_file_name
            })

        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        task_result = OpenIETaskResult(task_id=task_id, status="processing", total_files=len(valid_files),
                                       processed_files=0, failed_files=0, total_triples=0, processing_time="0s",
                                       results=[], errors=[], created_at=now, updated_at=now, project_id=project_id)
        with self.task_lock:
            self.tasks[task_id] = task_result
        future = self.executor.submit(self._async_entry_point, task_id, valid_files, provider, model, api_key, base_url,
                                      project_id)
        self.task_futures[task_id] = future
        return task_id

    def _async_entry_point(self, *args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._process_task_async(*args, **kwargs))
        finally:
            try:
                loop.run_until_complete(db_client.close_current_pool())
            except Exception as e:
                logger.error(f"清理失败: {e}")
            loop.close()

    async def _process_task_async(self, task_id: str, files: List[Dict[str, Any]],
                                  provider: str, model: Optional[str], api_key: Optional[str],
                                  base_url: Optional[str], project_id: int):
        # (保持不变)
        start_time = time.time()
        await db_client.get_pool()
        try:
            if task_id in self.cancelled_tasks: return
            resolved_base_url, resolved_api_key, resolved_model = make_client(provider, model, base_url, api_key)
            results, errors = [], []
            total_extracted_count, processed_count, failed_count = 0, 0, 0

            for i, file_info in enumerate(files, 1):
                if task_id in self.cancelled_tasks: return
                file_path, file_id = file_info['path'], file_info['material_id']
                original_file_name = file_info.get('file_name') or Path(file_path).name

                logger.info(f"[OpenIE] 处理 {original_file_name}")
                try:
                    file_stats = await self._process_single_file(file_path, file_id, project_id, task_id,
                                                                 resolved_base_url, resolved_api_key, resolved_model,
                                                                 original_file_name=original_file_name)
                    results.append(TaskFile(file_name=original_file_name, material_id=file_id, status='success',
                                            triples_count=file_stats['count'], output_files={}))
                    total_extracted_count += file_stats['count']
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    results.append(
                        TaskFile(file_name=original_file_name, material_id=file_id, status='failed', error=str(e)))
                    failed_count += 1
                self._update_task_progress(task_id, processed_count, failed_count, total_extracted_count, results,
                                           errors)

            final_status = "completed" if failed_count < len(files) else "failed"
            with self.task_lock:
                if task_id in self.tasks:
                    t = self.tasks[task_id]
                    t.status = final_status
                    t.processing_time = self._format_duration(time.time() - start_time)
                    t.updated_at = datetime.now(timezone.utc).isoformat()
        finally:
            if task_id in self.task_futures: del self.task_futures[task_id]
            self.cancelled_tasks.discard(task_id)

    async def _summarize_description(self, name: str, old_desc: str, new_desc: str,
                                     base_url: str, api_key: str, model: str) -> str:
        """调用 LLM 合并描述"""
        if not old_desc or old_desc == new_desc:
            return new_desc

        prompt = PROMPTS["summarize_entity_descriptions"].format(
            entity_name=name,
            existing_desc=old_desc,
            new_desc=new_desc
        )

        try:
            loop = asyncio.get_running_loop()
            # 复用 call_llm
            summary = await loop.run_in_executor(
                None, call_llm, base_url, api_key, model, prompt, ""
            )
            # 清理可能的 markdown 标记
            return summary.replace("```", "").strip()
        except Exception as e:
            logger.warning(f"合并描述失败，保留新描述: {e}")
            return new_desc

    async def _process_single_file(self, file_path: str, file_id: int, project_id: int, task_id: str,
                                   base_url: str, api_key: str, model: str,
                                   original_file_name: str = None) -> Dict[str, Any]:
        """
        分块 -> 向量化 -> 【并发抽取】 -> 【即时串行入库】
        """
        path_obj = Path(file_path)
        file_name = original_file_name if original_file_name else path_obj.name

        text = read_txt_text(path_obj)
        file_hash = db_client.generate_file_hash(text)
        chunks = chunk_text(text)

        logger.info(f"已将文本切分为 {len(chunks)} 个 Chunk，准备开始处理...")
        print(f"\n>>> [文件开始] {file_name} (共 {len(chunks)} 个 Chunk)")

        # 1. Chunk 批量向量化 (保持批量处理以提高效率)
        loop = asyncio.get_running_loop()
        t_chunk_embed = time.time()
        try:
            chunk_embeddings = await loop.run_in_executor(None, lambda: embedding_service.get_embeddings(chunks))
        except:
            chunk_embeddings = [None] * len(chunks)
        print(f"  [Time] 批量计算 {len(chunks)} 个 Chunk 向量: {time.time() - t_chunk_embed:.2f}s")

        # 2. 准备 Prompt
        prompt_tpl = PROMPTS["open_ie_extraction_system_prompt"]
        instruction = prompt_tpl.format(
            language="Chinese", tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"], input_text=""
        ).split("---Real Data")[0].strip()

        # === 并发控制配置 ===
        CONCURRENCY_LIMIT = 5  # 限制同时进行 LLM 抽取的数量，防止 429
        extract_sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        ingest_lock = asyncio.Lock()  # 数据库写入锁，确保融合逻辑的原子性

        extracted_count_container = {'count': 0}
        MERGE_THRESHOLD = 5

        # === 定义单个 Chunk 的处理管线 ===
        async def process_one_chunk_pipeline(idx, chunk, embedding):
            chunk_unique_id = db_client.generate_chunk_id(file_hash, idx, project_id)

            # --- Phase 1: 并发抽取 (Extraction) ---
            # 这一步是耗时大户，允许并行执行
            async with extract_sem:
                print(f"--- [Chunk {idx + 1}/{len(chunks)}] 开始抽取")

                # 先保存 Chunk 原文 (IO操作)
                # 传递 file_name 参数
                await db_client.upsert_chunk(project_id, file_id, chunk_unique_id, idx, chunk, embedding,
                                             file_name=file_name)

                # 调用 LLM 进行抽取 (IO操作，最慢的部分)
                t_extract = time.time()
                llm_output = await loop.run_in_executor(
                    None, call_llm, base_url, api_key, model, instruction, chunk
                )

                nodes, edges = parse_lightrag_output(llm_output)
                duration = time.time() - t_extract
                # print(f"  ✓ [Chunk {idx + 1}] 抽取完成 ({duration:.2f}s)")

            if not nodes and not edges:
                print(f"  [Chunk {idx + 1}] 未抽取到信息，跳过入库。")
                return

            # --- Phase 2: 即时串行入库 (Ingestion) ---
            # 获取锁，确保 "查询旧值 -> 融合 -> 写入新值" 这一过程是独占的，防止覆盖
            async with ingest_lock:
                print(f"  >>> [Chunk {idx + 1}] 正在入库 (持有锁)...")

                # 4.1 内部去重 (Chunk 内)
                unique_nodes_map = {}
                for n in nodes:
                    if n['label'] not in unique_nodes_map or len(n['description']) > len(
                            unique_nodes_map[n['label']]['description']):
                        unique_nodes_map[n['label']] = n

                # 补全关系实体
                for e in edges:
                    for role in ['source', 'target']:
                        if e[role] not in unique_nodes_map:
                            unique_nodes_map[e[role]] = {'label': e[role], 'type': 'Unknown',
                                                         'description': f"Entity in {e['relation']}"}

                final_nodes = list(unique_nodes_map.values())

                # 4.2 数据库查重
                node_labels = [n['label'] for n in final_nodes]
                existing_nodes_map = await db_client.get_nodes_map(project_id, node_labels)

                # 4.3 实体描述融合
                merge_tasks = []
                for node in final_nodes:
                    label = node['label']
                    existing_info = existing_nodes_map.get(label)

                    if existing_info:
                        if isinstance(existing_info, dict):
                            old_desc = existing_info.get('description', '')
                            current_weight = existing_info.get('weight', 0)
                        else:
                            old_desc = str(existing_info)
                            current_weight = 0

                        # 达到阈值调用 LLM 摘要，否则直接拼接
                        if (current_weight + 1) % MERGE_THRESHOLD == 0:
                            merge_tasks.append(
                                self._summarize_description(label, old_desc, node['description'], base_url, api_key,
                                                            model)
                            )
                        else:
                            f = asyncio.Future()
                            concatenated = f"{old_desc}\n{node['description']}"
                            f.set_result(concatenated)
                            merge_tasks.append(f)
                    else:
                        f = asyncio.Future()
                        f.set_result(node['description'])
                        merge_tasks.append(f)

                merged_descriptions = await asyncio.gather(*merge_tasks)
                for i, node in enumerate(final_nodes):
                    node['description'] = merged_descriptions[i]

                # 5. 关系融合
                edge_ids = [db_client.generate_edge_id(project_id, e['source'], e['target'], e['relation']) for e in
                            edges]
                existing_edges_map = await db_client.get_edges_map(project_id, edge_ids)

                edge_merge_tasks = []
                for i, edge in enumerate(edges):
                    eid = edge_ids[i]
                    existing_info = existing_edges_map.get(eid)
                    edge_name = f"{edge['source']} {edge['relation']} {edge['target']}"

                    if existing_info:
                        if isinstance(existing_info, dict):
                            old_desc = existing_info.get('description', '')
                            current_count = existing_info.get('count', 0)
                        else:
                            old_desc = str(existing_info)
                            current_count = 0

                        if (current_count + 1) % MERGE_THRESHOLD == 0:
                            edge_merge_tasks.append(
                                self._summarize_description(edge_name, old_desc, edge['description'], base_url, api_key,
                                                            model)
                            )
                        else:
                            f = asyncio.Future()
                            concatenated = f"{old_desc}\n{edge['description']}"
                            f.set_result(concatenated)
                            edge_merge_tasks.append(f)
                    else:
                        f = asyncio.Future()
                        f.set_result(edge['description'])
                        edge_merge_tasks.append(f)

                merged_edge_descriptions = await asyncio.gather(*edge_merge_tasks)
                for i, edge in enumerate(edges):
                    edge['description'] = merged_edge_descriptions[i]

                # 6. 重新向量化 (Re-Embedding)
                # 由于描述已更新，需要重新计算向量。此操作仍在锁内，确保写入的数据与描述一致。
                node_texts = [f"{n['label']} {n['description']}" for n in final_nodes]
                edge_texts = [f"{e['source']} {e['relation']} {e['target']}: {e['description']}" for e in edges]
                all_texts = node_texts + edge_texts

                try:
                    all_vecs = await loop.run_in_executor(
                        None, lambda: embedding_service.get_embeddings(all_texts)
                    )
                except:
                    all_vecs = [None] * len(all_texts)

                node_vecs = dict(zip([n['label'] for n in final_nodes], all_vecs[:len(final_nodes)]))
                edge_vecs = all_vecs[len(final_nodes):]

                # 7. 写入数据库 (Upsert)
                node_id_map = {}
                for node in final_nodes:
                    label = node['label']
                    internal_id = await db_client.upsert_node(
                        project_id, label, node['type'], node['description'],
                        chunk_unique_id, embedding=node_vecs.get(label)
                    )
                    node_id_map[label] = internal_id
                    await db_client.insert_provenance(project_id, task_id, chunk_unique_id,
                                                      node_internal_id=internal_id)
                    extracted_count_container['count'] += 1

                nodes_to_inc_degree = []
                for i, edge in enumerate(edges):
                    src_id = node_id_map.get(edge['source'])
                    tgt_id = node_id_map.get(edge['target'])

                    if src_id and tgt_id:
                        edge_internal_id, is_new = await db_client.upsert_edge(
                            project_id, src_id, tgt_id,
                            edge['source'], edge['target'],
                            edge['relation'], edge['description'], 1.0,
                            chunk_unique_id, embedding=edge_vecs[i]
                        )
                        if is_new: nodes_to_inc_degree.extend([src_id, tgt_id])
                        await db_client.insert_provenance(project_id, task_id, chunk_unique_id,
                                                          edge_internal_id=edge_internal_id)
                        extracted_count_container['count'] += 1

                if nodes_to_inc_degree:
                    await db_client.increment_nodes_degree(nodes_to_inc_degree)

                print(f"  ✓ [Chunk {idx + 1}] 入库完毕 (释放锁)")

        # 创建所有任务并并发执行
        tasks = [process_one_chunk_pipeline(i, chunk, chunk_embeddings[i]) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)

        return {'count': extracted_count_container['count']}

    def _update_task_progress(self, task_id, processed, failed, total_triples, results, errors):
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.processed_files = processed
                task.failed_files = failed
                task.total_triples = total_triples
                task.results = results[:]
                task.errors = errors[:]
                task.updated_at = datetime.now(timezone.utc).isoformat()

    def _update_task_status(self, task_id, status):
        with self.task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                self.tasks[task_id].updated_at = datetime.now(timezone.utc).isoformat()

    def get_task_status(self, task_id):
        with self.task_lock:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
        return None

    def get_task_list(self, status: Optional[str] = None, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        with self.task_lock:
            tasks = list(self.tasks.values())
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        if status:
            tasks = [t for t in tasks if t.status == status]
        total = len(tasks)
        tasks = tasks[offset:offset + limit]
        return {'tasks': [task.to_dict() for task in tasks], 'total': total, 'limit': limit, 'offset': offset}

    def cancel_task(self, task_id):
        with self.task_lock:
            if task_id in self.tasks:
                self.cancelled_tasks.add(task_id)
                self.tasks[task_id].status = "cancelled"
                return True
        return False

    @staticmethod
    def _format_duration(seconds):
        return f"{int(seconds)}s" if seconds < 60 else f"{int(seconds // 60)}m{int(seconds % 60)}s"

    def shutdown(self):
        self.executor.shutdown(wait=True)


open_ie_service_instance = None


def get_open_ie_service() -> OpenIEExtractionService:
    global open_ie_service_instance
    if open_ie_service_instance is None:
        open_ie_service_instance = OpenIEExtractionService()
    return open_ie_service_instance