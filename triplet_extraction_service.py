import asyncio
import json
import os
import uuid
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import threading
from queue import Queue
from database import db_client
from extract_triplets_from_docx import (
    read_txt_text, chunk_text, make_client, call_llm,
    parse_triples_from_text, PROVIDERS, save_outputs
)
from embedding_service import embedding_service

@dataclass
class TaskFile:
    file_name: str
    status: str  # success, failed, processing, pending
    material_id: int = None
    triples_count: int = 0
    output_files: Dict[str, str] = None
    error: str = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}


@dataclass
class TaskResult:
    task_id: str
    status: str  # completed, processing, failed, cancelled
    total_files: int
    processed_files: int
    failed_files: int
    total_triples: int
    processing_time: str
    results: List[TaskFile]
    errors: List[Dict[str, str]]
    created_at: str
    updated_at: str

    def to_dict(self):
        return asdict(self)


class TripletsExtractionService:
    def __init__(self, output_base_dir: str = "output", max_workers: int = 3):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 任务存储
        self.tasks: Dict[str, TaskResult] = {}
        self.task_lock = threading.Lock()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

        # 任务队列和状态管理
        self.task_futures: Dict[str, Any] = {}
        self.cancelled_tasks = set()

    def create_task(self,
                    files: Union[str, List[str], List[Dict[str, Any]]],
                    prompt_text: str,
                    provider: str = "deepseek",
                    project_id: int = 0,
                    model: Optional[str] = None,
                    api_key: Optional[str] = None,
                    base_url: Optional[str] = None) -> str:

        if project_id <= 0:
            raise ValueError("Limited Extraction 任务必须提供有效的 project_id")
        """创建新的抽取任务

        Args:
            files: 可以是以下格式之一：
                - 单个文件路径字符串
                - 文件路径字符串列表
                - 包含material_id和url的字典列表: [{"material_id": 1, "url": "/path/to/file"}, ...]
        """

        # 参数验证
        if provider not in PROVIDERS:
            raise ValueError(f"不支持的提供商: {provider}")

        if isinstance(files, str):
            files = [files]

        # 验证文件存在性，同时保存material_id信息
        valid_files = []
        for file_item in files:
            # 支持字符串或字典格式
            if isinstance(file_item, dict):
                file_path = file_item.get('url')
                material_id = file_item.get('material_id')
                # 【新增】提取 file_name，如果没有则默认为 None
                original_file_name = file_item.get('file_name')
            else:
                file_path = file_item
                material_id = None
                original_file_name = None

            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 【修改】将 file_name 一并存入 valid_files
            valid_files.append({
                'path': str(path),
                'material_id': material_id,
                'file_name': original_file_name  # 保存原始文件名
            })

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务结果对象
        now = datetime.now(timezone.utc).isoformat()
        task_result = TaskResult(
            task_id=task_id,
            status="processing",
            total_files=len(valid_files),
            processed_files=0,
            failed_files=0,
            total_triples=0,
            processing_time="0s",
            results=[],
            errors=[],
            created_at=now,
            updated_at=now
        )

        # 存储任务
        with self.task_lock:
            self.tasks[task_id] = task_result

        # 提交异步任务
        future = self.executor.submit(
            self._process_task,
            task_id, valid_files, prompt_text, provider, model, api_key, base_url, project_id  # 传入 project_id
        )
        self.task_futures[task_id] = future

        return task_id

    def _process_task(self, task_id: str, files: List[Dict[str, Any]], prompt_text: str,
                      provider: str, model: Optional[str], api_key: Optional[str],
                      base_url: Optional[str], project_id: int):
        """处理任务的内部方法（运行在独立线程中）"""
        start_time = time.time()

        # [新增] 1. 初始化 Asyncio Loop 和数据库连接
        # 因为 ThreadPoolExecutor 中的线程没有默认的 event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 初始化数据库连接池
            loop.run_until_complete(db_client.get_pool())

            # 检查任务是否被取消
            if task_id in self.cancelled_tasks:
                self._update_task_status(task_id, "cancelled")
                return

            # 初始化AI服务配置
            resolved_base_url, resolved_api_key, resolved_model = make_client(
                provider, model, base_url, api_key
            )

            # 创建任务专用输出目录 (保留文件备份)
            task_output_dir = self.output_base_dir / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)

            results = []
            errors = []
            total_triples = 0
            processed_count = 0
            failed_count = 0

            for i, file_info in enumerate(files, 1):
                # ... (取消检查逻辑保持不变) ...

                file_path = file_info['path']
                material_id = file_info.get('material_id', 0)

                # 【修改】优先使用传入的 file_name，如果为空则回退到使用路径文件名
                file_name = file_info.get('file_name') or Path(file_path).name

                print(f"[任务{task_id}] 处理文件 {i}/{len(files)}: {file_name} (material_id: {material_id})")

                try:

                    file_result = loop.run_until_complete(
                        self._process_single_file_async(
                            file_path, material_id, prompt_text,
                            resolved_base_url, resolved_api_key, resolved_model,
                            task_output_dir, project_id, task_id,
                            original_file_name=file_name  # <--- 新增参数传递
                        )
                    )

                    if file_result['status'] == 'success':
                        results.append(TaskFile(
                            file_name=file_name,
                            material_id=material_id,
                            status='success',
                            triples_count=file_result['triples_count'],
                            output_files=file_result.get('output_files', {})
                        ))
                        total_triples += file_result['triples_count']
                        processed_count += 1
                    else:
                        results.append(TaskFile(
                            file_name=file_name,
                            material_id=material_id,
                            status='failed',
                            error=file_result['error']
                        ))
                        errors.append({
                            'file_name': file_name,
                            'material_id': material_id,
                            'error': file_result['error']
                        })
                        failed_count += 1

                except Exception as e:
                    error_msg = f"处理文件时发生异常: {str(e)}"
                    print(f"[Error] {file_name}: {error_msg}")
                    results.append(TaskFile(
                        file_name=file_name,
                        material_id=material_id,
                        status='failed',
                        error=error_msg
                    ))
                    errors.append({
                        'file_name': file_name,
                        'material_id': material_id,
                        'error': error_msg
                    })
                    failed_count += 1

                # 更新进度
                self._update_task_progress(task_id, processed_count, failed_count, total_triples, results, errors)

            # 计算处理时间
            end_time = time.time()
            processing_time = self._format_duration(end_time - start_time)

            # 最终状态更新
            final_status = "completed" if failed_count == 0 else "completed"  # 即使有失败也标记为完成
            if processed_count == 0 and failed_count > 0:
                final_status = "failed"

            with self.task_lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.status = final_status
                    task.processed_files = processed_count
                    task.failed_files = failed_count
                    task.total_triples = total_triples
                    task.processing_time = processing_time
                    task.results = results
                    task.errors = errors
                    task.updated_at = datetime.now(timezone.utc).isoformat()

            print(f"[任务{task_id}] 完成处理。成功: {processed_count}, 失败: {failed_count}, 总三元组: {total_triples}")

        except Exception as e:
            print(f"[任务{task_id}] 任务处理失败: {str(e)}")
            with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].errors.append({
                        'file_name': 'SYSTEM',
                        'error': f"任务处理失败: {str(e)}"
                    })
                    self.tasks[task_id].updated_at = datetime.now(timezone.utc).isoformat()

        finally:
            # [新增] 清理资源
            try:
                # 关闭数据库连接池
                loop.run_until_complete(db_client.close_current_pool())
            except Exception as e:
                print(f"关闭数据库连接池失败: {e}")

            # 关闭循环
            loop.close()

            # 清理 futures
            if task_id in self.task_futures:
                del self.task_futures[task_id]
            self.cancelled_tasks.discard(task_id)

        # 增加一个包装器来在同步线程中运行异步 DB 操作
    def _process_single_file_sync_wrapper(self, loop, file_path, material_id, prompt_text,
                                        base_url, api_key, model, task_output_dir,
                                        project_id, task_id):
        return loop.run_until_complete(
            self._process_single_file_async(file_path, material_id, prompt_text,
                                            base_url, api_key, model, task_output_dir,
                                            project_id, task_id)
        )

    async def _process_single_file_async(self, file_path: str, material_id: int, prompt_text: str,
                                       base_url: str, api_key: str, model: str,
                                       task_output_dir: Path, project_id: int, task_id: str,
                                       original_file_name: str = None) -> Dict[str, Any]: # <--- 新增参数
        """
        处理单个文件：批量Chunk向量化 -> 并发抽取 -> 即时向量化与入库
        """
        try:
            file_path_obj = Path(file_path)
            # 【修改】优先使用传入的原始文件名
            file_name = original_file_name if original_file_name else file_path_obj.name

            # 1. 读取与切分
            text = read_txt_text(file_path_obj)
            file_hash = db_client.generate_file_hash(text)
            chunks = chunk_text(text)

            # 2. 批量计算所有 Chunk 的向量 (符合"一个文本下的所有分块一起批量计算")
            loop = asyncio.get_running_loop()
            print(f"[{file_name}] 正在计算 {len(chunks)} 个 Chunk 的向量...")
            try:
                chunk_embeddings = await loop.run_in_executor(
                    None, lambda: embedding_service.get_embeddings(chunks)
                )
            except Exception as e:
                print(f"Chunk 向量化失败: {e}")
                chunk_embeddings = [None] * len(chunks)

            # === 并发控制 ===
            CONCURRENCY_LIMIT = 3  # 限制 LLM 并发数
            extract_sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
            ingest_lock = asyncio.Lock()  # 数据库写入锁

            # 用于收集统计信息 (线程安全不是问题，因为主要在 gather 后统计，或者简单的 append)
            raw_outputs = [None] * len(chunks)
            parsed_triples_all = []
            chunk_indices = []

            # 统计计数器
            total_extracted_count = 0

            # === 定义单块处理管线 ===
            async def process_one_chunk_pipeline(idx, chunk, chunk_embedding):
                nonlocal total_extracted_count

                chunk_unique_id = db_client.generate_chunk_id(file_hash, idx, project_id)

                # Phase 1: 并发抽取 (受信号量限制)
                async with extract_sem:
                    print(f"--- [Chunk {idx + 1}/{len(chunks)}] 开始处理...")

                    # 1.1 先入库 Chunk (带向量) - IO密集型，无需锁
                    await db_client.upsert_limited_chunk(
                        project_id, material_id, chunk_unique_id, idx, chunk,
                        file_name=file_name, embedding=chunk_embedding
                    )

                    # 1.2 调用 LLM 抽取
                    content = call_llm(base_url, api_key, model, prompt_text, chunk)
                    raw_outputs[idx] = content  # 保存原始内容

                    triples = parse_triples_from_text(content)

                # Chunk 内去重 (内存操作)
                unique_triples_in_chunk = {}
                for t in triples:
                    key = (
                        t.get('head', {}).get('label'), t.get('head', {}).get('type'),
                        t.get('relationship', {}).get('label'), t.get('relationship', {}).get('type'),
                        t.get('tail', {}).get('label'), t.get('tail', {}).get('type')
                    )
                    if all(k is not None for k in key):
                        unique_triples_in_chunk[key] = t
                valid_triples = list(unique_triples_in_chunk.values())

                if not valid_triples:
                    print(f"  [Chunk {idx + 1}] 未抽取到三元组。")
                    return

                # Phase 2: 即时向量化 (针对该 Chunk 的结果)
                # 准备向量化文本
                node_labels = set()
                edge_texts = []
                # 建立映射以便后续查找
                triple_to_edge_text = []  # 保持顺序对应 valid_triples

                for t in valid_triples:
                    h_label = t['head']['label']
                    t_label = t['tail']['label']
                    r_label = t['relationship']['label']

                    node_labels.add(h_label)
                    node_labels.add(t_label)

                    # 边向量文本：头+关系+尾
                    e_text = f"{h_label} {r_label} {t_label}"
                    edge_texts.append(e_text)
                    triple_to_edge_text.append(e_text)

                node_labels_list = list(node_labels)
                all_texts = node_labels_list + edge_texts

                # 计算向量 (同步调用包装在 executor 中)
                try:
                    if all_texts:
                        all_vecs = await loop.run_in_executor(
                            None, lambda: embedding_service.get_embeddings(all_texts)
                        )
                    else:
                        all_vecs = []
                except Exception as e:
                    print(f"  [Chunk {idx + 1}] 实体关系向量化失败: {e}")
                    all_vecs = [None] * len(all_texts)

                # 映射向量
                node_vec_map = dict(zip(node_labels_list, all_vecs[:len(node_labels_list)]))
                # edge_texts可能有重复(不同三元组生成相同文本?), 这里按顺序取
                edge_vecs = all_vecs[len(node_labels_list):]

                # Phase 3: 串行入库 (持有锁)
                async with ingest_lock:
                    print(f"  >>> [Chunk {idx + 1}] 正在入库 ({len(valid_triples)} triples)...")

                    nodes_to_inc_degree = []

                    for i, t in enumerate(valid_triples):
                        h_label = t['head']['label']
                        t_label = t['tail']['label']
                        r_label = t['relationship']['label']

                        h_type = t['head']['type']
                        t_type = t['tail']['type']
                        r_type = t['relationship']['type']

                        # 插入节点 (带向量)
                        src_id = await db_client.upsert_limited_node(
                            project_id, h_label, h_type, chunk_unique_id,
                            embedding=node_vec_map.get(h_label)
                        )
                        await db_client.insert_limited_provenance(project_id, task_id, chunk_unique_id, node_id=src_id)

                        tgt_id = await db_client.upsert_limited_node(
                            project_id, t_label, t_type, chunk_unique_id,
                            embedding=node_vec_map.get(t_label)
                        )
                        await db_client.insert_limited_provenance(project_id, task_id, chunk_unique_id, node_id=tgt_id)

                        # 插入边 (带向量)
                        # 使用之前准备好的 edge_vecs (按顺序)
                        edge_internal_id, is_new = await db_client.upsert_limited_edge(
                            project_id, src_id, tgt_id, r_label, r_type, chunk_unique_id,
                            embedding=edge_vecs[i] if i < len(edge_vecs) else None
                        )

                        if is_new:
                            nodes_to_inc_degree.extend([src_id, tgt_id])

                        await db_client.insert_limited_provenance(project_id, task_id, chunk_unique_id,
                                                                  edge_id=edge_internal_id)
                        total_extracted_count += 1

                    # 批量更新度
                    if nodes_to_inc_degree:
                        await db_client.increment_limited_nodes_degree(list(set(nodes_to_inc_degree)))

                    # 收集结果用于最终文件保存
                    parsed_triples_all.extend(valid_triples)
                    chunk_indices.extend([idx] * len(valid_triples))

                    print(f"  ✓ [Chunk {idx + 1}] 入库完毕")

            # === 执行所有任务 ===
            tasks = [
                process_one_chunk_pipeline(i, chunk, chunk_embeddings[i])
                for i, chunk in enumerate(chunks)
            ]
            await asyncio.gather(*tasks)

            # 保存调试文件 (可选)
            # 注意：raw_outputs 可能包含 None (如果某个任务失败)，过滤一下
            clean_raw_outputs = [r for r in raw_outputs if r is not None]
            save_outputs(clean_raw_outputs, parsed_triples_all, chunks, chunk_indices,
                         task_output_dir / "txt" / f"{file_path_obj.stem}_extractions.txt",
                         task_output_dir / "jsonl" / f"{file_path_obj.stem}_extractions.jsonl",
                         task_output_dir / "txt" / f"{file_path_obj.stem}_chunks.txt")

            return {
                'status': 'success',
                'triples_count': total_extracted_count,
                'output_files': {
                    'txt': str(task_output_dir / "txt" / f"{file_path_obj.stem}_extractions.txt"),
                    'jsonl': str(task_output_dir / "jsonl" / f"{file_path_obj.stem}_extractions.jsonl")
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}

    def _update_task_progress(self, task_id: str, processed: int, failed: int,
                            total_triples: int, results: List[TaskFile],
                            errors: List[Dict[str, str]]):
        """更新任务进度"""
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.processed_files = processed
                task.failed_files = failed
                task.total_triples = total_triples
                task.results = results[:]  # 创建副本
                task.errors = errors[:]    # 创建副本
                task.updated_at = datetime.now(timezone.utc).isoformat()

    def _update_task_status(self, task_id: str, status: str):
        """更新任务状态"""
        with self.task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                self.tasks[task_id].updated_at = datetime.now(timezone.utc).isoformat()

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if task:
                return task.to_dict()
            return None

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.task_lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status in ["completed", "failed", "cancelled"]:
                return False

            # 标记为取消
            self.cancelled_tasks.add(task_id)
            task.status = "cancelled"
            task.updated_at = datetime.now(timezone.utc).isoformat()

            # 尝试取消future
            if task_id in self.task_futures:
                future = self.task_futures[task_id]
                future.cancel()

            return True

    def get_task_list(self, status: Optional[str] = None, limit: int = 10,
                     offset: int = 0) -> Dict[str, Any]:
        """获取任务列表"""
        with self.task_lock:
            tasks = list(self.tasks.values())

        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)

        # 状态过滤
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 分页
        total = len(tasks)
        tasks = tasks[offset:offset + limit]

        return {
            'tasks': [task.to_dict() for task in tasks],
            'total': total,
            'limit': limit,
            'offset': offset
        }

    def cleanup_old_tasks(self, days: int = 7):
        """清理旧任务"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()

        with self.task_lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.created_at < cutoff_str and task.status in ["completed", "failed", "cancelled"]:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]
                # 清理输出文件
                task_output_dir = self.output_base_dir / task_id
                if task_output_dir.exists():
                    import shutil
                    shutil.rmtree(task_output_dir, ignore_errors=True)

        return len(to_remove)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        with self.task_lock:
            total_tasks = len(self.tasks)
            processing_tasks = sum(1 for t in self.tasks.values() if t.status == "processing")
            completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
            failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")

        return {
            'service_status': 'running',
            'total_tasks': total_tasks,
            'processing_tasks': processing_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'max_workers': self.max_workers,
            'active_workers': len(self.task_futures)
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"

    def shutdown(self):
        """关闭服务"""
        print("正在关闭三元组抽取服务...")

        # 取消所有正在进行的任务
        with self.task_lock:
            for task_id in list(self.tasks.keys()):
                if self.tasks[task_id].status == "processing":
                    self.cancel_task(task_id)

        # 关闭线程池
        self.executor.shutdown(wait=True)
        print("三元组抽取服务已关闭")


# 全局服务实例
service_instance = None

def get_service() -> TripletsExtractionService:
    """获取服务实例（单例模式）"""
    global service_instance
    if service_instance is None:
        service_instance = TripletsExtractionService()
    return service_instance


if __name__ == "__main__":
    # 测试代码
    service = TripletsExtractionService()

    # 导入读取函数
    from pathlib import Path

    # 定义 prompt 文件路径
    prompt_file = Path("triple_extraction_prompt.txt")

    # 确保文件存在
    if not prompt_file.exists():
        print(f"错误：找不到 {prompt_file}，请确保文件在当前目录下")
        exit(1)

    # === 修复：读取完整的 Prompt 内容 ===
    prompt_text = prompt_file.read_text(encoding="utf-8")

    try:
        print("正在创建任务...")
        task_id = service.create_task(
            # 请替换为你本地实际存在的测试文件路径
            files=[r"D:\Personal_Project\kgplatform_backend\python-service\txt_test\【回顾】南师附小弹性离校 _ 放学别走！一起 “识天文知地理”！.txt"],
            prompt_text=prompt_text,  # 传入包含 JSON 约束的完整 Prompt
            provider="deepseek",
            api_key="sk-1bc317ee3858458d9648944a2184e4df"  # 确保 key 正确
        )

        print(f"任务创建成功，ID: {task_id}")

        # 监控任务状态
        import time

        while True:
            status = service.get_task_status(task_id)
            print(f"任务状态: {status['status']} | 已处理: {status['processed_files']}/{status['total_files']}")

            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)

        print("最终结果:", status)

        # 打印部分结果以验证
        if status['results']:
            print("\n--- 抽取结果预览 ---")
            res_file = status['results'][0]['output_files']['txt']
            if os.path.exists(res_file):
                with open(res_file, 'r', encoding='utf-8') as f:
                    print(f.read()[:500] + "...")

    except Exception as e:
        print(f"测试失败: {e}")

    finally:
        service.shutdown()