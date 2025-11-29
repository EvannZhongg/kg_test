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

from extract_triplets_from_docx import (
    read_txt_text, chunk_text, make_client, call_llm,
    parse_triples_from_text, PROVIDERS, save_outputs
)


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
                   model: Optional[str] = None,
                   api_key: Optional[str] = None,
                   base_url: Optional[str] = None) -> str:
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
            else:
                file_path = file_item
                material_id = None

            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            if not path.suffix.lower() == '.txt':
                raise ValueError(f"不支持的文件格式: {file_path}")
            valid_files.append({
                'path': str(path),
                'material_id': material_id
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
            task_id, valid_files, prompt_text, provider, model, api_key, base_url
        )
        self.task_futures[task_id] = future

        return task_id

    def _process_task(self, task_id: str, files: List[Dict[str, Any]], prompt_text: str,
                     provider: str, model: Optional[str], api_key: Optional[str],
                     base_url: Optional[str]):
        """处理任务的内部方法

        Args:
            files: 包含path和material_id的字典列表
        """
        start_time = time.time()

        try:
            # 检查任务是否被取消
            if task_id in self.cancelled_tasks:
                self._update_task_status(task_id, "cancelled")
                return

            # 初始化AI服务配置
            resolved_base_url, resolved_api_key, resolved_model = make_client(
                provider, model, base_url, api_key
            )

            # 创建任务专用输出目录
            task_output_dir = self.output_base_dir / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)

            results = []
            errors = []
            total_triples = 0
            processed_count = 0
            failed_count = 0

            for i, file_info in enumerate(files, 1):
                # 检查任务是否被取消
                if task_id in self.cancelled_tasks:
                    self._update_task_status(task_id, "cancelled")
                    return

                file_path = file_info['path']
                material_id = file_info.get('material_id')
                file_name = Path(file_path).name
                print(f"[任务{task_id}] 处理文件 {i}/{len(files)}: {file_name} (material_id: {material_id})")

                try:
                    # 处理单个文件
                    file_result = self._process_single_file(
                        file_path, prompt_text, resolved_base_url,
                        resolved_api_key, resolved_model, task_output_dir
                    )

                    if file_result['status'] == 'success':
                        results.append(TaskFile(
                            file_name=file_name,
                            material_id=material_id,
                            status='success',
                            triples_count=file_result['triples_count'],
                            output_files=file_result['output_files']
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
            # 清理
            if task_id in self.task_futures:
                del self.task_futures[task_id]
            self.cancelled_tasks.discard(task_id)

    def _process_single_file(self, file_path: str, prompt_text: str,
                           base_url: str, api_key: str, model: str,
                           task_output_dir: Path) -> Dict[str, Any]:
        """处理单个文件"""
        try:
            file_path_obj = Path(file_path)
            stem = file_path_obj.stem

            # 创建输出文件路径
            txt_dir = task_output_dir / "txt"
            jsonl_dir = task_output_dir / "jsonl"
            txt_dir.mkdir(parents=True, exist_ok=True)
            jsonl_dir.mkdir(parents=True, exist_ok=True)

            out_txt = txt_dir / f"{stem}_extractions.txt"
            out_jsonl = jsonl_dir / f"{stem}_extractions.jsonl"
            chunks_txt = txt_dir / f"{stem}_chunks.txt"

            # 读取和处理文档
            text = read_txt_text(file_path_obj)
            chunks = chunk_text(text)
            raw_outputs = []
            parsed_triples = []
            chunk_indices = []  # 记录每个三元组来自哪个chunk

            for idx, chunk in enumerate(chunks, start=1):
                content = call_llm(base_url, api_key, model, prompt_text, chunk)
                raw_outputs.append(content)
                triples_from_chunk = parse_triples_from_text(content)
                parsed_triples.extend(triples_from_chunk)
                # 为每个三元组记录它来自哪个chunk（索引从0开始）
                chunk_indices.extend([idx - 1] * len(triples_from_chunk))

            # 保存输出（包含chunk溯源信息）
            save_outputs(raw_outputs, parsed_triples, chunks, chunk_indices, out_txt, out_jsonl, chunks_txt)

            return {
                'status': 'success',
                'triples_count': len(parsed_triples),
                'output_files': {
                    'txt': str(out_txt.absolute()),
                    'jsonl': str(out_jsonl.absolute()),
                    'chunks': str(chunks_txt.absolute())
                }
            }

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

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