#!/usr/bin/env python3
import json
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time
import atexit
from datetime import datetime
import requests
import uuid
from urllib.parse import urlparse

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import logging
import asyncio
from database import db_client
from triplet_extraction_service import TripletsExtractionService, get_service
from open_ie_service import get_open_ie_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class APIServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        self.host = host
        self.port = port
        self.debug = debug

        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)  # 启用CORS支持

        # 获取服务实例
        self.service = get_service()

        # 获取 OpenIE 服务实例
        self.open_ie_service = get_open_ie_service()

        # 注册路由
        self._register_routes()

        # 注册错误处理器
        self._register_error_handlers()

        # SSE连接管理
        self.sse_connections: Dict[str, list] = {}  # task_id -> [connection_queue, ...]
        self.sse_lock = threading.Lock()

        # 启动SSE推送线程
        self.sse_thread = threading.Thread(target=self._sse_monitor_thread, daemon=True)
        self.sse_thread.start()

        # 注册清理函数
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"API服务器初始化完成，监听 {host}:{port}")

    def _is_valid_url(self, url: str) -> bool:
        """验证URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _download_file_from_url(self, url: str, download_dir: Path, allowed_extensions: list = None) -> str:
        """从URL下载文件到指定目录"""
        try:
            # 验证URL
            if not self._is_valid_url(url):
                raise ValueError(f"无效的URL: {url}")

            # 设置请求头，模拟浏览器请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }

            # 检查文件扩展名
            parsed_url = urlparse(url)
            file_ext = Path(parsed_url.path).suffix.lower()
            
            if allowed_extensions and file_ext not in allowed_extensions:
                raise ValueError(f"URL指向的文件格式不支持: {url}，支持的格式: {allowed_extensions}")

            # 发送HEAD请求获取文件信息（可选，如果失败则跳过）
            try:
                head_response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
                head_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.warning(f"HEAD请求失败，将直接尝试下载: {str(e)}")

            # 生成唯一文件名
            original_filename = os.path.basename(parsed_url.path)
            if not original_filename:
                original_filename = f"downloaded_{uuid.uuid4().hex[:8]}{file_ext}"
            
            # URL解码文件名
            import urllib.parse
            decoded_filename = urllib.parse.unquote(original_filename)
            logger.debug(f"原始文件名: {original_filename}")
            logger.debug(f"解码后文件名: {decoded_filename}")
            
            # 简化文件名处理：直接使用UUID生成安全文件名
            # 这样可以避免中文路径和特殊字符的问题
            safe_filename = f"sample_{uuid.uuid4().hex[:8]}{file_ext}"
            
            logger.debug(f"最终安全文件名: {safe_filename}")
            file_path = download_dir / safe_filename
            
            # 下载文件
            logger.info(f"开始从URL下载文件: {url}")
            response = requests.get(url, headers=headers, timeout=300, stream=True)  # 5分钟超时
            response.raise_for_status()
            
            # 检查文件大小（限制为100MB）
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 100 * 1024 * 1024:
                raise ValueError("文件大小超过100MB限制")
            
            # 写入文件
            with open(file_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # 动态检查文件大小
                        if downloaded_size > 100 * 1024 * 1024:
                            f.close()
                            file_path.unlink()  # 删除部分下载的文件
                            raise ValueError("文件大小超过100MB限制")
            
            logger.info(f"文件下载完成: {file_path}")
            return str(file_path)
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"下载文件失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"处理URL文件时出错: {str(e)}")

    def _generate_prompt_from_urls(
        self, 
        schema_url: str, 
        sample_text_url: str = None, 
        sample_xlsx_url: str = None,
        target_domain: str = None,
        dictionary_url: str = None,
        priority_extractions: list = None,
        extraction_requirements: str = None,
        base_instruction: str = None
    ) -> str:
        """
        从URL生成Prompt
        
        Args:
            schema_url: 知识图谱schema的URL
            sample_text_url: 样例原文的URL（支持.txt, .docx格式）
            sample_xlsx_url: 样例三元组的URL（.xlsx格式）
            target_domain: 目标领域描述
            dictionary_url: 专业词典的URL（.txt格式）
            priority_extractions: 抽取意向优先级列表，格式：["城市", "建筑师", "设计作品"]
            extraction_requirements: 抽取要求描述
            base_instruction: 基础指导语（如果不提供，使用默认指导）
        
        Returns:
            生成的prompt字符串
        """
        try:
            # 导入gen_triplet_prompt模块
            from gen_triplet_prompt import load_schema, collect_types, build_prompt, read_txt_text, read_xlsx_text
            import tempfile
            
            # 下载schema文件
            logger.info(f"下载schema文件: {schema_url}")
            schema_content = self._download_text_from_url(schema_url)
            
            # 解析schema
            schema = json.loads(schema_content)
            entity_type_to_children, relationship_types, triplet_types = collect_types(schema)
            
            # 构建基础prompt
            prompt_lines = []

            # 1. 添加基础指导语
            if base_instruction:
                prompt_lines.append(base_instruction)
            else:
                prompt_lines.append("你是一个专业的数据分析师，请从输入文本中抽取结构化的知识图谱三元组。")
            prompt_lines.append("")

            # 2. 添加动态思维链 (Priority / Rounds) - 新增逻辑
            prompt_lines.append("=== 抽取思维链（Step-by-Step） ===")
            if priority_extractions and len(priority_extractions) > 0:
                for i, entity_type in enumerate(priority_extractions, 1):
                    prompt_lines.append(
                        f"Round {i}：扫描全文，重点寻找并抽取【{entity_type}】类型的实体，以及它作为头实体发出的所有关系。")
                next_round = len(priority_extractions) + 1
                prompt_lines.append(f"Round {next_round}：扫描剩余文本，补充抽取其他未覆盖的实体和关系。")
                prompt_lines.append(f"Round {next_round + 1}：总结核心实践活动的成果及技能。")
            else:
                prompt_lines.append("Round 1：抽取核心实践活动实体。")
                prompt_lines.append("Round 2：抽取活动的属性（时间、地点、类型等）。")
                prompt_lines.append("Round 3：抽取参与人物、团队及其行为。")
                prompt_lines.append("Round 4：总结成果与技能。")
            prompt_lines.append("")

            # 3. 添加目标领域
            if target_domain:
                prompt_lines.append(f"=== 目标领域: {target_domain} ===")
                prompt_lines.append("")

            # 4. 添加扁平化输出格式 - 修改逻辑
            prompt_lines.append("=== 输出格式要求 ===")
            prompt_lines.append("请严格按照以下JSON格式输出：")
            prompt_lines.append("```json")
            prompt_lines.append("{")
            prompt_lines.append('  "activity": "核心实践活动名称",')
            prompt_lines.append('  "triples": [')
            prompt_lines.append(
                '    // 格式：["头实体(标准名)", "头类型", "关系", "关系类型", "尾实体(标准名)", "尾类型"]')
            prompt_lines.append('    ["实体A", "类型A", "关系X", "类型X", "实体B", "类型B"]')
            prompt_lines.append('  ]')
            prompt_lines.append("}")
            prompt_lines.append("```")
            prompt_lines.append("")

            # 5. 添加归一化规则 - 新增逻辑
            prompt_lines.append("=== 关键规则：实体归一化 ===")
            prompt_lines.append(
                "1. **指代消解**：文中出现的简称、别名（如'南师附小'），必须统一归一化为全称（如'南京师范大学附属小学'）。")
            prompt_lines.append("2. **词典匹配**：若实体在下文提供的【专业词典】中，必须使用词典中的标准名称。")
            prompt_lines.append("")
            
            # 4. 添加输出示例
            prompt_lines.append("=== 输出示例 ===")
            prompt_lines.append("```json")
            prompt_lines.append("{")
            prompt_lines.append('  "activity": "南师附小弹性离校活动",')
            prompt_lines.append('  "triples": [')
            prompt_lines.append('    ["南师附小弹性离校活动", "实践活动", "实践类型", "实践类型", "支教", "实践类型"],')
            prompt_lines.append('    ["志愿者团队", "团队", "实践", "实践", "南师附小弹性离校活动", "实践活动"]')
            prompt_lines.append('  ]')
            prompt_lines.append("}")
            prompt_lines.append("```")
            prompt_lines.append("")


            
            # 5. 添加抽取规则
            prompt_lines.append("=== 抽取规则 ===")
            prompt_lines.append("- 必填字段：`label` 为原文中的表面词；`type` 必须从下方枚举中选择。")
            prompt_lines.append("- 如 tail 的表面词来自某类型的`细分`（includes），则 `type` 写该父类型名称，`label` 写细分项文本（如果不在其中，也可以自行总结细分项）。")
            prompt_lines.append("- 关系的 `type` 也必须来自下方关系类型枚举，`label` 写原文触发词（可与 type 相同或为同义表达）。")
            prompt_lines.append("- 三元组的类型必须来自下方三元组类型枚举，不要出现其他的类型，并且不要头尾颠倒")
            prompt_lines.append("- 当遇见并列的'、'、'，'等表达，将并列的实体分别抽取")
            prompt_lines.append("- 严格按照上述JSON格式输出，不要添加任何其他内容")
            prompt_lines.append("- 确保JSON格式完全正确，可以被直接解析")
            prompt_lines.append("")
            
            # 6. 添加抽取意向优先级（如果提供）
            if priority_extractions and len(priority_extractions) > 0:
                prompt_lines.append("=== 抽取意向优先级 ===")
                prompt_lines.append("以下实体类型需要特别关注：")
                prompt_lines.append(", ".join(priority_extractions))
                prompt_lines.append("")
            
            # 7. 添加自定义抽取要求（如果提供）
            if extraction_requirements:
                prompt_lines.append("=== 抽取要求描述 ===")
                prompt_lines.append(extraction_requirements)
                prompt_lines.append("")

            # 8. 添加实体类型
            prompt_lines.append("=== 实体类型与细分（type -> includes） ===")
            for etype in sorted(entity_type_to_children.keys()):
                children = sorted([c for c in entity_type_to_children[etype] if c])
                if children:
                    prompt_lines.append(f"- {etype}：{', '.join(children)}")
                else:
                    prompt_lines.append(f"- {etype}：<无细分>")
            prompt_lines.append("")

            # 9. 添加关系类型
            prompt_lines.append("=== 关系类型与细分（type -> includes） ===")
            if relationship_types:
                for rtype in sorted(relationship_types):
                    children = sorted([c for c in relationship_types[rtype] if c])
                    if children:
                        prompt_lines.append(f"- {rtype}：{', '.join(children)}")
                    else:
                        prompt_lines.append(f"- {rtype}：<无细分>")
            else:
                prompt_lines.append("- <未在schema中定义>")
            prompt_lines.append("")
            
            # 10. 添加三元组类型
            prompt_lines.append("=== 三元组类型（type） ===")
            if triplet_types:
                # 按头实体分组
                from collections import defaultdict
                head_entity_groups = defaultdict(list)
                for triplet in triplet_types:
                    parts = triplet.split('->')
                    if len(parts) == 3:
                        head_entity = parts[0]
                        head_entity_groups[head_entity].append(triplet)
                
                # 按头实体分组显示
                for head_entity in sorted(head_entity_groups.keys()):
                    prompt_lines.append(f"\n{head_entity}相关：")
                    for triplet in sorted(head_entity_groups[head_entity]):
                        prompt_lines.append(f"  {triplet}")
            else:
                prompt_lines.append("- <未在schema中定义>")
            prompt_lines.append("")
            
            # 11. 添加专业词典（如果提供）
            if dictionary_url:
                prompt_lines.append("=== 专业词典 ===")
                try:
                    # 直接下载txt文件内容
                    dict_content = self._download_text_from_url(dictionary_url)
                    
                    prompt_lines.append("以下是专业词典，抽取时请参考这些术语：")
                    # 按行分割词典内容
                    for line in dict_content.split('\n'):
                        line = line.strip()
                        if line:  # 跳过空行
                            prompt_lines.append(f"  {line}")
                except Exception as e:
                    logger.warning(f"下载专业词典失败: {str(e)}")
                    prompt_lines.append("<专业词典下载失败>")
                prompt_lines.append("")
            
            # 12. 添加样例（如果提供了URL）
            if sample_text_url or sample_xlsx_url:
                prompt_lines.append("=== 抽取样例 ===")
                prompt_lines.append("原文：")
                
                if sample_text_url:
                    try:
                        # 检查文件格式并相应处理
                        parsed_url = urlparse(sample_text_url)
                        file_ext = Path(parsed_url.path).suffix.lower()
                        
                        if file_ext == '.txt':
                            # 直接下载文本内容
                            sample_text = self._download_text_from_url(sample_text_url)
                            prompt_lines.append(sample_text)
                        elif file_ext == '.docx':
                            # 下载docx文件并提取文本
                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_path = Path(temp_dir)
                                docx_file = self._download_file_from_url(sample_text_url, temp_path, ['.docx'])
                                
                                # 读取docx内容
                                from docx import Document
                                doc = Document(docx_file)
                                sample_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                                prompt_lines.append(sample_text)
                        else:
                            prompt_lines.append("<不支持的样例文本格式>")
                    except Exception as e:
                        logger.warning(f"下载样例文本失败: {str(e)}")
                        prompt_lines.append("<样例文本下载失败>")
                else:
                    prompt_lines.append("<未提供样例文本>")
                
                prompt_lines.append("")
                prompt_lines.append("抽取结果：")
                
                if sample_xlsx_url:
                    try:
                        # 下载Excel文件到临时目录
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            excel_file = self._download_file_from_url(sample_xlsx_url, temp_path, ['.xlsx'])
                            
                            # 读取Excel内容
                            sample_xlsx_text = read_xlsx_text(Path(excel_file))
                            prompt_lines.append(sample_xlsx_text)
                    except Exception as e:
                        logger.warning(f"下载样例Excel失败: {str(e)}")
                        prompt_lines.append("<样例Excel下载失败>")
                else:
                    prompt_lines.append("<未提供样例Excel>")
                
                prompt_lines.append("")
            
            # 13. 再次强调
            prompt_lines.append("=== 再次强调 ===")
            prompt_lines.append("请严格按照JSON格式输出，不要添加任何其他文字、说明或格式。")
            prompt_lines.append("输出必须是有效的JSON，可以被直接解析。")
            prompt_lines.append("")

            return "\n".join(prompt_lines)
            
        except Exception as e:
            raise ValueError(f"生成Prompt失败: {str(e)}")

    def _download_text_from_url(self, url: str) -> str:
        """从URL下载文本内容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/plain,text/*,*/*,application/json',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }

            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
            for encoding in encodings:
                try:
                    return response.content.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用utf-8并忽略错误
            return response.content.decode('utf-8', errors='ignore').strip()
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"下载文本失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"处理文本URL时出错: {str(e)}")

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': self.service.get_service_status()
            })

        @self.app.route('/api/v1/tasks', methods=['POST'])
        def create_task():
            """创建新任务（支持上传目录文件名、绝对路径和URL，仅支持.txt格式）

            请求格式支持两种：
            1. 旧格式：files为字符串数组 ["file1.txt", "file2.txt"]
            2. 新格式：files为对象数组 [{"material_id": 1, "url": "file1.txt"}, ...]
            """
            try:
                data = request.get_json()
                if not data:
                    raise BadRequest("请求体不能为空")

                # 定义固定的上传目录
                UPLOAD_DIR = Path(
                    os.getenv("UPLOAD_DIR", r"E:\GoLandProject\exam\item_reslove\gin-task-server\uploads"))
                # 确保上传目录存在
                if not UPLOAD_DIR.exists():
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    logger.warning(f"上传目录不存在，已自动创建: {UPLOAD_DIR}")

                # 获取任务类型和项目ID
                task_type = data.get('task_type', 'schema')  # 默认为 schema 模式
                project_id = data.get('project_id', 0)

                # 参数验证
                # 基础必需参数
                required_fields = ['files', 'provider', 'api_key']

                # 仅在 Schema 模式下 prompt_text 是必需的
                if task_type == 'schema':
                    required_fields.append('prompt_text')

                for field in required_fields:
                    if field not in data:
                        raise BadRequest(f"缺少必需参数: {field}")

                files = data['files']

                # 判断是新格式（对象数组）还是旧格式（字符串数组）
                is_new_format = all(isinstance(f, dict) for f in files) if files else False
                is_old_format = all(isinstance(f, str) for f in files) if files else False

                if not (is_new_format or is_old_format):
                    raise BadRequest("files 格式错误：必须全为字符串或全为对象（包含material_id和url字段）")

                # 提取URL列表用于分类判断
                if is_new_format:
                    url_list = [f.get('url') for f in files]
                else:
                    url_list = files

                # 分类校验：支持三种格式（绝对路径、上传目录中的文件名、URL）
                is_path_list = all(Path(url).is_absolute() for url in url_list)
                is_filename_list = all(not Path(url).is_absolute() and not self._is_valid_url(url) for url in url_list)
                is_url_list = all(self._is_valid_url(url) for url in url_list)

                if not (is_path_list or is_filename_list or is_url_list):
                    raise BadRequest("files 格式错误：需全为绝对路径、全为上传目录中的文件名或全为URL（不可混合）")

                # 处理文件列表，保留material_id信息
                resolved_files = []

                # 处理文件名列表：拼接上传目录路径
                if is_filename_list:
                    invalid_files = []
                    for file_item in files:
                        # 提取URL和material_id
                        if is_new_format:
                            filename = file_item.get('url')
                            material_id = file_item.get('material_id')
                        else:
                            filename = file_item
                            material_id = None

                        # 拼接完整路径
                        file_path = UPLOAD_DIR / filename
                        # 安全检查：确保文件确实在上传目录内
                        if not str(file_path).startswith(str(UPLOAD_DIR)):
                            invalid_files.append(f"{filename}（文件名不合法，可能存在路径穿越风险）")
                            continue
                        # 校验1：文件是否存在
                        if not file_path.exists():
                            invalid_files.append(f"{filename}（文件不存在于上传目录）")
                            continue
                        # 校验2：是否为文件（非文件夹）
                        if not file_path.is_file():
                            invalid_files.append(f"{filename}（是文件夹，非文件）")
                            continue
                        # 校验3：是否为 .txt 格式
                        if file_path.suffix.lower() != '.txt':
                            invalid_files.append(f"{filename}（仅支持 .txt 格式）")
                            continue
                        resolved_files.append({
                            'material_id': material_id,
                            'url': str(file_path)
                        })

                    if invalid_files:
                        raise BadRequest(f"无效文件列表：{'; '.join(invalid_files)}")

                # 处理绝对路径列表
                elif is_path_list:
                    invalid_files = []
                    for file_item in files:
                        # 提取URL和material_id
                        if is_new_format:
                            file_path_str = file_item.get('url')
                            material_id = file_item.get('material_id')
                        else:
                            file_path_str = file_item
                            material_id = None

                        path_obj = Path(file_path_str)
                        if not path_obj.exists():
                            invalid_files.append(f"{file_path_str}（文件不存在）")
                            continue
                        if not path_obj.is_file():
                            invalid_files.append(f"{file_path_str}（是文件夹，非文件）")
                            continue
                        if path_obj.suffix.lower() != '.txt':
                            invalid_files.append(f"{file_path_str}（仅支持 .txt 格式）")
                            continue
                        resolved_files.append({
                            'material_id': material_id,
                            'url': file_path_str
                        })

                    if invalid_files:
                        raise BadRequest(f"无效文件列表：{'; '.join(invalid_files)}")

                # 处理URL列表：下载文件到上传目录
                elif is_url_list:
                    invalid_files = []
                    for file_item in files:
                        # 提取URL和material_id
                        if is_new_format:
                            url = file_item.get('url')
                            material_id = file_item.get('material_id')
                        else:
                            url = file_item
                            material_id = None

                        try:
                            downloaded_file = self._download_file_from_url(url, UPLOAD_DIR)
                            resolved_files.append({
                                'material_id': material_id,
                                'url': downloaded_file
                            })
                        except ValueError as e:
                            invalid_files.append(f"{url}（{str(e)}）")

                    if invalid_files:
                        raise BadRequest(f"无效URL列表：{'; '.join(invalid_files)}")

                # 根据任务类型分发到不同的服务
                if task_type == 'open_ie':
                    # OpenIE 模式：调用 open_ie_service
                    if project_id <= 0:
                        raise BadRequest("OpenIE 模式必须提供有效的 project_id")

                    logger.info(f"接收到 OpenIE 任务，项目ID: {project_id}")
                    task_id = self.open_ie_service.create_task(
                        files=resolved_files,
                        provider=data['provider'],
                        project_id=project_id,
                        model=data.get('model'),
                        api_key=data['api_key'],
                        base_url=data.get('base_url')
                    )
                else:
                    # Schema 模式 (默认)：调用原有的 service
                    task_id = self.service.create_task(
                        files=resolved_files,
                        prompt_text=data.get('prompt_text', ''),
                        provider=data['provider'],
                        model=data.get('model'),
                        api_key=data['api_key'],
                        base_url=data.get('base_url')
                    )

                logger.info(f"创建任务成功: {task_id} (类型: {task_type})")
                return jsonify({
                    'task_id': task_id,
                    'status': 'created',
                    'message': f'{task_type} 任务创建成功'
                }), 201

            except ValueError as e:
                logger.warning(f"参数验证失败: {str(e)}")
                raise BadRequest(str(e))
            except FileNotFoundError as e:
                logger.warning(f"文件不存在: {str(e)}")
                raise BadRequest(str(e))
            except Exception as e:
                logger.error(f"创建任务失败: {str(e)}")
                raise InternalServerError(f"创建任务失败: {str(e)}")

        @self.app.route('/api/v1/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id: str):
            """获取任务状态 (自动在两个服务中查找)"""
            try:
                # 1. 优先尝试从 Schema 服务获取
                status = self.service.get_task_status(task_id)
                if status:
                    # 标记类型方便前端区分
                    status['task_type'] = 'schema'
                    return jsonify(status)

                # 2. 尝试从 OpenIE 服务获取
                status = self.open_ie_service.get_task_status(task_id)
                if status:
                    status['task_type'] = 'open_ie'
                    return jsonify(status)

                raise NotFound(f"任务不存在: {task_id}")

            except NotFound:
                raise
            except Exception as e:
                logger.error(f"获取任务状态失败: {str(e)}")
                raise InternalServerError(f"获取任务状态失败: {str(e)}")

        @self.app.route('/api/v1/tasks/<task_id>', methods=['DELETE'])
        def cancel_task(task_id: str):
            """取消任务"""
            try:
                # 1. 尝试在 Schema 服务取消
                if self.service.cancel_task(task_id):
                    logger.info(f"Schema 任务取消成功: {task_id}")
                    return jsonify({
                        'task_id': task_id,
                        'status': 'cancelled',
                        'message': '任务取消成功'
                    })

                # 2. 尝试在 OpenIE 服务取消
                if self.open_ie_service.cancel_task(task_id):
                    logger.info(f"OpenIE 任务取消成功: {task_id}")
                    return jsonify({
                        'task_id': task_id,
                        'status': 'cancelled',
                        'message': '任务取消成功'
                    })

                raise BadRequest(f"无法取消任务 {task_id}，任务可能不存在或已完成")

            except BadRequest:
                raise
            except Exception as e:
                logger.error(f"取消任务失败: {str(e)}")
                raise InternalServerError(f"取消任务失败: {str(e)}")

        @self.app.route('/api/v1/tasks', methods=['GET'])
        def get_task_list():
            """获取任务列表 (合并展示 Schema 和 OpenIE 任务)"""
            try:
                status_filter = request.args.get('status')
                limit = min(int(request.args.get('limit', 10)), 100)
                offset = int(request.args.get('offset', 0))

                # 1. 获取 Schema 服务的所有任务 (limit设大一点以获取全量用于排序)
                # 注意：如果生产环境任务量极大，这里建议后续重构为数据库查询
                schema_res = self.service.get_task_list(status_filter, limit=10000, offset=0)
                schema_tasks = schema_res.get('tasks', [])
                for t in schema_tasks:
                    t['task_type'] = 'schema'

                # 2. 获取 OpenIE 服务的所有任务
                open_res = self.open_ie_service.get_task_list(status_filter, limit=10000, offset=0)
                open_tasks = open_res.get('tasks', [])
                for t in open_tasks:
                    t['task_type'] = 'open_ie'

                # 3. 合并并按创建时间倒序排序
                all_tasks = schema_tasks + open_tasks
                all_tasks.sort(key=lambda x: x['created_at'], reverse=True)

                # 4. 内存分页
                total = len(all_tasks)
                paged_tasks = all_tasks[offset: offset + limit]

                return jsonify({
                    'tasks': paged_tasks,
                    'total': total,
                    'limit': limit,
                    'offset': offset
                })

            except ValueError as e:
                raise BadRequest(f"参数格式错误: {str(e)}")
            except Exception as e:
                logger.error(f"获取任务列表失败: {str(e)}")
                raise InternalServerError(f"获取任务列表失败: {str(e)}")

        @self.app.route('/api/v1/tasks/<task_id>/stream', methods=['GET'])
        def task_stream(task_id: str):
            """任务进度实时推送(SSE)"""

            def generate():
                # 检查任务是否存在 (先查 Schema，再查 OpenIE)
                status = self.service.get_task_status(task_id)
                if not status:
                    status = self.open_ie_service.get_task_status(task_id)

                if not status:
                    yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                    return

                # 创建连接队列
                from queue import Queue
                connection_queue = Queue()

                # 注册连接
                with self.sse_lock:
                    if task_id not in self.sse_connections:
                        self.sse_connections[task_id] = []
                    self.sse_connections[task_id].append(connection_queue)

                try:
                    # 发送当前状态
                    yield f"data: {json.dumps(status)}\n\n"

                    # 如果任务已完成，直接返回
                    if status['status'] in ['completed', 'failed', 'cancelled']:
                        return

                    # 持续推送更新
                    while True:
                        try:
                            # 等待更新（超时30秒发送心跳）
                            update = connection_queue.get(timeout=30)
                            if update is None:  # 结束信号
                                break

                            yield f"data: {json.dumps(update)}\n\n"

                            # 如果任务完成，退出循环
                            if update.get('status') in ['completed', 'failed', 'cancelled']:
                                break

                        except:
                            # 超时发送心跳
                            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"

                            # 再次检查任务是否存在（防止任务被物理删除后死循环）
                            current_status = self.service.get_task_status(
                                task_id) or self.open_ie_service.get_task_status(task_id)
                            if not current_status:
                                break

                finally:
                    # 清理连接
                    with self.sse_lock:
                        if task_id in self.sse_connections:
                            try:
                                if connection_queue in self.sse_connections[task_id]:
                                    self.sse_connections[task_id].remove(connection_queue)
                                if not self.sse_connections[task_id]:
                                    del self.sse_connections[task_id]
                            except ValueError:
                                pass

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*'
                }
            )

        @self.app.route('/api/v1/service/status', methods=['GET'])
        def service_status():
            """获取服务状态"""
            return jsonify({
                'schema_service': self.service.get_service_status(),
                'open_ie_service': self.open_ie_service.get_service_status(),
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/v1/service/cleanup', methods=['POST'])
        def cleanup_old_tasks():
            """清理旧任务"""
            try:
                data = request.get_json() or {}
                days = data.get('days', 7)

                # 清理两个服务
                count1 = self.service.cleanup_old_tasks(days)
                count2 = self.open_ie_service.cleanup_old_tasks(days)

                return jsonify({
                    'message': f'清理完成',
                    'removed_tasks': count1 + count2,
                    'details': {
                        'schema_service': count1,
                        'open_ie_service': count2
                    }
                })

            except Exception as e:
                logger.error(f"清理任务失败: {str(e)}")
                raise InternalServerError(f"清理任务失败: {str(e)}")

        @self.app.route('/api/v1/genprompt', methods=['POST'])
        def generate_prompt():
            """
            生成三元组抽取Prompt
            
            请求体参数：
            - schema_url (必需): 知识图谱schema的URL
            - sample_text_url (可选): 样例原文的URL（支持.txt, .docx格式）
            - sample_xlsx_url (可选): 样例三元组的URL（.xlsx格式）
            - target_domain (可选): 目标领域描述，如"建筑学"、"医学"等
            - dictionary_url (可选): 专业词典的URL（.txt格式）
            - priority_extractions (可选): 抽取意向优先级列表
              格式：["城市", "建筑师", "设计作品"]
            - extraction_requirements (可选): 抽取要求描述文本
            - base_instruction (可选): 自定义基础指导语
            """
            try:
                data = request.get_json()
                if not data:
                    raise BadRequest("请求体不能为空")

                # 参数验证
                required_fields = ['schema_url']
                for field in required_fields:
                    if field not in data:
                        raise BadRequest(f"缺少必需参数: {field}")

                # 提取所有参数
                schema_url = data['schema_url']
                sample_text_url = data.get('sample_text_url')
                sample_xlsx_url = data.get('sample_xlsx_url')
                target_domain = data.get('target_domain')
                dictionary_url = data.get('dictionary_url')
                priority_extractions = data.get('priority_extractions')
                extraction_requirements = data.get('extraction_requirements')
                base_instruction = data.get('base_instruction')

                # 验证URL格式
                if not self._is_valid_url(schema_url):
                    raise BadRequest(f"无效的schema URL: {schema_url}")
                
                if sample_text_url and not self._is_valid_url(sample_text_url):
                    raise BadRequest(f"无效的样例文本URL: {sample_text_url}")
                
                if sample_xlsx_url and not self._is_valid_url(sample_xlsx_url):
                    raise BadRequest(f"无效的样例Excel URL: {sample_xlsx_url}")
                
                if dictionary_url and not self._is_valid_url(dictionary_url):
                    raise BadRequest(f"无效的词典URL: {dictionary_url}")
                
                # 验证文件格式
                if sample_text_url:
                    parsed_url = urlparse(sample_text_url)
                    file_ext = Path(parsed_url.path).suffix.lower()
                    if file_ext not in ['.txt', '.docx']:
                        raise BadRequest(f"样例文本文件必须是.txt或.docx格式: {sample_text_url}")
                
                if sample_xlsx_url:
                    parsed_url = urlparse(sample_xlsx_url)
                    file_ext = Path(parsed_url.path).suffix.lower()
                    if file_ext != '.xlsx':
                        raise BadRequest(f"样例Excel文件必须是.xlsx格式: {sample_xlsx_url}")
                
                if dictionary_url:
                    parsed_url = urlparse(dictionary_url)
                    file_ext = Path(parsed_url.path).suffix.lower()
                    if file_ext not in ['.txt']:
                        raise BadRequest(f"词典文件必须是.txt格式: {dictionary_url}")
                
                # 验证priority_extractions格式
                if priority_extractions:
                    if not isinstance(priority_extractions, list):
                        raise BadRequest("priority_extractions必须是数组格式")
                    for item in priority_extractions:
                        if not isinstance(item, str):
                            raise BadRequest("priority_extractions中的每一项必须是字符串")

                # 生成Prompt
                prompt_content = self._generate_prompt_from_urls(
                    schema_url=schema_url,
                    sample_text_url=sample_text_url,
                    sample_xlsx_url=sample_xlsx_url,
                    target_domain=target_domain,
                    dictionary_url=dictionary_url,
                    priority_extractions=priority_extractions,
                    extraction_requirements=extraction_requirements,
                    base_instruction=base_instruction
                )

                logger.info(f"Prompt生成成功，长度: {len(prompt_content)} 字符")
                response_data = {
                    'prompt': prompt_content,
                    'schema_url': schema_url,
                    'message': 'Prompt生成成功'
                }
                
                # 添加可选字段到响应（如果提供）
                if sample_text_url:
                    response_data['sample_text_url'] = sample_text_url
                if sample_xlsx_url:
                    response_data['sample_xlsx_url'] = sample_xlsx_url
                if target_domain:
                    response_data['target_domain'] = target_domain
                if dictionary_url:
                    response_data['dictionary_url'] = dictionary_url
                if priority_extractions:
                    response_data['priority_extractions'] = priority_extractions
                if extraction_requirements:
                    response_data['extraction_requirements'] = extraction_requirements
                
                return jsonify(response_data)

            except ValueError as e:
                logger.warning(f"参数验证失败: {str(e)}")
                raise BadRequest(str(e))
            except Exception as e:
                logger.error(f"生成Prompt失败: {str(e)}")
                raise InternalServerError(f"生成Prompt失败: {str(e)}")

    def _register_error_handlers(self):
        """注册错误处理器"""

        @self.app.errorhandler(BadRequest)
        def handle_bad_request(e):
            return jsonify({
                'error': 'Bad Request',
                'message': str(e.description)
            }), 400

        @self.app.errorhandler(NotFound)
        def handle_not_found(e):
            return jsonify({
                'error': 'Not Found',
                'message': str(e.description)
            }), 404

        @self.app.errorhandler(InternalServerError)
        def handle_internal_error(e):
            return jsonify({
                'error': 'Internal Server Error',
                'message': str(e.description)
            }), 500

        @self.app.errorhandler(Exception)
        def handle_generic_error(e):
            logger.error(f"未处理的异常: {str(e)}")
            return jsonify({
                'error': 'Internal Server Error',
                'message': '服务器内部错误'
            }), 500

    def _sse_monitor_thread(self):
        """SSE监控线程，定期推送任务状态更新"""
        last_status_cache = {}

        while True:
            try:
                time.sleep(1)  # 每秒检查一次

                with self.sse_lock:
                    # 获取所有有SSE连接的任务
                    active_task_ids = list(self.sse_connections.keys())

                for task_id in active_task_ids:
                    try:
                        # 获取当前任务状态 (尝试两个服务)
                        current_status = self.service.get_task_status(task_id)
                        if not current_status:
                            current_status = self.open_ie_service.get_task_status(task_id)

                        if not current_status:
                            # 任务在两个服务都不存在，清理连接
                            with self.sse_lock:
                                if task_id in self.sse_connections:
                                    for queue in self.sse_connections[task_id]:
                                        try:
                                            queue.put(None)  # 结束信号
                                        except:
                                            pass
                                    del self.sse_connections[task_id]
                            continue

                        # 检查状态是否有变化
                        last_status = last_status_cache.get(task_id)

                        # 简单比较状态字典是否相等（注意：这里依赖字典转JSON的稳定性，或者可以只比对特定字段）
                        # 为了性能，可以只比较 'status', 'processed_files', 'updated_at' 等关键字段
                        status_changed = False
                        if not last_status:
                            status_changed = True
                        elif last_status['updated_at'] != current_status['updated_at']:
                            status_changed = True
                        elif last_status['status'] != current_status['status']:
                            status_changed = True

                        if status_changed:
                            # 状态有变化，推送更新
                            with self.sse_lock:
                                if task_id in self.sse_connections:
                                    for queue in self.sse_connections[task_id][:]:
                                        try:
                                            queue.put(current_status)
                                        except:
                                            try:
                                                self.sse_connections[task_id].remove(queue)
                                            except:
                                                pass

                            last_status_cache[task_id] = current_status

                        # 如果任务完成，清理连接
                        if current_status['status'] in ['completed', 'failed', 'cancelled']:
                            with self.sse_lock:
                                if task_id in self.sse_connections:
                                    for queue in self.sse_connections[task_id]:
                                        try:
                                            queue.put(None)  # 结束信号
                                        except:
                                            pass
                                    del self.sse_connections[task_id]

                            # 从缓存中移除
                            last_status_cache.pop(task_id, None)

                    except Exception as e:
                        logger.error(f"SSE监控任务 {task_id} 时出错: {str(e)}")

            except Exception as e:
                logger.error(f"SSE监控线程出错: {str(e)}")
                time.sleep(5)

    def _cleanup(self):
        """清理资源"""
        logger.info("正在清理API服务器资源...")

        # 清理所有SSE连接
        with self.sse_lock:
            for task_id, queues in self.sse_connections.items():
                for queue in queues:
                    try:
                        queue.put(None)  # 发送结束信号
                    except:
                        pass
            self.sse_connections.clear()

        if hasattr(self, 'service'):
            self.service.shutdown()

        if hasattr(self, 'open_ie_service'):
            self.open_ie_service.shutdown()

            # 关闭数据库连接池 (需要处理 asyncio 循环)
            # 由于 Flask 是同步的，这里可能需要一个简单的同步包装来运行异步关闭
            # 或者依赖 database.py 自身的析构（如果有的话）

        try:
            import asyncio
            from database import db_client
            # 创建一个新的 loop 来运行关闭操作，或者使用现有的
            loop = asyncio.new_event_loop()
            loop.run_until_complete(db_client.close_pool())
            loop.close()
        except Exception as e:
            logger.warning(f"关闭数据库连接池时出错: {e}")

        logger.info("API服务器资源清理完成")


    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在优雅关闭...")
        self._cleanup()
        sys.exit(0)

    def run(self):
        """启动服务器"""
        try:
            logger.info(f"启动API服务器，地址: http://{self.host}:{self.port}")

            # 打印可用的API端点
            logger.info("可用的API端点:")
            logger.info("  GET  /health - 健康检查")
            logger.info("  POST /api/v1/tasks - 创建任务（支持文件路径、文件名、URL，仅支持.txt格式）")
            logger.info("  GET  /api/v1/tasks/{task_id} - 获取任务状态")
            logger.info("  DELETE /api/v1/tasks/{task_id} - 取消任务")
            logger.info("  GET  /api/v1/tasks - 获取任务列表")
            logger.info("  GET  /api/v1/tasks/{task_id}/stream - 任务进度SSE推送")
            logger.info("  GET  /api/v1/service/status - 获取服务状态")
            logger.info("  POST /api/v1/service/cleanup - 清理旧任务")
            logger.info("  POST /api/v1/genprompt - 生成三元组抽取Prompt")

            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True,
                use_reloader=False  # 避免与我们的线程冲突
            )

        except Exception as e:
            logger.error(f"启动服务器失败: {str(e)}")
            raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="三元组抽取API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--max-workers", type=int, default=3, help="最大工作线程数")
    parser.add_argument("--output-dir", default="output", help="输出目录")

    args = parser.parse_args()

    # 设置环境变量
    os.environ.setdefault("OUTPUT_DIR", args.output_dir)
    os.environ.setdefault("MAX_WORKERS", str(args.max_workers))

    # 创建并启动服务器
    server = APIServer(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"服务器运行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()