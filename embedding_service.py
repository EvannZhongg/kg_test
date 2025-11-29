import os
import logging
import requests
from typing import List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# === 1. 强制加载 .env 文件 ===
current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

from config import config

# 获取当前环境配置
env = os.getenv("FLASK_ENV", "default")
cfg = config.get(env, config['default'])

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        """
        初始化 Embedding 服务
        """
        # 1. Base URL
        self.default_base_url = getattr(cfg, "EMBEDDING_BASE_URL", None)
        if not self.default_base_url:
            self.default_base_url = os.getenv("EMBEDDING_BASE_URL", os.getenv("DASHSCOPE_BASE_URL"))

        # 2. API Key
        self.default_api_key = getattr(cfg, "EMBEDDING_API_KEY", None)
        if not self.default_api_key:
            self.default_api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY"))

        # 3. Model
        self.default_model = getattr(cfg, "EMBEDDING_MODEL", None)
        if not self.default_model:
            self.default_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

        # 4. Batch Size (新增)
        self.default_batch_size = getattr(cfg, "EMBEDDING_BATCH_SIZE", 10)

        logger.info("=== Embedding Service 配置 ===")
        logger.info(f"Base URL:   {self.default_base_url}")
        logger.info(f"Model:      {self.default_model}")
        logger.info(f"Batch Size: {self.default_batch_size}")
        if self.default_api_key:
            masked_key = self.default_api_key[:6] + "*" * 6 + self.default_api_key[-4:]
            logger.info(f"API Key:    {masked_key}")
        else:
            logger.warning("API Key:    [未配置!]")
        logger.info("===============================")

    def get_embeddings(self, texts: List[str], base_url: str = None, api_key: str = None,
                       model: str = None, batch_size: int = None) -> List[List[float]]:
        """
        批量获取文本向量 (自动分批处理)
        """
        if not texts:
            return []

        # 确定参数
        use_base = (base_url or self.default_base_url).rstrip('/')
        if use_base.endswith("/embeddings"):
            url = use_base
        else:
            url = f"{use_base}/embeddings"

        use_key = api_key or self.default_api_key
        use_model = model or self.default_model
        # 使用传入的 batch_size 或默认配置
        use_batch_size = batch_size or self.default_batch_size

        if not use_key:
            logger.error("未配置 Embedding API Key，无法获取向量")
            return [None] * len(texts)

        headers = {
            "Authorization": f"Bearer {use_key}",
            "Content-Type": "application/json"
        }

        # 初始化结果列表 (全为 None)
        all_embeddings = [None] * len(texts)

        # === 分批处理循环 ===
        total_texts = len(texts)
        for start_idx in range(0, total_texts, use_batch_size):
            end_idx = min(start_idx + use_batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]

            # 预处理：替换换行符
            clean_batch = [str(t).replace("\n", " ") for t in batch_texts]

            try:
                payload = {
                    "input": clean_batch,
                    "model": use_model,
                    "encoding_format": "float"
                }

                # logger.info(f"正在请求向量批次: {start_idx+1}-{end_idx} / {total_texts}")

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code != 200:
                    logger.error(f"Batch {start_idx}-{end_idx} Failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    # 当前批次保持为 None，继续处理下一批
                    continue

                data = response.json()

                if "data" in data and isinstance(data["data"], list):
                    # 将批次结果填入总结果列表
                    for item in data["data"]:
                        # API 返回的 index 通常是批次内的索引 (0 ~ batch_size-1)
                        batch_inner_idx = item.get("index")
                        vec = item.get("embedding")

                        if batch_inner_idx is not None:
                            global_idx = start_idx + batch_inner_idx
                            if global_idx < len(all_embeddings):
                                all_embeddings[global_idx] = vec
                        else:
                            # 如果 API 没返回 index，假设按顺序返回 (兜底)
                            # 这种情况较少见
                            pass

                    # 如果上述 index 逻辑没覆盖（例如 API 没返回 index 字段）
                    # 尝试按顺序填充
                    current_filled = False
                    for i in range(start_idx, end_idx):
                        if all_embeddings[i] is not None:
                            current_filled = True
                            break

                    if not current_filled and len(data["data"]) == len(batch_texts):
                        for i, item in enumerate(data["data"]):
                            all_embeddings[start_idx + i] = item.get("embedding")

                else:
                    logger.error(f"Batch {start_idx}-{end_idx} Format Error: {data}")

            except Exception as e:
                logger.error(f"Batch {start_idx}-{end_idx} Exception: {str(e)}")
                # 继续下一批

        return all_embeddings


# 创建全局单例
embedding_service = EmbeddingService()