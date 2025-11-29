import os
from pathlib import Path
from dotenv import load_dotenv

# === 新增：强制加载当前目录下的 .env 文件 ===
# 确保在读取 os.getenv 之前加载
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # 服务实例配置
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
    OUTPUT_BASE_DIR = os.getenv("OUTPUT_BASE_DIR", "output")

    # 任务配置
    TASK_CLEANUP_DAYS = int(os.getenv("TASK_CLEANUP_DAYS", 7))
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 800))

    # API限制
    MAX_FILES_PER_TASK = int(os.getenv("MAX_FILES_PER_TASK", 50))
    MAX_TASKS_PER_LIST = int(os.getenv("MAX_TASKS_PER_LIST", 100))

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "api_server.log")

    # AI服务默认配置
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.0))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 1.0))

    # PostgreSQL 数据库配置
    PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
    PG_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    print(f"Config loaded PG_PORT: {PG_PORT} from env: {os.getenv('POSTGRES_PORT')}")
    PG_USER = os.getenv("POSTGRES_USER", "postgres")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    PG_DB = os.getenv("POSTGRES_DB", "kgplatform_chidu")

    # 连接池配置
    PG_MIN_SIZE = int(os.getenv("POSTGRES_MIN_SIZE", 1))
    PG_MAX_SIZE = int(os.getenv("POSTGRES_MAX_SIZE", 10))

    # Embedding 服务配置
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 10))

    @classmethod
    def init_app(cls, app):
        """初始化Flask应用配置"""
        app.config.from_object(cls)


class DevelopmentConfig(Config):
    DEBUG = True
    MAX_WORKERS = 2


class ProductionConfig(Config):
    DEBUG = False
    MAX_WORKERS = 5


class TestingConfig(Config):
    TESTING = True
    MAX_WORKERS = 1
    OUTPUT_BASE_DIR = "test_output"


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}