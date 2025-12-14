import requests
import json
import time
import os
import subprocess
from dotenv import load_dotenv  # 新增：导入 dotenv

# 加载当前目录下的 .env 文件
load_dotenv()

# ================= 配置区域 =================

# API 服务地址
API_BASE_URL = "http://localhost:8001"

# 测试文件路径 (请修改为您本地实际存在的路径)
TEST_FILE_PATH = r"D:\Personal_Project\kgplatform_backend\python-service\txt_test\水浒传.txt"

# API 配置
CONFIG = {
    "provider": "forward",  # 修改：使用 forward 提供商模式
    "api_key": os.getenv("FORWARD_API_KEY"),           # 从 .env 读取
    "base_url": os.getenv("FORWARD_BASE_URL"),         # 从 .env 读取
    "model": os.getenv("FORWARD_DEFAULT_MODEL"),       # 从 .env 读取

    # OpenIE 必需参数 (建议每次测试换一个新的 ID 以便观察全新数据)
    "project_id": 158,
    "material_id": 2001,
}


# ===========================================

def verify_database_content(project_id):
    """
    通过 Docker 执行 SQL 验证数据库内容
    重点查看：描述(Description)、权重(Weight)、度(Degree)、向量(Vector)、来源文件(file_name)
    """
    print(f"\n{'=' * 20} 数据库存储验证 (Project ID: {project_id}) {'=' * 20}")

    # 定义查询
    queries = [
        {
            "name": "1. 分块表 (open_graph_chunks) - 检查原文、向量和文件名",
            "sql": f"""
                SELECT chunk_index, 
                       file_name,            -- <--- [新增] 检查文件名是否正确存入
                       left(id, 20) as chunk_id_prefix, 
                       left(text_content, 30) as content_preview, 
                       (embedding IS NOT NULL) as has_vector 
                FROM open_graph_chunks 
                WHERE project_id = {project_id} 
                ORDER BY chunk_index ASC LIMIT 5;
            """
        },
        {
            "name": "2. 实体表 (open_graph_nodes) - 检查【描述】、【权重】、【度】",
            "sql": f"""
                SELECT id, 
                       label, 
                       weight, 
                       degree, 
                       left(description, 30) as desc_preview, 
                       (embedding IS NOT NULL) as has_vec 
                FROM open_graph_nodes 
                WHERE project_id = {project_id} 
                ORDER BY weight DESC LIMIT 10;
            """
        },
        {
            "name": "3. 关系表 (open_graph_edges) - 检查【描述】、【权重】",
            "sql": f"""
                SELECT e.id, 
                       s.label as source, 
                       e.relation, 
                       t.label as target, 
                       e.weight,
                       left(e.description, 30) as desc_preview,
                       (e.embedding IS NOT NULL) as has_vec
                FROM open_graph_edges e 
                JOIN open_graph_nodes s ON e.source_id = s.id 
                JOIN open_graph_nodes t ON e.target_id = t.id 
                WHERE e.project_id = {project_id} 
                ORDER BY e.weight DESC LIMIT 10;
            """
        }
    ]

    for q in queries:
        print(f"\n--- {q['name']} ---")
        cmd = [
            "docker", "exec", "-i", "pgvector-db",
            "psql", "-U", "postgres", "-d", "kgplatform_chidu",
            "-c", q['sql']
        ]

        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode == 0:
                print(result.stdout)
            else:
                print("❌ 查询失败:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ 执行异常: {e}")


def test_open_ie_task():
    # 1. 检查文件是否存在
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ 错误：测试文件不存在 -> {TEST_FILE_PATH}")
        return

    # 2. 构造请求 URL 和 Payload
    url = f"{API_BASE_URL}/api/v1/tasks"

    payload = {
        "task_type": "open_ie",
        "project_id": CONFIG["project_id"],
        "provider": CONFIG["provider"],
        "api_key": CONFIG["api_key"],
        "model": CONFIG.get("model"),
        "base_url": CONFIG.get("base_url"),

        "files": [
            {
                "material_id": CONFIG["material_id"],
                "url": TEST_FILE_PATH
            }
        ]
    }

    print(f"\n[1] 正在提交 OpenIE 任务...")
    print(f"    目标文件: {TEST_FILE_PATH}")
    print(f"    项目ID: {CONFIG['project_id']}")
    print(f"    Provider: {CONFIG['provider']}")
    print(f"    Base URL: {CONFIG.get('base_url')}")
    print(f"    Model: {CONFIG.get('model')}")

    try:
        # 3. 发送创建任务请求
        response = requests.post(url, json=payload)

        if response.status_code != 201:
            print(f"❌ 创建失败 (Status {response.status_code}):")
            print(response.text)
            return

        data = response.json()
        task_id = data.get("task_id")
        print(f"✅ 任务创建成功! Task ID: {task_id}")

        # 4. 轮询监控进度
        print(f"\n[2] 开始监控任务进度...")
        start_time = time.time()

        while True:
            status_url = f"{API_BASE_URL}/api/v1/tasks/{task_id}"
            status_resp = requests.get(status_url)

            if status_resp.status_code != 200:
                print(f"    获取状态失败: {status_resp.text}")
                break

            status_data = status_resp.json()
            state = status_data.get("status")
            processed = status_data.get("processed_files", 0)
            total = status_data.get("total_files", 0)
            total_triples = status_data.get("total_triples", 0)

            elapsed = int(time.time() - start_time)
            print(f"\r    >> [{elapsed}s] 状态: {state} | 进度: {processed}/{total} | 已抽取: {total_triples}", end="")

            if state in ["completed", "failed", "cancelled"]:
                print(f"\n\n[3] 任务结束. 最终状态: {state}")

                if state == "completed":
                    # === 任务成功后，执行数据库验证 ===
                    verify_database_content(CONFIG["project_id"])

                if state == "failed":
                    errors = status_data.get("errors", [])
                    print("❌ 错误信息:")
                    print(json.dumps(errors, indent=2, ensure_ascii=False))
                break

            time.sleep(2)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到服务器 {API_BASE_URL}，请确认 python-service 是否已启动。")
    except Exception as e:
        print(f"\n❌ 发生异常: {str(e)}")


if __name__ == "__main__":
    test_open_ie_task()