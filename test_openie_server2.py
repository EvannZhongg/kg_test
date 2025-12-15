import requests
import json
import time
import os
import subprocess
import csv
from pathlib import Path
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件
load_dotenv()

# ================= 配置区域 =================

# API 服务地址
API_BASE_URL = "http://localhost:8001"

# 测试文件夹路径 (修改为文件夹路径，脚本将读取该目录下所有 .txt 文件)
TEST_DIR_PATH = r"D:\Personal_Project\kgplatform_backend\python-service\example"

# API 配置
CONFIG = {
    "provider": "forward",
    "api_key": os.getenv("FORWARD_API_KEY"),
    "base_url": os.getenv("FORWARD_BASE_URL"),
    "model": os.getenv("FORWARD_DEFAULT_MODEL"),

    # OpenIE 必需参数
    "project_id": 66,
    "start_material_id": 2001,  # 修改：作为起始素材ID，后续文件自动累加
}

# 导出文件的保存目录
EXPORT_DIR = "exports"


# ===========================================

def run_sql_export_csv(sql, output_file):
    """
    通过 docker exec 执行 SQL 并将结果导出为 CSV
    """
    # 确保导出目录存在
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    full_path = Path(EXPORT_DIR) / output_file

    print(f"    正在导出到: {full_path} ...")

    # 使用 PSQL 的 COPY ... TO STDOUT (CSV HEADER) 命令
    # 注意：需要转义 SQL 中的双引号
    copy_cmd = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"

    cmd = [
        "docker", "exec", "-i", "pgvector-db",
        "psql", "-U", "postgres", "-d", "kgplatform_chidu",
        "-c", copy_cmd
    ]

    try:
        with open(full_path, "w", encoding="utf-8-sig", newline="") as f:
            # 执行命令，将标准输出直接写入文件
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'  # 容器输出通常是 UTF-8
            )

        if result.returncode != 0:
            print(f"❌ 导出失败: {result.stderr}")
        else:
            print(f"✅ 导出成功: {output_file}")

    except Exception as e:
        print(f"❌ 导出异常: {e}")


def export_project_data_to_csv(project_id):
    """
    导出指定项目的所有实体和关系到 CSV，包含来源文件名等详细信息
    """
    print(f"\n{'=' * 20} 开始导出项目数据 (Project ID: {project_id}) {'=' * 20}")

    # 1. 导出实体 (Nodes)
    # 使用子查询聚合 source_chunk_ids 对应的文件名
    nodes_sql = f"""
        SELECT 
            n.id,
            n.label,
            n.entity_type,
            n.weight,
            n.degree,
            n.description,
            -- 聚合来源 Chunk ID
            array_to_string(n.source_chunk_ids, '|') as chunk_ids,
            -- 聚合来源文件名 (从 chunks 表关联)
            (
                SELECT string_agg(DISTINCT c.file_name, ' | ')
                FROM open_graph_chunks c
                WHERE c.id = ANY(n.source_chunk_ids)
            ) as source_file_names
        FROM open_graph_nodes n
        WHERE n.project_id = {project_id}
        ORDER BY n.weight DESC
    """
    run_sql_export_csv(nodes_sql, f"project_{project_id}_nodes.csv")

    # 2. 导出关系 (Edges)
    edges_sql = f"""
        SELECT 
            e.id,
            s.label as source_node,
            e.relation,
            t.label as target_node,
            e.weight,
            e.description,
            -- 聚合来源 Chunk ID
            array_to_string(e.source_chunk_ids, '|') as chunk_ids,
            -- 聚合来源文件名
            (
                SELECT string_agg(DISTINCT c.file_name, ' | ')
                FROM open_graph_chunks c
                WHERE c.id = ANY(e.source_chunk_ids)
            ) as source_file_names
        FROM open_graph_edges e
        JOIN open_graph_nodes s ON e.source_id = s.id
        JOIN open_graph_nodes t ON e.target_id = t.id
        WHERE e.project_id = {project_id}
        ORDER BY e.weight DESC
    """
    run_sql_export_csv(edges_sql, f"project_{project_id}_edges.csv")


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
                       file_name,            -- 检查文件名是否正确存入
                       left(id, 20) as chunk_id_prefix, 
                       left(text_content, 30) as content_preview, 
                       (embedding IS NOT NULL) as has_vector 
                FROM open_graph_chunks 
                WHERE project_id = {project_id} 
                ORDER BY file_id, chunk_index ASC LIMIT 5;
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
                ORDER BY weight DESC LIMIT 5;
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
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ 查询失败: {result.stderr}")
        except Exception as e:
            print(f"❌ 执行异常: {e}")


def test_open_ie_task():
    # 1. 检查目录是否存在
    dir_path = Path(TEST_DIR_PATH)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"❌ 错误：测试目录不存在或不是文件夹 -> {TEST_DIR_PATH}")
        return

    # 2. 获取目录下所有 .txt 文件
    txt_files = list(dir_path.glob("*.txt"))
    if not txt_files:
        print(f"❌ 错误：在目录中未找到 .txt 文件 -> {TEST_DIR_PATH}")
        return

    print(f"📂 找到 {len(txt_files)} 个文本文件，准备批量处理...")

    # 3. 构造 files 列表
    files_payload = []
    start_id = CONFIG["start_material_id"]

    for idx, file_obj in enumerate(txt_files):
        # 使用绝对路径
        abs_path = str(file_obj.resolve())
        # 为每个文件分配递增的 material_id
        current_material_id = start_id + idx

        files_payload.append({
            "material_id": current_material_id,
            "url": abs_path
        })
        print(f"  - [{current_material_id}] {file_obj.name}")

    # 4. 构造请求 Payload
    url = f"{API_BASE_URL}/api/v1/tasks"

    payload = {
        "task_type": "open_ie",
        "project_id": CONFIG["project_id"],
        "provider": CONFIG["provider"],
        "api_key": CONFIG["api_key"],
        "model": CONFIG.get("model"),
        "base_url": CONFIG.get("base_url"),
        "files": files_payload  # 传入文件列表
    }

    print(f"\n[1] 正在提交 OpenIE 任务 (批量模式)...")
    print(f"    文件数量: {len(files_payload)}")
    print(f"    项目ID: {CONFIG['project_id']}")

    try:
        # 5. 发送创建任务请求
        response = requests.post(url, json=payload)

        if response.status_code != 201:
            print(f"❌ 创建失败 (Status {response.status_code}):")
            print(response.text)
            return

        data = response.json()
        task_id = data.get("task_id")
        print(f"✅ 任务创建成功! Task ID: {task_id}")

        # 6. 轮询监控进度
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
                    # === 验证数据库 ===
                    verify_database_content(CONFIG["project_id"])

                    # === 导出 CSV ===
                    export_project_data_to_csv(CONFIG["project_id"])

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