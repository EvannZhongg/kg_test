import requests
import time
import json
import os
import asyncio
import csv
from pathlib import Path
from dotenv import load_dotenv

# 尝试导入数据库客户端用于结果验证
try:
    from database import db_client

    DB_AVAILABLE = True
except ImportError:
    print("Warning: database.py not found, skipping database verification.")
    DB_AVAILABLE = False

# 加载环境变量
load_dotenv()

# === 配置 ===
API_BASE_URL = "http://127.0.0.1:8001"
# 测试文本目录
TEST_TXT_DIR = Path(r"D:\Personal_Project\kgplatform_backend\python-service\example")
OUTPUT_DIR = Path("output_api_test_results")
TEST_PROJECT_ID = 9997
# Prompt 文件路径
PROMPT_FILE = Path("triple_extraction_prompt2.txt")

# 根据 .env 配置使用 forward provider
PROVIDER = "forward"
API_KEY = os.getenv("FORWARD_API_KEY", "")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not TEST_TXT_DIR.exists():
    TEST_TXT_DIR.mkdir(parents=True, exist_ok=True)
    # 创建一个包含重复信息的测试文件，用于测试权重累加
    (TEST_TXT_DIR / "test_degree_weight.txt").write_text(
        "马云创立了阿里巴巴。马云创立了阿里巴巴。马云出生于杭州。",
        encoding="utf-8"
    )


def create_extraction_task(files):
    url = f"{API_BASE_URL}/api/v1/tasks"

    # 读取 Prompt 文件
    if PROMPT_FILE.exists():
        print(f"正在使用本地 Prompt 文件: {PROMPT_FILE.absolute()}")
        prompt_content = PROMPT_FILE.read_text(encoding="utf-8")
    else:
        print("警告: 本地 Prompt 文件不存在，使用默认硬编码 Prompt")
        prompt_content = """
        你是一个信息抽取助手。请从文本中抽取实体和关系。
        请严格按照以下JSON格式输出：
        ```json
        {
          "triples": [
            ["头实体", "头类型", "关系", "关系类型", "尾实体", "尾类型"]
          ]
        }
        ```
        """

    files_payload = []
    for i, file_path in enumerate(files, 1):
        files_payload.append({
            "url": str(file_path.absolute()),
            "material_id": 1000 + i
        })

    payload = {
        "task_type": "schema",
        "project_id": TEST_PROJECT_ID,
        "files": files_payload,
        "provider": PROVIDER,
        "api_key": API_KEY,
        "model": "",
        "prompt_text": prompt_content  # <--- 使用读取到的 Prompt 内容
    }

    try:
        print(f"正在发送请求到 {url} ...")
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json().get("task_id")
    except Exception as e:
        print(f"请求失败: {e}")
        return None


def monitor_task(task_id):
    url = f"{API_BASE_URL}/api/v1/tasks/{task_id}"
    print(f"开始监控任务 {task_id} ...")

    while True:
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            status_data = resp.json()
            status = status_data.get("status")

            print(f"Status: {status} | 进度: {status_data.get('processed_files')}/{status_data.get('total_files')}")

            if status in ["completed", "failed", "cancelled"]:
                if status == "failed":
                    print("错误列表:", json.dumps(status_data.get("errors", []), ensure_ascii=False, indent=2))
                return status_data

        except Exception as e:
            print(f"获取状态失败: {e}")

        time.sleep(2)


async def verify_database_data():
    """【修改重点】增强的数据库验证逻辑"""
    if not DB_AVAILABLE:
        return

    print("\n=== 正在验证数据库数据 (新特性检查) ===")
    pool = await db_client.get_pool()
    async with pool.acquire() as conn:
        # 1. 验证 Chunks 及文件名记录
        print("\n[1. 检查 Chunks 表]")
        chunks = await conn.fetch(
            "SELECT id, file_name, substring(text_content, 1, 20) as preview FROM limited_graph_chunks WHERE project_id = $1",
            TEST_PROJECT_ID
        )
        print(f"  -> 找到 {len(chunks)} 个 Chunk")
        for c in chunks:
            print(f"     File: {c['file_name']} | Preview: {c['preview']}...")

        # 2. 验证 Nodes (重点检查 Degree 和 Weight)
        print("\n[2. 检查 Nodes 表 (验证 Degree & Weight)]")
        nodes = await conn.fetch(
            """
            SELECT label, entity_type, weight, degree 
            FROM limited_graph_nodes 
            WHERE project_id = $1 
            ORDER BY degree DESC, weight DESC 
            LIMIT 10
            """,
            TEST_PROJECT_ID
        )

        print(f"  -> Top 10 Nodes:")
        print(f"     {'Label':<15} | {'Type':<10} | {'Weight':<6} | {'Degree':<6}")
        print(f"     {'-' * 15}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 6}")
        for n in nodes:
            print(f"     {n['label']:<15} | {n['entity_type']:<10} | {n['weight']:<6} | {n['degree']:<6}")

        # 验证逻辑：如果输入文本有重复句子，Weight 应该 > 1
        has_weight_gt_1 = any(n['weight'] > 1 for n in nodes)
        # 验证逻辑：如果有关系连接，Degree 应该 > 0
        has_degree_gt_0 = any(n['degree'] > 0 for n in nodes)

        if has_weight_gt_1:
            print("  [√] 验证通过: 检测到权重累加 (Weight > 1)，去重逻辑生效。")
        else:
            print("  [?] 警告: 未检测到权重累加，可能是文本无重复或抽取不稳定。")

        if has_degree_gt_0:
            print("  [√] 验证通过: 检测到节点度数 (Degree > 0)，图计算逻辑生效。")
        else:
            print("  [x] 失败: 所有节点 Degree 均为 0，请检查 increment_limited_nodes_degree 调用。")

        # 3. 验证 Edges
        print("\n[3. 检查 Edges 表]")
        edges = await conn.fetch(
            """
            SELECT s.label as src, e.relation, t.label as tgt, e.weight
            FROM limited_graph_edges e
            JOIN limited_graph_nodes s ON e.source_id = s.id
            JOIN limited_graph_nodes t ON e.target_id = t.id
            WHERE e.project_id = $1 
            ORDER BY e.weight DESC
            LIMIT 5
            """,
            TEST_PROJECT_ID
        )
        for e in edges:
            print(f"     {e['src']} --[{e['relation']}]--> {e['tgt']} (Weight: {e['weight']})")

        # 4. 导出 CSV (包含自定义 SQL)
        print("\n[4. 导出 CSV]")

        # 导出 Nodes (默认全字段，除去 embedding)
        await export_to_csv(conn, "limited_graph_nodes", "nodes_schema.csv")

        # 导出 Edges (使用自定义 SQL，替换 ID 为 Label)
        edges_sql_with_names = """
            SELECT 
                e.id, 
                e.edge_id, 
                e.project_id,
                s.label AS source,      -- 替换 source_id 为 source_label
                t.label AS target,      -- 替换 target_id 为 target_label
                e.relation, 
                e.relation_type, 
                e.weight, 
                e.source_chunk_ids,
                e.created_at
            FROM limited_graph_edges e
            JOIN limited_graph_nodes s ON e.source_id = s.id
            JOIN limited_graph_nodes t ON e.target_id = t.id
            WHERE e.project_id = $1
            ORDER BY e.weight DESC
        """
        await export_to_csv(conn, "limited_graph_edges", "edges_schema.csv", custom_sql=edges_sql_with_names)

    await db_client.close_current_pool()


async def export_to_csv(conn, table_name, filename, custom_sql=None):
    filepath = OUTPUT_DIR / filename

    # 如果提供了自定义 SQL，则使用它；否则使用默认的全表查询
    if custom_sql:
        rows = await conn.fetch(custom_sql, TEST_PROJECT_ID)
    else:
        rows = await conn.fetch(f"SELECT * FROM {table_name} WHERE project_id = $1", TEST_PROJECT_ID)

    if rows:
        # 1. 获取所有列名
        all_keys = list(rows[0].keys())

        # 2. 过滤掉 'embedding' 列
        keys = [k for k in all_keys if k != 'embedding']

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(keys)

            for row in rows:
                # 3. 只写入不在排除列表中的列
                writer.writerow([str(row[k]) for k in keys])

        print(f"  -> 已导出 {table_name} 到 {filepath} (已剔除 embedding)")
    else:
        print(f"  -> 表 {table_name} (或查询结果) 为空，跳过导出")


def main():
    files = list(TEST_TXT_DIR.glob("*.txt"))
    if not files:
        print("没有找到测试文件")
        return

    task_id = create_extraction_task(files)
    if not task_id: return

    result = monitor_task(task_id)

    if result and result.get("status") == "completed" and DB_AVAILABLE:
        asyncio.run(verify_database_data())


if __name__ == "__main__":
    main()