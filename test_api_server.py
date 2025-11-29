import requests
import json
import time
import os
from pathlib import Path

# ================= 配置区域 =================

CONFIG = {
    # API 服务器地址
    "api_base_url": "http://localhost:8001",

    # AI 提供商配置
    "provider": "deepseek",  # 或 "qwen"
    "api_key": "sk-1bc317ee3858458d9648944a2184e4df",  # 您的 API Key

    # === 1. Prompt 生成参数 (测试核心) ===
    # 注意：API server 需要通过 URL 下载这些配置文件
    # 您可以在本地起一个 python -m http.server 8080 来服务这些文件
    "gen_prompt_params": {
        # Schema URL (必须)
        "schema_url": "http://localhost:8080/knowledge_graph_schema.json",

        # 目标领域
        "target_domain": "建筑学与社会实践",

        # 专业词典 URL (用于测试归一化)
        "dictionary_url": "http://localhost:8080/dictionary.txt",

        # 抽取优先级 (用于测试思维链 Round)
        "priority_extractions": [
            "实践活动",
            "参与人物",
            "实践成果"
        ],

        # 自定义要求
        "extraction_requirements": "请特别注意区分活动的主办方和承办方。",

        # 样例数据 (可选)
        # "sample_text_url": "http://localhost:8080/sample.txt",
        # "sample_xlsx_url": "http://localhost:8080/sample.xlsx"
    },

    # === 2. 任务测试文件 ===
    # 本地待抽取的测试文件路径
    "test_file_path": r"C:\Users\YourName\Documents\test_article.txt"
}


# ================= 功能函数 =================

def test_generate_prompt():
    """测试 Prompt 生成接口"""
    url = f"{CONFIG['api_base_url']}/api/v1/genprompt"
    print(f"\n[1] 正在请求生成 Prompt: {url}")
    print(f"    参数: {json.dumps(CONFIG['gen_prompt_params'], ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(url, json=CONFIG['gen_prompt_params'])
        response.raise_for_status()
        result = response.json()

        print("\n✅ Prompt 生成成功!")
        print("-" * 40)
        # 打印生成的 Prompt 前 500 字符和关键部分，供检查
        prompt_content = result['prompt']
        print(f"Prompt 长度: {len(prompt_content)} 字符")

        # 检查关键特征是否包含在 Prompt 中
        checks = {
            "数组格式要求": "使用紧凑的数组格式",
            "思维链 Round": "Round 1",
            "归一化规则": "指代消解",
            "目标领域": CONFIG['gen_prompt_params']['target_domain']
        }

        print("\n关键特征检查:")
        for feature, keyword in checks.items():
            status = "✔ 存在" if keyword in prompt_content else "❌ 未找到"
            print(f"  - {feature}: {status}")

        print("-" * 40)
        return prompt_content

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Prompt 生成失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"    错误详情: {e.response.text}")
        return None


def test_create_and_monitor_task(prompt_text):
    """测试任务创建与执行"""
    # 1. 构造任务请求
    url = f"{CONFIG['api_base_url']}/api/v1/tasks"
    file_path = CONFIG['test_file_path']

    if not os.path.exists(file_path):
        print(f"\n❌ 测试文件不存在: {file_path}")
        # 创建一个临时文件用于测试
        print("    正在创建临时测试文件...")
        with open("temp_test.txt", "w", encoding="utf-8") as f:
            f.write("2023年，东南大学建筑学院的张三教授带领团队在南京进行了乡村振兴实践。")
        file_path = os.path.abspath("temp_test.txt")

    payload = {
        "files": [file_path],  # 支持绝对路径
        "prompt_text": prompt_text,  # 使用刚才生成的 Prompt
        "provider": CONFIG['provider'],
        "api_key": CONFIG['api_key']
    }

    print(f"\n[2] 正在创建抽取任务...")

    try:
        # 创建任务
        response = requests.post(url, json=payload)
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data['task_id']
        print(f"✅ 任务创建成功! ID: {task_id}")

        # 监控进度
        print("\n[3] 开始监控任务进度...")
        while True:
            status_url = f"{CONFIG['api_base_url']}/api/v1/tasks/{task_id}"
            resp = requests.get(status_url)
            status = resp.json()

            state = status['status']
            processed = status.get('processed_files', 0)
            total = status.get('total_files', 0)

            print(f"    >> 状态: {state} | 进度: {processed}/{total}")

            if state in ['completed', 'failed', 'cancelled']:
                break

            time.sleep(2)

        # 结果展示
        print(f"\n[4] 任务结束. 最终状态: {state}")
        if state == 'completed':
            results = status.get('results', [])
            for res in results:
                print(f"\n    📄 文件: {res['file_name']}")
                print(f"    📊 三元组数量: {res['triples_count']}")

                # 读取并展示输出文件内容（前几行）
                out_file = res['output_files']['jsonl']
                if os.path.exists(out_file):
                    print(f"    💾 输出路径: {out_file}")
                    print("    📝 抽取结果预览 (Array Format):")
                    with open(out_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        # 打印前 2 个三元组
                        print(json.dumps(content[:2], indent=2, ensure_ascii=False))
                else:
                    print(f"    ❌ 输出文件未找到: {out_file}")
        else:
            print(f"❌ 任务失败原因: {status.get('errors', '未知错误')}")

    except Exception as e:
        print(f"\n❌ 任务执行出错: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("       API Server 全流程测试脚本")
    print("=" * 50)

    # 步骤 1: 生成 Prompt
    generated_prompt = test_generate_prompt()

    # 步骤 2: 如果 Prompt 生成成功，则使用它去跑任务
    if generated_prompt:
        test_create_and_monitor_task(generated_prompt)
    else:
        print("\n⚠️ 由于 Prompt 生成失败，跳过任务执行测试。")