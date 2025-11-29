#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenPrompt API 高级功能测试示例

展示如何使用新的参数来生成定制化的Prompt
"""

import requests
import json

# API服务地址
API_BASE_URL = "http://localhost:5000/api/v1"


def test_basic_prompt():
    """测试基础Prompt生成（向后兼容性）"""
    print("=" * 60)
    print("测试1: 基础Prompt生成")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json"
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        print(f"  Prompt长度: {len(result['prompt'])} 字符")
        print(f"  消息: {result['message']}")
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_with_target_domain():
    """测试带目标领域的Prompt生成"""
    print("=" * 60)
    print("测试2: 带目标领域的Prompt生成")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "target_domain": "建筑学领域知识图谱构建，重点关注建筑风格、设计理念和历史演变"
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        print(f"  目标领域: {result.get('target_domain')}")
        print(f"  Prompt长度: {len(result['prompt'])} 字符")
        
        # 检查Prompt中是否包含目标领域
        if "目标领域" in result['prompt']:
            print("  ✓ Prompt中包含目标领域部分")
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_with_priority_extractions():
    """测试带抽取优先级的Prompt生成"""
    print("=" * 60)
    print("测试3: 带抽取优先级的Prompt生成")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "target_domain": "建筑学领域",
        "priority_extractions": ["城市", "建筑师", "设计作品", "建筑风格"]
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        print(f"  优先级配置: {len(result.get('priority_extractions', []))} 项")
        
        # 检查Prompt中是否包含优先级部分
        if "抽取意向优先级" in result['prompt']:
            print("  ✓ Prompt中包含抽取优先级部分")
            if "城市, 建筑师, 设计作品, 建筑风格" in result['prompt']:
                print("  ✓ 优先级列表正确")
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_with_custom_instruction():
    """测试自定义指导语"""
    print("=" * 60)
    print("测试4: 自定义基础指导语")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "base_instruction": """你是一个医学知识图谱构建专家。
请从输入的医学文献中抽取以下内容：
1. 疾病名称及其分类
2. 症状描述及严重程度
3. 治疗方法和药物
4. 疾病与症状、治疗方法之间的关系

注意：所有医学术语必须保持原文准确性，不要进行同义词替换。""",
        "target_domain": "医学文献知识抽取"
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        
        # 检查是否包含自定义指导语
        if "医学知识图谱构建专家" in result['prompt']:
            print("  ✓ 使用了自定义指导语")
        if "实践活动" not in result['prompt'][:200]:
            print("  ✓ 未包含默认的实践活动指导语")
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_with_extraction_requirements():
    """测试自定义抽取要求"""
    print("=" * 60)
    print("测试5: 自定义抽取要求")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "target_domain": "法律文书分析",
        "extraction_requirements": """请按照以下要求进行抽取：
1. 保留所有法律条文的准确引用，包括具体的条款号
2. 明确区分原告和被告的行为和主张
3. 标注所有重要的时间节点（起诉时间、审理时间、判决时间等）
4. 抽取判决结果和法律依据
5. 对于专业法律术语，保持原文不变"""
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        print(f"  抽取要求已添加: {len(result.get('extraction_requirements', ''))} 字符")
        
        # 检查Prompt中是否包含抽取要求部分
        if "抽取要求描述" in result['prompt']:
            print("  ✓ Prompt中包含抽取要求描述部分")
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_full_configuration():
    """测试完整配置"""
    print("=" * 60)
    print("测试6: 完整配置（所有参数）")
    print("=" * 60)
    
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "sample_text_url": "http://localhost:5000/uploads/sample.txt",
        "sample_xlsx_url": "http://localhost:5000/uploads/sample_result.xlsx",
        "target_domain": "建筑学领域知识图谱构建",
        "priority_extractions": ["城市", "建筑师", "设计作品"],
        "extraction_requirements": "请特别关注建筑风格的描述，保留所有专业术语",
        "base_instruction": "你是一个建筑学知识图谱构建专家。请从建筑相关文献中抽取实体和关系。"
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt生成成功")
        print(f"  Schema URL: {result.get('schema_url')}")
        print(f"  Sample Text URL: {result.get('sample_text_url')}")
        print(f"  Sample XLSX URL: {result.get('sample_xlsx_url')}")
        print(f"  Target Domain: {result.get('target_domain')}")
        print(f"  Priority Extractions: {len(result.get('priority_extractions', []))} 项")
        print(f"  Extraction Requirements: {len(result.get('extraction_requirements', ''))} 字符")
        print(f"  Prompt长度: {len(result['prompt'])} 字符")
        
        # 显示Prompt的前500字符
        print("\nPrompt预览（前500字符）:")
        print("-" * 60)
        print(result['prompt'][:500])
        print("...")
        print("-" * 60)
    else:
        print(f"✗ 失败: {response.text}")
    print()


def test_invalid_priority_format():
    """测试无效的priority_extractions格式"""
    print("=" * 60)
    print("测试7: 无效的priority_extractions格式（预期失败）")
    print("=" * 60)
    
    # 测试1: 不是数组
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "priority_extractions": "invalid format"
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 400:
        print("✓ 正确拒绝了非数组格式")
    else:
        print(f"✗ 应该返回400错误")
    
    # 测试2: 数组中包含非字符串元素
    data = {
        "schema_url": "http://localhost:5000/uploads/schema.json",
        "priority_extractions": ["城市", 123, "建筑师"]  # 包含数字
    }
    
    response = requests.post(f"{API_BASE_URL}/genprompt", json=data)
    if response.status_code == 400:
        print("✓ 正确拒绝了包含非字符串元素的数组")
    else:
        print(f"✗ 应该返回400错误")
    print()


def save_prompt_to_file(prompt_text, filename):
    """保存Prompt到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"✓ Prompt已保存到: {filename}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GenPrompt API 高级功能测试")
    print("=" * 60 + "\n")
    
    try:
        # 运行各项测试
        test_basic_prompt()
        test_with_target_domain()
        test_with_priority_extractions()
        test_with_custom_instruction()
        test_with_extraction_requirements()
        test_full_configuration()
        test_invalid_priority_format()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ 错误: 无法连接到API服务")
        print(f"  请确保服务运行在 {API_BASE_URL}")
        print("  可以运行: python api_server.py")
    except Exception as e:
        print(f"\n✗ 测试过程中出错: {str(e)}")


if __name__ == "__main__":
    main()

