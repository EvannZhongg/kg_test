#!/usr/bin/env python3
"""
测试genprompt接口对不同文件格式的支持
"""
import requests
import json

def test_genprompt_formats():
    """测试不同文件格式的genprompt接口"""
    
    base_url = "http://localhost:8000"
    
    # 测试用例
    test_cases = [
        {
            "name": "基础测试 - 仅schema",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json"
            }
        },
        {
            "name": "txt样例测试",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_text_url": "https://example.com/sample.txt"
            }
        },
        {
            "name": "docx样例测试",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_text_url": "https://example.com/sample.docx"
            }
        },
        {
            "name": "xlsx样例测试",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_xlsx_url": "https://example.com/sample.xlsx"
            }
        },
        {
            "name": "完整样例测试",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_text_url": "https://example.com/sample.docx",
                "sample_xlsx_url": "https://example.com/sample.xlsx"
            }
        }
    ]
    
    print("=== genprompt接口格式支持测试 ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"请求数据: {json.dumps(test_case['data'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{base_url}/api/v1/genprompt",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 成功")
                print(f"Prompt长度: {len(result.get('prompt', ''))} 字符")
                print(f"消息: {result.get('message', '')}")
            else:
                print(f"❌ 失败 - HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"错误信息: {error_data}")
                except:
                    print(f"响应内容: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {str(e)}")
        
        print("-" * 50)

def test_invalid_formats():
    """测试无效格式的错误处理"""
    
    base_url = "http://localhost:8000"
    
    invalid_cases = [
        {
            "name": "无效的文本格式",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_text_url": "https://example.com/sample.pdf"  # 不支持的格式
            }
        },
        {
            "name": "无效的Excel格式",
            "data": {
                "schema_url": "https://raw.githubusercontent.com/example/schema.json",
                "sample_xlsx_url": "https://example.com/sample.xls"  # 不支持的格式
            }
        },
        {
            "name": "缺少必需参数",
            "data": {
                "sample_text_url": "https://example.com/sample.txt"
                # 缺少schema_url
            }
        }
    ]
    
    print("\n=== 无效格式错误处理测试 ===\n")
    
    for i, test_case in enumerate(invalid_cases, 1):
        print(f"错误测试 {i}: {test_case['name']}")
        print(f"请求数据: {json.dumps(test_case['data'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{base_url}/api/v1/genprompt",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 400:
                print(f"✅ 正确返回400错误")
                try:
                    error_data = response.json()
                    print(f"错误信息: {error_data}")
                except:
                    print(f"响应内容: {response.text}")
            else:
                print(f"❌ 期望400错误，实际返回: {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("开始测试genprompt接口的文件格式支持...")
    print("请确保Python API服务器正在运行 (http://localhost:8000)")
    print()
    
    # 测试有效格式
    test_genprompt_formats()
    
    # 测试无效格式
    test_invalid_formats()
    
    print("\n测试完成！")
