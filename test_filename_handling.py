#!/usr/bin/env python3
"""
测试文件名处理功能
"""
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import uuid

def test_filename_handling():
    """测试文件名处理逻辑"""
    
    # 模拟问题中的URL
    test_urls = [
        "http://124.222.5.132:9000/uploads/%E3%80%90%E5%9B%9E%E9%A1BEE38091%E5%8D%97%E5%B8%88%E9%99%84%E5%B0%8F%E5%BC%B9%E6%80%A7%E7%A6%BB%E6%A0%A120_20%E6%94%BE%E5%AD%A6%E5%88%AB%E8%B5%B0EFBC81%E4%B8%80%E8%B5%B720%E2809CE8AF86%E5%A4%A9%E6%96%87%E7%9F%A5%E5%9C%B0%E7%90%86%E2809DEFBC81_extractions_20251004113908_63dd6490.xlsx",
        "https://example.com/sample.txt",
        "https://example.com/测试文档.docx",
        "https://example.com/test file with spaces.pdf"
    ]
    
    print("=== 文件名处理测试 ===\n")
    
    for i, url in enumerate(test_urls, 1):
        print(f"测试 {i}: {url}")
        
        # 解析URL
        parsed_url = urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        file_ext = Path(parsed_url.path).suffix.lower()
        
        print(f"  原始文件名: {original_filename}")
        print(f"  文件扩展名: {file_ext}")
        
        # URL解码
        import urllib.parse
        decoded_filename = urllib.parse.unquote(original_filename)
        print(f"  解码后文件名: {decoded_filename}")
        
        # 生成安全文件名
        safe_filename = f"sample_{uuid.uuid4().hex[:8]}{file_ext}"
        print(f"  安全文件名: {safe_filename}")
        
        # 测试文件路径创建
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / safe_filename
            
            try:
                # 创建测试文件
                file_path.write_text("test content")
                print(f"  ✅ 文件创建成功: {file_path}")
                
                # 验证文件存在
                if file_path.exists():
                    print(f"  ✅ 文件验证成功")
                else:
                    print(f"  ❌ 文件验证失败")
                    
            except Exception as e:
                print(f"  ❌ 文件创建失败: {str(e)}")
        
        print("-" * 50)

def test_actual_download():
    """测试实际下载功能"""
    print("\n=== 实际下载测试 ===\n")
    
    # 这里可以添加实际的URL测试
    # 注意：需要确保URL是可访问的
    test_url = "https://httpbin.org/json"  # 测试URL
    
    print(f"测试URL: {test_url}")
    
    try:
        import requests
        
        # 模拟下载请求
        response = requests.get(test_url, timeout=10)
        print(f"响应状态: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 下载测试成功")
        else:
            print("❌ 下载测试失败")
            
    except Exception as e:
        print(f"❌ 下载测试异常: {str(e)}")

if __name__ == "__main__":
    print("开始文件名处理测试...")
    print()
    
    # 测试文件名处理
    test_filename_handling()
    
    # 测试实际下载
    test_actual_download()
    
    print("\n测试完成！")
