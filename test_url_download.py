#!/usr/bin/env python3
"""
测试URL下载功能的脚本
"""
import sys
from pathlib import Path
import tempfile
import os

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from api_server import APIServer

def test_url_download():
    """测试URL下载功能"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建API服务器实例
        server = APIServer()
        
        # 测试URL（您提供的URL）
        test_url = "http://124.222.5.132:9000/uploads/ocr_text_20251002021321_85dafdb2.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20251001%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251001T182314Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=72556a44354a3f10f169bb47521e56aa3234e9ea324cfa02de23fdf8ba9fb333"
        
        try:
            print(f"测试URL: {test_url}")
            print("开始下载...")
            
            # 测试下载
            downloaded_file = server._download_file_from_url(test_url, temp_path)
            
            print(f"✅ 下载成功！")
            print(f"文件路径: {downloaded_file}")
            
            # 检查文件内容
            with open(downloaded_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"文件大小: {len(content)} 字符")
                print(f"文件内容预览: {content[:200]}...")
                
        except Exception as e:
            print(f"❌ 下载失败: {str(e)}")
            return False
            
    return True

if __name__ == "__main__":
    print("开始测试URL下载功能...")
    success = test_url_download()
    if success:
        print("🎉 测试通过！")
    else:
        print("💥 测试失败！")
        sys.exit(1)
