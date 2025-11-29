import os
import sys
from pathlib import Path

# 尝试导入 chunk_text 函数
try:
    from extract_triplets_from_docx import chunk_text
except ImportError:
    print("错误: 无法导入 'extract_triplets_from_docx'。请确保此脚本与该文件在同一目录下。")
    sys.exit(1)


def visualize_chunks(file_path, max_chars=800, overlap=200):
    """
    读取文件并可视化分块结果
    """
    txt_path = Path(file_path)

    if not txt_path.exists():
        print(f"错误: 文件不存在 -> {txt_path}")
        return

    print(f"\n{'=' * 20} 开始测试分块 {'=' * 20}")
    print(f"文件路径: {txt_path}")
    print(f"参数设置: max_chars={max_chars}, overlap={overlap}")

    # 1. 读取文件
    try:
        # 尝试读取，兼容不同编码
        content = ""
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-16']:
            try:
                content = txt_path.read_text(encoding=encoding).strip()
                break
            except UnicodeDecodeError:
                continue
        if not content:
            content = txt_path.read_text(encoding='utf-8', errors='ignore').strip()

        print(f"文件总长度: {len(content)} 字符")
        print(f"{'-' * 50}\n")

    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 执行分块
    # 注意：确保你的 chunk_text 函数签名支持 overlap 参数
    # 如果不支持，请移除 overlap 参数调用: chunks = chunk_text(content, max_chars)
    try:
        chunks = chunk_text(content, max_chars=max_chars, overlap=overlap)
    except TypeError:
        print("警告: 当前 chunk_text 可能不支持 overlap 参数，尝试仅使用 max_chars...")
        chunks = chunk_text(content, max_chars=max_chars)

    # 3. 可视化展示
    print(f"共切分为 {len(chunks)} 个块 (Chunk):\n")

    for i, chunk in enumerate(chunks):
        print(f"┌{'─' * 15} Chunk {i + 1} / {len(chunks)} (长度: {len(chunk)}) {'─' * 15}┐")

        # 打印内容，为了不刷屏，过长内容可以截断展示，或者全部展示
        # 这里全部展示以便观察完整性
        print(chunk)

        print(f"└{'─' * 55}┘")

        # 如果有下一个块，尝试展示重叠部分（仅供视觉参考）
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            # 简单的后缀匹配查找重叠（仅用于视觉验证逻辑是否生效）
            overlap_len = 0
            for j in range(min(len(chunk), len(next_chunk)), 0, -1):
                if chunk.endswith(next_chunk[:j]):
                    overlap_len = j
                    break

            if overlap_len > 0:
                print(f"   ↓ 重叠部分 (Overlap): {overlap_len} 字符 ↓")
                preview = next_chunk[:overlap_len].replace('\n', '\\n')
                if len(preview) > 50:
                    preview = preview[:20] + " ... " + preview[-20:]
                print(f"   \"{preview}\"\n")
            else:
                print("   ↓ 无重叠 (No Overlap detected) ↓\n")


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 在这里硬编码你的测试文件路径
    # 建议使用绝对路径，或者相对于当前脚本的路径
    # Windows示例: r"E:\Data\test_file.txt"
    TEST_FILE_PATH = r"D:\Personal_Project\kgplatform_backend\python-service\txt_test\三国演义.txt"

    # 分块参数
    MAX_CHARS = 1000
    OVERLAP = 200

    # ===========================================

    # 如果想临时创建一个测试文件（如果没有现成的）
    if not os.path.exists(TEST_FILE_PATH):
        print("未找到指定文件，正在生成临时测试文件...")
        TEST_FILE_PATH = "temp_test_chunking.txt"
        with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
            # 生成一些模拟段落
            for i in range(1, 20):
                f.write(f"这是第 {i} 个自然段。它包含了一些用于测试的文本内容。" * 5 + "\n\n")

    visualize_chunks(TEST_FILE_PATH, MAX_CHARS, OVERLAP)