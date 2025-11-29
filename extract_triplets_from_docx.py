import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

CHINESE_TRIPLE_REGEX = re.compile(r"head：(\{.*?\})；relationship：(\{.*?\})；tail：(\{.*?\})")


class ProviderConfig:
    def __init__(self, name: str, base_url: str, api_key_env: str, default_model: str):
        self.name = name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.default_model = default_model


# 模型提供商配置
PROVIDERS: Dict[str, ProviderConfig] = {
    # DeepSeek OpenAI-Compatible
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    # Qwen DashScope OpenAI-Compatible
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen2.5-72b-instruct",
    ),
    # Generic OpenAI-compatible forward proxy
    "forward": ProviderConfig(
        name="forward",
        base_url=os.environ.get("FORWARD_BASE_URL", "http://localhost:3000/v1"),
        api_key_env="FORWARD_API_KEY",
        default_model=os.environ.get("FORWARD_DEFAULT_MODEL", "gpt-4o-mini"),
    ),
}


def read_txt_text(txt_path: Path) -> str:
    """读取文本文件内容"""
    try:
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']

        for encoding in encodings:
            try:
                return txt_path.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue

        # 如果所有编码都失败，使用utf-8并忽略错误
        return txt_path.read_text(encoding='utf-8', errors='ignore').strip()

    except Exception as e:
        raise ValueError(f"无法读取文本文件 {txt_path}: {str(e)}")


def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """
    按自然段切分文本，支持Overlap，并尽量保持段落和句子的完整性。
    """
    text = text.strip()
    if not text:
        return []

    # === 1. 预处理：生成原子片段列表 (segments) ===

    # 识别自然段：按空行（连续换行）分隔
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    raw_paragraphs = re.split(r'\n\s*\n+', normalized)

    # 句子结束标点（用于细分超长段落）
    sentence_endings = ['。', '！', '？', '；', '…', '.', '!', '?', ';']

    def split_long_paragraph_to_segments(para: str) -> List[str]:
        """将超长段落拆解为句子片段列表"""
        if len(para) <= max_chars:
            return [para]

        segments: List[str] = []
        start = 0
        while start < len(para):
            # 期望的结束位置（硬切分点）
            end = min(start + max_chars, len(para))

            if end >= len(para):
                segments.append(para[start:].strip())
                break

            # 在 max_chars 范围内寻找最佳句子边界（从后往前找）
            split_pos = -1
            # 1. 优先找明确的句末标点
            for i in range(end - 1, max(start, end - max_chars // 2) - 1, -1):
                if para[i] in sentence_endings:
                    split_pos = i + 1
                    break

            # 2. 其次找逗号、空格等软边界
            if split_pos == -1:
                for i in range(end - 1, max(start, end - max_chars // 2) - 1, -1):
                    if para[i] in [' ', '\n', '\t', '，', '、', ',', ':', '：']:
                        split_pos = i + 1
                        break

            # 3. 实在找不到，只能硬切
            if split_pos == -1:
                split_pos = end

            segment = para[start:split_pos].strip()
            if segment:
                segments.append(segment)

            start = split_pos
            # 跳过切分点后的空白字符
            while start < len(para) and para[start] in [' ', '\n', '\t']:
                start += 1

        return segments

    all_segments: List[str] = []
    for para in raw_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue

        if len(cleaned) <= max_chars:
            all_segments.append(cleaned)
        else:
            # 递归拆解长段落
            sub_segments = split_long_paragraph_to_segments(cleaned)
            all_segments.extend(sub_segments)

    if not all_segments:
        return []

    # === 2. 滑动窗口：生成 Chunk (带 Overlap) ===

    chunks: List[str] = []
    start_idx = 0

    while start_idx < len(all_segments):
        current_chunk_len = 0
        end_idx = start_idx

        # 贪心向后填充，直到超过 max_chars
        while end_idx < len(all_segments):
            seg_len = len(all_segments[end_idx])

            # 如果加入当前片段会超出最大长度，且当前chunk不为空，则停止
            if current_chunk_len + seg_len > max_chars and current_chunk_len > 0:
                break

            current_chunk_len += seg_len
            # 加上换行符的长度预估（连接时会用换行）
            if end_idx > start_idx:
                current_chunk_len += 1

            end_idx += 1

        # 这里的 end_idx 是开区间，不包含
        chunk_content = "\n".join(all_segments[start_idx:end_idx])
        chunks.append(chunk_content)

        # 如果已经处理完所有片段，结束
        if end_idx >= len(all_segments):
            break

        # 计算下一个 start_idx (回溯逻辑)
        # 我们需要从 end_idx 往前倒推，找到一个位置，使得重叠部分的长度 >= overlap
        overlap_len = 0
        next_start_idx = end_idx

        while next_start_idx > start_idx:
            # 尝试将前一个片段纳入重叠区
            prev_seg_len = len(all_segments[next_start_idx - 1])
            if overlap_len + prev_seg_len > overlap:
                # 如果加上这个片段已经大大超过overlap（比如这是一个长段落），
                # 我们需要权衡：是包含它导致overlap很大，还是不包含它导致overlap很小？
                # 题目要求：保持段落/句子完整性。因此我们通常倾向于包含完整的片段，
                # 哪怕这会导致 overlap > 设定值 (Better context preservation).
                # 除非 overlap 还是 0，否则我们尽量贴近设定值。

                # 如果当前累积 overlap 还是 0，那必须包含这个片段，否则就没有 overlap
                if overlap_len == 0:
                    overlap_len += prev_seg_len
                    next_start_idx -= 1
                break

            overlap_len += prev_seg_len
            next_start_idx -= 1

        # 死循环保护：如果逻辑导致 next_start_idx 没有前进（还是 start_idx），强制前进一步
        if next_start_idx == start_idx:
            next_start_idx += 1

        start_idx = next_start_idx

    return chunks


def make_client(provider: str, model: Optional[str], base_url: Optional[str], api_key: Optional[str]):
    cfg = PROVIDERS[provider]
    resolved_base = base_url or cfg.base_url
    resolved_api_key = api_key or os.environ.get(cfg.api_key_env)
    if not resolved_api_key:
        raise RuntimeError(f"缺少API Key，请设置环境变量 {cfg.api_key_env} 或通过 --api-key 传入。")
    resolved_model = model or cfg.default_model
    return resolved_base, resolved_api_key, resolved_model


def call_llm(base_url: str, api_key: str, model: str, prompt_text: str, chunk: str, temperature: float = 0.0,
             top_p: float = 1.0) -> str:
    messages = [
        {"role": "system", "content": "你是一个严谨的中文信息抽取助手。"},
        {"role": "user", "content": f"{prompt_text}\n\n以下为需要抽取的原文：\n\n{chunk}"},
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def parse_triples_from_text(text: str) -> List[Dict[str, Any]]:
    triples: List[Dict[str, Any]] = []

    # 1. 尝试解析 JSON 格式
    try:
        # 查找 JSON 代码块
        json_start = text.find('```json')
        if json_start != -1:
            json_end = text.find('```', json_start + 7)
            if json_end != -1:
                json_content = text[json_start + 7:json_end].strip()
                # 处理可能的尾部逗号等非标准JSON情况
                try:
                    data = json.loads(json_content)
                except json.JSONDecodeError:
                    # 简单的容错尝试：如果是因为尾部逗号
                    try:
                        import re
                        fixed_content = re.sub(r',\s*}', '}', json_content)
                        fixed_content = re.sub(r',\s*]', ']', fixed_content)
                        data = json.loads(fixed_content)
                    except:
                        raise

                # === 新逻辑：处理紧凑数组格式 (triples: [[...]]) ===
                # 格式定义: ["头实体", "头类型", "关系", "关系类型", "尾实体", "尾类型"]
                if "triples" in data and isinstance(data["triples"], list):
                    for item in data["triples"]:
                        if isinstance(item, list) and len(item) >= 6:
                            # 自动还原为系统通用的字典格式
                            triples.append({
                                "head": {"label": str(item[0]), "type": str(item[1])},
                                "relationship": {"label": str(item[2]), "type": str(item[3])},
                                "tail": {"label": str(item[4]), "type": str(item[5])}
                            })
                    return triples

                # === 兼容旧逻辑：处理嵌套字典格式 (三元组列表: [{...}]) ===
                if "三元组列表" in data:
                    for triple in data["三元组列表"]:
                        if all(key in triple for key in ["head", "relationship", "tail"]):
                            triples.append(triple)
                    return triples

    except Exception as e:
        print(f"JSON解析警告: {str(e)}")
        # JSON失败后继续尝试正则兜底
        pass

    # 2. 正则兜底（保持不变，处理非JSON的文本输出）
    for match in CHINESE_TRIPLE_REGEX.finditer(text.replace("\r", "")):
        head_str, rel_str, tail_str = match.groups()
        try:
            # 尝试清理中文引号等常见问题
            def clean_json_str(s):
                return s.replace("'", '"').replace("，", ",")

            head_obj = json.loads(clean_json_str(head_str))
            rel_obj = json.loads(clean_json_str(rel_str))
            tail_obj = json.loads(clean_json_str(tail_str))
            triples.append({"head": head_obj, "relationship": rel_obj, "tail": tail_obj})
        except Exception:
            continue

    return triples


def parse_lightrag_output(text: str) -> tuple[List[dict], List[dict]]:
    """
    解析 LightRAG 风格的输出 (自定义分隔符 <|#|>)
    格式变更：不包含权重字段
    返回: (nodes_list, edges_list)
    """
    nodes = []
    edges = []

    # 定义分隔符
    TUPLE_DELIMITER = "<|#|>"

    # 按行分割
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(TUPLE_DELIMITER)

        # 解析实体: entity<|#|>实体名称<|#|>实体类型<|#|>实体描述
        if line.startswith('entity') and len(parts) >= 4:
            nodes.append({
                "label": parts[1].strip(),
                "type": parts[2].strip(),
                "description": parts[3].strip()
            })

        # 解析关系: relation<|#|>源实体<|#|>目标实体<|#|>关系关键词<|#|>关系描述
        # 移除权重解析，权重将在后续由出现次数累加
        elif line.startswith('relation') and len(parts) >= 5:
            edges.append({
                "source": parts[1].strip(),
                "target": parts[2].strip(),
                "relation": parts[3].strip(),  # 关系关键词
                "description": parts[4].strip(),
                # "weight": 1.0  <-- 不再从LLM解析，后续逻辑统一处理
            })

    return nodes, edges

def save_outputs(raw_outputs: List[str], parsed: List[Dict[str, Any]], chunks: List[str],
                 chunk_indices: List[int], out_txt: Path, out_jsonl: Path, chunks_txt: Path) -> None:
    """保存输出结果，包含三元组的chunk索引信息"""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    chunks_txt.parent.mkdir(parents=True, exist_ok=True)

    # 保存原始输出
    out_txt.write_text("\n\n".join(raw_outputs), encoding="utf-8")

    # 为每个三元组添加chunk溯源信息
    enriched_parsed = []
    for i, triple in enumerate(parsed):
        enriched_triple = triple.copy()
        # 添加chunk索引信息
        if i < len(chunk_indices):
            chunk_idx = chunk_indices[i]
            enriched_triple["_chunk_index"] = chunk_idx
            # 确保索引有效
            if 0 <= chunk_idx < len(chunks):
                enriched_triple["_source_text"] = chunks[chunk_idx]
            else:
                enriched_triple["_source_text"] = f"[索引 {chunk_idx} 超出范围，共 {len(chunks)} 个chunks]"
        enriched_parsed.append(enriched_triple)

    with out_jsonl.open("w", encoding="utf-8") as f:
        json.dump(enriched_parsed, f, ensure_ascii=False, indent=2)

    # 保存分段文本
    chunk_content = []
    for i, chunk in enumerate(chunks, 1):
        chunk_content.append(f"=== 分段 {i} ===")
        chunk_content.append(f"长度: {len(chunk)} 字符")
        chunk_content.append(f"内容:\n{chunk}")
        chunk_content.append("")  # 空行分隔

    chunks_txt.write_text("\n".join(chunk_content), encoding="utf-8")


def process_single_txt(txt_path: Path, prompt_text: str, base_url: str, api_key: str, model: str,
                       output_dir: Path, file_index: int, total_files: int) -> Dict[str, Any]:
    """处理单个txt文件"""
    print(f"[{file_index}/{total_files}] 处理文件: {txt_path.name}")

    # 生成输出文件名（基于原文件名）
    stem = txt_path.stem
    out_txt = output_dir / "txt" / f"{stem}_extractions.txt"
    out_jsonl = output_dir / "jsonl" / f"{stem}_extractions.jsonl"
    chunks_txt = output_dir / "txt" / f"{stem}_chunks.txt"

    try:
        text = read_txt_text(txt_path)
        # 调用更新后的 chunk_text，使用默认 overlap=200
        chunks = chunk_text(text)
        raw_outputs: List[str] = []
        parsed_triples: List[Dict[str, Any]] = []
        chunk_indices: List[int] = []  # 记录每个三元组来自哪个chunk

        for idx, chunk in enumerate(chunks, start=1):
            print(f"  处理分段 {idx}/{len(chunks)}，长度 {len(chunk)} 字符...")
            content = call_llm(base_url, api_key, model, prompt_text, chunk)
            raw_outputs.append(content)
            triples_from_chunk = parse_triples_from_text(content)
            parsed_triples.extend(triples_from_chunk)
            # 为每个三元组记录它来自哪个chunk（索引从0开始）
            chunk_indices.extend([idx - 1] * len(triples_from_chunk))

        save_outputs(raw_outputs, parsed_triples, chunks, chunk_indices, out_txt, out_jsonl, chunks_txt)

        print(f"  ✓ 完成。输出文件：{out_txt.name}, {out_jsonl.name}, {chunks_txt.name}")

        return {
            "file": txt_path.name,
            "status": "success",
            "chunks_processed": len(chunks),
            "triples_extracted": len(parsed_triples),
            "output_files": [str(out_txt), str(out_jsonl), str(chunks_txt)]
        }

    except Exception as e:
        print(f"  ✗ 处理失败: {str(e)}")
        return {
            "file": txt_path.name,
            "status": "error",
            "error": str(e)
        }


def process_folder(input_dir: Path, prompt_text: str, base_url: str, api_key: str, model: str,
                   output_dir: Path) -> List[Dict[str, Any]]:
    """批量处理文件夹中的所有txt文件"""
    # 查找所有txt文件
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"在目录 {input_dir} 中未找到任何.txt文件")
        return []

    print(f"找到 {len(txt_files)} 个txt文件，开始批量处理...")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, txt_file in enumerate(txt_files, 1):

        # 如果目标目录下已有对应文件名的txt，则跳过
        if (output_dir / "txt" / f"{txt_file.stem}_extractions.txt").exists():
            print(f"目标目录下 {txt_file.stem}_extractions.txt 已存在，跳过处理")
            continue

        result = process_single_txt(txt_file, prompt_text, base_url, api_key, model, output_dir, i, len(txt_files))
        results.append(result)

    # 生成汇总报告
    summary_file = output_dir / "batch_processing_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n批量处理完成！汇总报告已保存到: {summary_file}")

    # 打印统计信息
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    total_triples = sum(r.get("triples_extracted", 0) for r in successful)

    print(f"成功处理: {len(successful)} 个文件")
    print(f"处理失败: {len(failed)} 个文件")
    print(f"总共抽取三元组: {total_triples} 个")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="从TXT读取文本，使用Prompt调用大模型进行三元组抽取")

    # 输入参数组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--txt", type=Path, help="输入的单个txt文件路径")
    input_group.add_argument("--folder", type=Path, help="输入的txt文件所在文件夹路径")

    parser.add_argument("--prompt", type=Path, default=Path("triple_extraction_prompt.txt"),
                        help="生成好的Prompt文件路径")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="deepseek",
                        help="模型提供方：deepseek/qwen/forward")
    parser.add_argument("--model", type=str, default=None, help="模型名称，留空则使用该provider的默认模型")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI兼容Base URL，留空则使用provider默认")
    parser.add_argument("--api-key", type=str, default=None, help="API Key，留空则读取各provider约定的环境变量")

    # 输出参数组
    parser.add_argument("--out", type=Path, default=Path("output/extractions.txt"),
                        help="保存单个文件原始输出的txt路径")
    parser.add_argument("--out-jsonl", type=Path, default=Path("output/extractions.jsonl"),
                        help="保存单个文件解析后三元组的jsonl路径")
    parser.add_argument("--chunks-txt", type=Path, default=Path("output/chunks.txt"),
                        help="保存单个文件分段文本的txt路径")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="批量处理时的输出目录")

    args = parser.parse_args()

    # 先加载 .env，以便后续读取环境变量
    load_dotenv(dotenv_path='.env')

    prompt_text = load_prompt(args.prompt)
    base_url, api_key, model = make_client(args.provider, args.model, args.base_url, args.api_key)

    if args.txt:
        # 处理单个文件
        text = read_txt_text(args.txt)
        chunks = chunk_text(text)
        raw_outputs: List[str] = []
        parsed_triples: List[Dict[str, Any]] = []
        chunk_indices: List[int] = []  # 记录每个三元组来自哪个chunk

        for idx, chunk in enumerate(chunks, start=1):
            print(f"处理分段 {idx}/{len(chunks)}，长度 {len(chunk)} 字符...")
            content = call_llm(base_url, api_key, model, prompt_text, chunk)
            raw_outputs.append(content)
            triples_from_chunk = parse_triples_from_text(content)
            parsed_triples.extend(triples_from_chunk)
            # 为每个三元组记录它来自哪个chunk（索引从0开始）
            chunk_indices.extend([idx - 1] * len(triples_from_chunk))

        save_outputs(raw_outputs, parsed_triples, chunks, chunk_indices, args.out, args.out_jsonl, args.chunks_txt)
        print(f"已完成。原始输出写入：{args.out}；解析后的JSONL写入：{args.out_jsonl}；分段文本写入：{args.chunks_txt}")

    elif args.folder:
        # 批量处理文件夹
        process_folder(args.folder, prompt_text, base_url, api_key, model, args.output_dir)


if __name__ == "__main__":
    main()