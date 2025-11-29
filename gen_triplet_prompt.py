import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd


def load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_types(schema: dict) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Collect entity types (with includes) and relationship types from schema."""
    entity_type_to_children: Dict[str, Set[str]] = defaultdict(set)
    relationship_types: Dict[str, Set[str]] = defaultdict(set)
    triplet_types: Set[str] = set()


    for triple in schema.get("triples", []):
        triplet_str=""

        relationship=triple.get("relationship", {})

        head = triple.get("head", {})
        tail = triple.get("tail", {})

        relationship_type = relationship.get("type", "").strip()
        head_type = head.get("type", "").strip()
        tail_type = tail.get("type", "").strip()

        triplet_str+=head_type+'->'+triple.get("relationship", "").get("type", "")+'->'+tail_type
        triplet_types.add(triplet_str)          
        
        
        # Merge includes for entity types
        if head_type:
            for child in head.get("includes", []) or []:
                if child:
                    entity_type_to_children[head_type].add(str(child).strip())
            # Ensure the type key exists even if no includes
            entity_type_to_children.setdefault(head_type, set())
        if tail_type:
            for child in tail.get("includes", []) or []:
                if child:
                    entity_type_to_children[tail_type].add(str(child).strip())
            entity_type_to_children.setdefault(tail_type, set())
        
        if relationship_type:
            for child in relationship.get("includes", []):
                if child:
                    relationship_types[relationship_type].add(str(child).strip())
            relationship_types.setdefault(relationship_type, set())

    # Remove empty relationship names if any
    # relationship_types = {r.get("type", ""):r.get("includes", []) for r in relationship_types}
    return entity_type_to_children, relationship_types,triplet_types

def collect_triplet_types(schema: dict) -> Tuple[Dict[str, Set[str]], List[str]]:
    entity_type_to_children: Dict[str, Set[str]] = defaultdict(set)
    relationship_types: Set[str] = set()


    for triple in schema.get("triples", []):
        relationship_types.add(triple.get("relationship", "").strip())
        head = triple.get("head", {})
        tail = triple.get("tail", {})


def build_prompt(entity_type_to_children: Dict[str, Set[str]],
                 relationship_types: Dict[str, Set[str]],
                 triplet_types: Set[str],
                 priority_extractions: List[str] = None) -> str:
    lines: List[str] = []
    lines.append("你是东南大学建筑学院的一个行政人员。输入的文本代表你院的一些实践活动记录。")
    lines.append("请扮演一个严谨的数据分析师，按照以下思维链步骤，从输入文本中抽取结构化的知识图谱三元组。")
    lines.append("")

    # === 1. 动态思维链 (Priority / Rounds) ===
    lines.append("=== 抽取思维链（Step-by-Step） ===")
    if priority_extractions and len(priority_extractions) > 0:
        for i, entity_type in enumerate(priority_extractions, 1):
            lines.append(f"Round {i}：扫描全文，重点寻找并抽取【{entity_type}】类型的实体，以及它作为头实体发出的所有关系。")
        next_round = len(priority_extractions) + 1
        lines.append(f"Round {next_round}：扫描剩余文本，补充抽取其他上述轮次未覆盖的实体和关系（如人物、时间、地点等）。")
        lines.append(f"Round {next_round + 1}：最后，针对核心实践活动，总结其成果及依托的专业技能。")
    else:
        # 默认逻辑
        lines.append("Round 1：通读全文，抽取唯一的核心【实践活动】实体。")
        lines.append("Round 2：围绕该实践活动，抽取其属性（如时间、地点、类型）。")
        lines.append("Round 3：抽取参与该活动的人物/团队，以及他们的具体行为。")
        lines.append("Round 4：总结活动的成果及其依托的专业技能。")
    lines.append("")

    # === 2. 扁平化数组输出格式 (Token Optimization) ===
    lines.append("=== 重要：输出格式要求 ===")
    lines.append("你必须严格按照以下JSON格式输出，使用紧凑的数组格式以节省空间：")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "activity": "核心实践活动名称",')
    lines.append('  "triples": [')
    lines.append('    // 格式：["头实体(标准名)", "头类型", "关系", "关系类型", "尾实体(标准名)", "尾类型"]')
    lines.append('    ["实体A", "类型A", "关系X", "类型X", "实体B", "类型B"],')
    lines.append('    ["实体C", "类型C", "关系Y", "类型Y", "实体D", "类型D"]')
    lines.append('  ]')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("=== 输出示例 ===")
    lines.append("```json")
    lines.append("{")
    lines.append('  "activity": "南师附小弹性离校活动",')
    lines.append('  "triples": [')
    lines.append('    ["南师附小弹性离校活动", "实践活动", "实践类型", "实践类型", "支教", "实践类型"],')
    lines.append('    ["志愿者团队", "团队", "实践", "实践", "南师附小弹性离校活动", "实践活动"]')
    lines.append('  ]')
    lines.append("}")
    lines.append("```")
    lines.append("")

    # === 3. 强归一化指令 (Normalization) ===
    lines.append("=== 关键规则：实体归一化 ===")
    lines.append(
        "1. **指代消解**：如果实体在文中存在简称、别名或代词（如'南师附小'、'附小'），**必须**统一使用最正式的全称（如'南京师范大学附属小学'）作为实体名称。")
    lines.append(
        "2. **词典优先**：如果下文提供了【专业词典】，且文中的词汇是词典中某标准词的变体，**必须**使用词典中的标准词。")
    lines.append("3. **拒绝碎片化**：不要将同一个对象的不同叫法抽取为两个实体。")
    lines.append("")

    # === Schema 注入 (保持原有逻辑) ===
    lines.append("=== 实体类型与细分（type -> includes） ===")
    for etype in sorted(entity_type_to_children.keys()):
        children = sorted([c for c in entity_type_to_children[etype] if c])
        if children:
            lines.append(f"- {etype}：{', '.join(children)}")
        else:
            lines.append(f"- {etype}：<无细分>")
    lines.append("")

    lines.append("=== 关系类型与细分（type -> includes） ===")
    if relationship_types:
        for rtype in sorted(relationship_types):
            children = sorted([c for c in relationship_types[rtype] if c])
            if children:
                lines.append(f"- {rtype}：{', '.join(children)}")
            else:
                lines.append(f"- {rtype}：<无细分>")
    else:
        lines.append("- <未在schema中定义>")
    lines.append("")

    lines.append("=== 合法的三元组结构（仅允许以下组合） ===")
    if triplet_types:
        # 简单的去重和排序
        sorted_triplets = sorted(list(triplet_types))
        for t in sorted_triplets:
            lines.append(f"- {t}")
    lines.append("")

    lines.append("=== 再次强调 ===")
    lines.append("1. 输出必须是合法的JSON，不要包含Markdown代码块标记以外的文字。")
    lines.append("2. 严格遵守归一化规则，不要输出同义异名的冗余实体。")
    lines.append("")

    # lines.append("抽取样例：")
    # lines.append("原文：")
    # docx_path=Path(r"D:\Personal_Project\kgplatform_backend\python-service\txt_test\【回顾】南师附小弹性离校 _ 放学别走！一起 “识天文知地理”！.txt")
    # docx_text=read_txt_text(docx_path)
    # lines.append(docx_text)
    # lines.append("")
    # lines.append("抽取结果：")
    # xlsx_path=Path(r"D:\Personal_Project\kgplatform_backend\python-service\txt_test\【回顾】南师附小弹性离校 _ 放学别走！一起 “识天文知地理”！_extractions.xlsx")
    # xlsx_text=read_xlsx_text(xlsx_path)
    # lines.append(xlsx_text)
    # lines.append("")



    return "\n".join(lines)
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

def read_xlsx_text(xlsx_path: Path) -> str:
    df = pd.read_excel(xlsx_path)
    # 删除空行
    df = df.dropna(how='all')
    return df.to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description="根据知识图谱schema生成三元组抽取Prompt")
    parser.add_argument("--schema", type=Path, default=Path("knowledge_graph_schema.json"), help="schema JSON 文件路径")
    parser.add_argument("--out", type=Path, default=Path("output/triple_extraction_prompt.txt"), help="输出的prompt文件路径")
    parser.add_argument("--priority", nargs='+', help="测试用的优先级列表，如：实践活动 人物")

    args = parser.parse_args()
    schema = load_schema(args.schema)
    entity_type_to_children, relationship_types, triplet_types = collect_types(schema)

    # 传入优先级
    prompt_text = build_prompt(
        entity_type_to_children,
        relationship_types,
        triplet_types,
        priority_extractions=args.priority  # 传入参数
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(prompt_text, encoding="utf-8")
    print(prompt_text)


if __name__ == "__main__":
    main() 