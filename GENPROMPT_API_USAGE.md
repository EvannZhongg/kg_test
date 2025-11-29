# Prompt生成API使用文档

## 概述

`/api/v1/genprompt` API端点用于生成三元组抽取的Prompt。该API支持多种配置选项，允许用户根据不同的需求和领域生成定制化的Prompt。

## 端点信息

- **URL**: `/api/v1/genprompt`
- **方法**: `POST`
- **Content-Type**: `application/json`

## 请求参数

### 必需参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `schema_url` | string | 知识图谱schema的URL（JSON格式） |

### 可选参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `sample_text_url` | string | 样例原文的URL（支持.txt, .docx格式） |
| `sample_xlsx_url` | string | 样例三元组的URL（.xlsx格式） |
| `target_domain` | string | 目标领域描述，如"建筑学"、"医学"、"法律"等 |
| `dictionary_url` | string | 专业词典的URL（.txt格式） |
| `priority_extractions` | array | 抽取意向优先级列表（字符串数组） |
| `extraction_requirements` | string | 自定义抽取要求描述文本 |
| `base_instruction` | string | 自定义基础指导语（覆盖默认抽取指导） |

### priority_extractions 格式说明

`priority_extractions` 是一个字符串数组，列出需要特别关注的实体类型：

```json
["城市", "建筑师", "设计作品"]
```

## 请求示例

### 基础使用（仅必需参数）

```bash
curl -X POST http://localhost:5000/api/v1/genprompt \
  -H "Content-Type: application/json" \
  -d '{
    "schema_url": "http://example.com/schema.json"
  }'
```

### 完整配置示例

```bash
curl -X POST http://localhost:5000/api/v1/genprompt \
  -H "Content-Type: application/json" \
  -d '{
    "schema_url": "http://example.com/schema.json",
    "sample_text_url": "http://example.com/sample.docx",
    "sample_xlsx_url": "http://example.com/sample_triplets.xlsx",
    "target_domain": "建筑学领域知识图谱抽取",
    "dictionary_url": "http://example.com/architecture_dict.txt",
    "priority_extractions": ["城市", "建筑师", "设计作品"],
    "extraction_requirements": "请特别关注建筑风格和设计理念的描述，抽取时需要保留原文的专业术语"
  }'
```

### 使用自定义指导语

```bash
curl -X POST http://localhost:5000/api/v1/genprompt \
  -H "Content-Type: application/json" \
  -d '{
    "schema_url": "http://example.com/schema.json",
    "base_instruction": "你是一个医学知识图谱构建专家。请从输入的医学文献中抽取实体和关系，形成三元组。注意：所有医学术语需要保持原文准确性。",
    "target_domain": "医学文献知识抽取"
  }'
```

## 响应格式

### 成功响应 (200 OK)

```json
{
  "prompt": "生成的完整Prompt内容...",
  "schema_url": "http://example.com/schema.json",
  "sample_text_url": "http://example.com/sample.docx",
  "sample_xlsx_url": "http://example.com/sample_triplets.xlsx",
  "target_domain": "建筑学领域知识图谱抽取",
  "dictionary_url": "http://example.com/architecture_dict.xlsx",
  "priority_extractions": [...],
  "extraction_requirements": "...",
  "message": "Prompt生成成功"
}
```

### 错误响应

#### 400 Bad Request - 参数错误

```json
{
  "error": "Bad Request",
  "message": "缺少必需参数: schema_url"
}
```

#### 500 Internal Server Error - 服务器错误

```json
{
  "error": "Internal Server Error",
  "message": "生成Prompt失败: ..."
}
```

## Prompt生成逻辑

生成的Prompt按以下顺序包含各个部分：

1. **基础指导语** - 告诉AI它的角色和任务
2. **目标领域** - 如果提供，说明抽取的目标领域
3. **输出格式要求** - JSON格式规范
4. **输出示例** - 展示正确的输出格式
5. **抽取规则** - 详细的抽取规则说明
6. **抽取意向优先级** - 如果提供，列出需要特别关注的实体类型
7. **抽取要求描述** - 如果提供，额外的自定义要求
8. **实体类型** - 从schema中提取的所有实体类型及其细分
9. **关系类型** - 从schema中提取的所有关系类型及其细分
10. **三元组类型** - 从schema中提取的所有允许的三元组模式
11. **专业词典** - 如果提供，展示专业术语参考
12. **抽取样例** - 如果提供，展示样例原文和对应的抽取结果
13. **再次强调** - 强调输出格式要求

## 使用场景

### 场景1：基础抽取任务

只提供schema，使用默认配置：

```json
{
  "schema_url": "http://example.com/schema.json"
}
```

### 场景2：带样例的学习型抽取

提供schema和样例，让模型从样例中学习：

```json
{
  "schema_url": "http://example.com/schema.json",
  "sample_text_url": "http://example.com/sample.txt",
  "sample_xlsx_url": "http://example.com/sample_result.xlsx"
}
```

### 场景3：特定领域的定制化抽取

使用目标领域和专业词典：

```json
{
  "schema_url": "http://example.com/schema.json",
  "target_domain": "医疗病历信息抽取",
  "dictionary_url": "http://example.com/medical_terms.xlsx"
}
```

### 场景4：强调特定实体的抽取

使用抽取优先级：

```json
{
  "schema_url": "http://example.com/schema.json",
  "priority_extractions": ["疾病名称", "药物名称", "症状"]
}
```

### 场景5：完全自定义的抽取任务

使用自定义指导语和要求：

```json
{
  "schema_url": "http://example.com/schema.json",
  "base_instruction": "你是一个法律文书分析专家。请从法律判决书中抽取案件相关的实体和关系。",
  "target_domain": "法律判决书知识抽取",
  "extraction_requirements": "1. 保留所有法律条文的准确引用\n2. 区分原告和被告的行为\n3. 标注时间节点"
}
```

## 专业词典格式说明

专业词典文件（.txt）应该是一个纯文本文件，每行包含一个术语或术语说明。例如：

```
新古典主义建筑风格
柱廊建筑元素
飞扶壁结构元素
哥特式建筑风格
巴洛克建筑装饰
```

或者更详细的格式：

```
新古典主义 | 建筑风格 | 18世纪兴起的建筑风格
柱廊 | 建筑元素 | 由柱子支撑的走廊
飞扶壁 | 结构元素 | 哥特式建筑的支撑结构
```

词典内容会被直接嵌入到Prompt中供模型参考。

## 注意事项

1. 所有URL必须可访问，否则会返回错误
2. 文件格式必须正确：
   - schema: JSON格式
   - sample_text: .txt或.docx
   - sample_xlsx: .xlsx
   - dictionary: .txt
3. 生成的Prompt可能会很长，请确保后续使用的模型支持足够的上下文长度
4. `priority_extractions` 中的 `entity_type` 应该在schema中定义过，否则可能无法正确抽取
5. 如果同时提供了 `base_instruction` 和默认配置，将使用 `base_instruction` 覆盖默认指导语

## 更新日志

### 版本 2.0 (当前)

- 新增 `target_domain` 参数，支持目标领域描述
- 新增 `dictionary_url` 参数，支持专业词典
- 新增 `priority_extractions` 参数，支持抽取优先级配置
- 新增 `extraction_requirements` 参数，支持自定义抽取要求
- 新增 `base_instruction` 参数，支持自定义基础指导语
- 优化了Prompt的组织结构和可读性

### 版本 1.0

- 基础版本，支持 schema、sample_text 和 sample_xlsx 参数

