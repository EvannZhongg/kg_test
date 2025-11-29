

其中运行`test_open_ie_api.py`可以测试开放图谱
注意每次测试修改项目隔离ID：
```angular2html
{
    "task_type": "open_ie",          // 必填：指定模式
    "project_id": 101,               // 必填：项目隔离ID
    "provider": "deepseek",
    "api_key": "sk-...",
    "files": [
        {
            "material_id": 2001,     // 必填：关联的文件ID
            "url": "/abs/path/to/file.txt"
        }
    ]
}
```

## 接口协议变更 (API Interface)

前端（Go 服务）调用 Python 服务 `/api/v1/tasks` 接口时，针对 OpenIE 任务需要传递特定的参数。

### API 接口使用说明
创建任务接口,该接口通过 task_type 参数区分两种模式。
  * **Endpoint**: `POST /api/v1/tasks`
  * **变更点**：
    1.  新增 `task_type`: 必须指定为 `"open_ie"`。
    2.  新增 `project_id`: 必须提供有效的项目 ID（用于数据隔离）。
    3.  `files` 结构要求：必须包含 `material_id`。
    4.  `prompt_text`: 在 OpenIE 模式下为可选（后端使用内置 Prompt）。

**开放抽取 请求示例 Payload**：

```json
{
    "task_type": "open_ie",
    "project_id": 105,
    "provider": "deepseek",
    "api_key": "sk-xxxxxx",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "files": [
        {
            "material_id": 2001,
            "url": "/abs/path/to/file.txt"
        }
    ]
}
```

**限定抽取 请求示例 Payload**：

```json
{
    "task_type": "schema",           // 默认为 schema
    "prompt_text": "请抽取...",       // 必填：包含 Schema 约束的 Prompt
    "provider": "deepseek",
    "api_key": "sk-...",
    "files": [
        "/abs/path/to/file.txt"      // 兼容旧格式字符串列表
    ]
}
```

后端采用 `open_ie_service.py` 独立处理该流程，逻辑如下：
```
1.  预处理：读取文本，计算文件哈希，进行文本分块 (`chunk_text`)。
2.  Chunk 向量化*：调用 `embedding_service` 对分块进行批量向量化，并存入 `open_graph_chunks`。
3.  LLM 抽取：使用 OpenIE Prompt 调用 LLM，提取 `Entity` 和 `Relation`（LightRAG 格式 `<|#|>` 分隔）。
4.  语义级融合去重： 
* **查询旧值**：根据 ID 查询数据库中已存在的实体/关系描述。 
* **LLM 摘要**：如果存在旧描述，调用 LLM 将 **"旧描述 + 新抽取描述"** 融合成一段更全面、连贯的新描述。
* 基于融合后的新描述，重新计算实体和关系的向量。
6.  增量入库：
* **更新描述与向量**：写入新的描述和向量。 
* **更新权重**：`weight` + 1。 
* **更新度**：如果是新建立的关系，两端节点 `degree` + 1。 
* **追加溯源**：将当前 `chunk_id` 追加到 `source_chunk_ids` 数组。
```


### 数据库必须包含vector扩展
```angular2html
docker exec -it pgvector-db psql -U postgres -c "CREATE DATABASE kg;"
cd ...你的路径\kgplatform_backend\python-service
```

在 Python 服务目录 \kgplatform_backend\python-service 下，创建一个名为 init_open_graph.sql 的新文件。

```使用CMD
type init_open_graph.sql | docker exec -i pgvector-db psql -U postgres -d kg
```

```使用PowerShell
Get-Content init_open_graph.sql | docker exec -i pgvector-db psql -U postgres -d kg
```

.env增加数据库配置
```angular2html
# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=kg
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

# (可选) 连接池配置
POSTGRES_MIN_SIZE=1
POSTGRES_MAX_SIZE=10
```

检查是否创建成功：
```angular2html
docker exec -it pgvector-db psql -U postgres -d kg -c "\dt"
```


### **待办事项 (Go前端侧)**：

1.  修改 Go 代码中的 `CreateTask` 调用逻辑，确保传递 `task_type="open_ie"` 和 `project_id`。
2.  开发新的查询接口，直接通过 SQL 查询 PostgreSQL 中的图数据（不再依赖 JSONL 文件解析）。