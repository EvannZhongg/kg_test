# GenPrompt API 修改说明

## 修改概述

本次修改主要解决两个问题：
1. 多文件抽取时返回结果携带 `material_id`，以便Go服务端识别结果对应的材料
2. 返回的输出文件路径改为绝对路径

## 修改的文件

### 1. `triplet_extraction_service.py`

#### 修改内容：

1. **TaskFile 数据类添加 material_id 字段**
   ```python
   @dataclass
   class TaskFile:
       file_name: str
       status: str
       material_id: int = None  # 新增字段
       triples_count: int = 0
       output_files: Dict[str, str] = None
       error: str = None
   ```

2. **create_task 方法支持接收包含 material_id 的文件信息**
   - 参数类型从 `Union[str, List[str]]` 改为 `Union[str, List[str], List[Dict[str, Any]]]`
   - 支持三种格式：
     - 单个文件路径字符串
     - 文件路径字符串列表（兼容旧格式）
     - 包含material_id和url的字典列表（新格式）
   ```python
   # 新格式示例
   files = [
       {"material_id": 1, "url": "/path/to/file1.txt"},
       {"material_id": 2, "url": "/path/to/file2.txt"}
   ]
   ```

3. **_process_task 方法保存并传递 material_id**
   - 参数类型从 `List[str]` 改为 `List[Dict[str, Any]]`
   - 在处理每个文件时提取并保存 `material_id`
   - 创建 TaskFile 时携带 `material_id`

4. **_process_single_file 方法返回绝对路径**
   - 使用 `Path.absolute()` 确保返回绝对路径
   ```python
   'output_files': {
       'txt': str(out_txt.absolute()),
       'jsonl': str(out_jsonl.absolute()),
       'chunks': str(chunks_txt.absolute())
   }
   ```

### 2. `api_server.py`

#### 修改内容：

1. **create_task 路由支持新旧两种请求格式**

   **旧格式（仍然支持）：**
   ```json
   {
       "files": ["file1.txt", "file2.txt"],
       "prompt_text": "...",
       "provider": "deepseek",
       "api_key": "..."
   }
   ```

   **新格式（推荐）：**
   ```json
   {
       "files": [
           {"material_id": 1, "url": "file1.txt"},
           {"material_id": 2, "url": "file2.txt"}
       ],
       "prompt_text": "...",
       "provider": "deepseek",
       "api_key": "..."
   }
   ```

2. **自动检测请求格式**
   - 检查 files 数组中的元素类型
   - 如果全为字符串，使用旧格式逻辑
   - 如果全为对象，使用新格式逻辑
   - 混合格式会报错

3. **处理所有三种文件来源（文件名/绝对路径/URL）时都保留 material_id**
   - 文件名格式：从上传目录中查找文件
   - 绝对路径格式：直接使用文件路径
   - URL格式：下载文件到上传目录
   - 所有情况下都保留并传递 `material_id`

## API响应格式变化

### 任务状态响应

**旧格式：**
```json
{
    "task_id": "xxx",
    "status": "completed",
    "results": [
        {
            "file_name": "file1.txt",
            "status": "success",
            "triples_count": 10,
            "output_files": {
                "txt": "output/xxx/txt/file1_extractions.txt",
                "jsonl": "output/xxx/jsonl/file1_extractions.jsonl",
                "chunks": "output/xxx/txt/file1_chunks.txt"
            }
        }
    ]
}
```

**新格式：**
```json
{
    "task_id": "xxx",
    "status": "completed",
    "results": [
        {
            "file_name": "file1.txt",
            "material_id": 1,
            "status": "success",
            "triples_count": 10,
            "output_files": {
                "txt": "E:/path/to/output/xxx/txt/file1_extractions.txt",
                "jsonl": "E:/path/to/output/xxx/jsonl/file1_extractions.jsonl",
                "chunks": "E:/path/to/output/xxx/txt/file1_chunks.txt"
            }
        }
    ]
}
```

主要变化：
1. 新增 `material_id` 字段
2. `output_files` 中的路径从相对路径改为绝对路径

## 兼容性

- **向后兼容**：旧格式（files为字符串数组）仍然支持，此时 `material_id` 为 `null`
- **推荐使用新格式**：传递 `material_id` 以便正确关联抽取结果

## 使用示例

### Go服务端调用示例

```go
// 准备文件列表（带material_id）
files := []File{
    {MaterialId: 1, URL: "http://example.com/file1.txt"},
    {MaterialId: 2, URL: "http://example.com/file2.txt"},
}

// 创建任务请求
req := &PythonCreateTaskRequest{
    Files:      files,
    PromptText: promptText,
    Provider:   "deepseek",
    APIKey:     apiKey,
}

// 调用Python服务
resp, err := pythonClient.CreateTask(ctx, req)

// 获取任务状态
status, err := pythonClient.GetTaskStatus(ctx, resp.TaskID)

// 处理结果，现在可以通过material_id识别对应的材料
for _, result := range status.Result {
    fmt.Printf("Material %d: %d triples extracted\n", 
        result.MaterialId, result.TriplesCount)
    fmt.Printf("Output files: %+v\n", result.OutputFiles)
}
```

## 测试建议

1. 测试新格式（带material_id）：验证material_id正确传递和返回
2. 测试旧格式（不带material_id）：验证向后兼容性
3. 测试绝对路径返回：验证output_files中的路径为绝对路径
4. 测试多文件处理：验证每个文件结果都正确关联material_id

## 注意事项

1. 确保Go服务端的 `TaskFile` 结构已包含 `MaterialId` 字段（已确认）
2. 如果使用URL下载文件，下载的文件仍会保存到配置的上传目录
3. 输出文件路径现在是绝对路径，跨平台使用时注意路径分隔符（Python的Path会自动处理）

