# T14 - LangGraph 工具集成进阶 (Advanced Tool Integration)

## 1. ToolNode vs 手动工具调用

在 LangGraph 中集成工具时，有两种主要方式：使用内置的 **ToolNode** 或手动处理工具调用。理解两者的差异是做出正确技术选型的基础。

### 1.1 手动工具调用（传统方式）

在没有 ToolNode 时，需要手动解析和执行工具调用：

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json

@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return f"关于 '{query}' 的搜索结果：..."

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 今天晴朗，25°C"

tools = [search_web, get_weather]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# 手动处理流程
def manual_tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]

    # 1. 检查是否有工具调用
    if not last_message.tool_calls:
        return {"messages": []}

    # 2. 手动遍历每个工具调用
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # 3. 查找并执行对应工具
        selected_tool = next((t for t in tools if t.name == tool_name), None)
        if selected_tool:
            result = selected_tool.invoke(tool_args)
            # 4. 手动包装为 ToolMessage
            tool_messages.append(
                ToolMessage(content=result, name=tool_name, tool_call_id=tool_id)
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"错误：未知工具 {tool_name}",
                    name=tool_name,
                    tool_call_id=tool_id,
                    is_error=True
                )
            )

    return {"messages": tool_messages}
```

**手动方式的问题**：
- 代码冗长，容易出错
- 需要自己管理工具查找逻辑
- 错误处理分散且不一致
- 难以支持并行工具执行

### 1.2 ToolNode 方式（推荐）

```python
from langgraph.prebuilt import ToolNode

# 一行代码搞定！
tool_node = ToolNode(tools)

# 直接加入图
graph.add_node("tools", tool_node)

# 条件边自动判断是否需要调用工具
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

graph.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "chatbot")
```

### 1.3 对比总结

| 维度 | 手动调用 | ToolNode |
|------|----------|----------|
| **代码量** | 多（30-50行） | 少（1-3行） |
| **可靠性** | 取决于实现质量 | 经过充分测试 |
| **错误处理** | 需要自行实现 | 内置统一机制 |
| **并行执行** | 需要额外编写 | 自动支持 |
| **可维护性** | 较差 | 良好 |
| **灵活性** | 完全可控 | 通过参数配置 |

---

## 2. ToolNode 基础用法

### 2.1 创建与注册

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# 定义工具集
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息"""
    # 实际实现中可接入 Tavily、SerpAPI 等
    return f"已搜索: {query}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    weather_data = {
        "北京": "晴天, 28°C",
        "上海": "多云, 26°C",
        "广州": "雷阵雨, 30°C",
    }
    return weather_data.get(city, f"暂无{city}的天气数据")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持加减乘除和基础函数"""
    try:
        result = eval(expression)  # 生产环境应使用更安全的方式
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

# 收集所有工具
tools = [search_web, get_weather, calculator]

# 创建 ToolNode
tool_node = ToolNode(tools)

print(f"ToolNode 包含 {len(tool_node.tools)} 个工具:")
for t in tool_node.tools:
    print(f"  - {t.name}: {t.description[:50]}...")
```

### 2.2 完整的图构建示例

```python
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# 定义系统提示
SYSTEM_PROMPT = """你是一个智能助手。你可以使用以下工具来帮助用户回答问题：
- search_web: 搜索网络信息
- get_weather: 查询天气
- calculator: 数学计算

请根据用户的问题判断是否需要使用工具，如果需要则调用相应的工具。"""

def chatbot(state: MessagesState):
    """聊天机器人节点"""
    model = ChatOpenAI(model="gpt-4o-mini")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def route_tools(state: MessagesState):
    """路由函数：决定下一步走向"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# 构建图
graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "__end__": END})
graph.add_edge("tools", "chatbot")

# 编译
app = graph.compile()
```

### 2.3 执行流程可视化

```
用户输入: "北京今天天气怎么样？气温比上海高多少度？"
         │
         ▼
   ┌─────────────┐
   │   chatbot   │ ← LLM 分析需要调用 get_weather (两次)
   │             │   生成 tool_calls: [get_weather("北京"), get_weather("上海")]
   └──────┬──────┘
          │ 有 tool_calls → "tools"
          ▼
   ┌─────────────┐
   │    tools    │ ← ToolNode 并行执行两个工具调用
   │             │   返回两条 ToolMessage
   └──────┬──────┘
          │ 固定回到 chatbot
          ▼
   ┌─────────────┐
   │   chatbot   │ ← LLM 结合工具结果生成最终回答
   │             │   "北京今天晴天28°C，比上海的26°C高2°C"
   └──────┬──────┘
          │ 无 tool_calls → __end__
          ▼
       返回结果
```

---

## 3. 工具调用错误处理

生产环境中，工具调用难免会遇到各种异常情况。LangGraph 提供了多层错误处理机制。

### 3.1 全局错误控制：raise_on_error 参数

```python
from langgraph.prebuilt import ToolNode

# 方式1：单个工具出错不中断其他工具（默认行为）
tool_node_lenient = ToolNode(tools, raise_on_error=False)
# 行为：某个工具抛异常时，返回包含错误信息的 ToolMessage，继续执行其他工具

# 方式2：任何工具出错立即中断
tool_node_strict = ToolNode(tools, raise_on_error=True)
# 行为：第一个工具出错就抛出异常，整个节点执行终止
```

**raise_on_error=False 的效果演示**：

```python
# 假设有三个工具调用，第二个会失败
tool_calls = [
    {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"},
    {"name": "risky_operation", "args": {"data": "bad"}, "id": "call_2"},  # 会失败
    {"name": "calculator", "args": {"expression": "1+1"}, "id": "call_3"},
]

# raise_on_error=False 时，返回结果：
result = tool_node_lenient.invoke({"messages": [AIMessage("", tool_calls=tool_calls)]})
# result["messages"] 包含：
# - ToolMessage(get_weather 结果, tool_call_id="call_1")     ✅ 成功
# - ToolMessage("Error: ...", tool_call_id="call_2", is_error=True)  ❌ 失败但记录了
# - ToolMessage("计算结果: 2", tool_call_id="call_3")           ✅ 成功
```

### 3.2 单个工具内部错误处理

```python
from langchain_core.tools import tool
import requests

@tool
def fetch_api_data(endpoint: str) -> str:
    """
    安全地获取 API 数据
    内部捕获所有可能的异常并返回友好的错误信息
    """
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()  # 检查 HTTP 状态码
        return response.text[:2000]  # 限制返回长度

    except requests.exceptions.Timeout:
        return f"请求超时: 端点 {endpoint} 在10秒内无响应，请检查服务状态或稍后重试。"

    except requests.exceptions.ConnectionError:
        return f"连接失败: 无法连接到 {endpoint}，请确认 URL 正确且服务可用。"

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "未知"
        return f"HTTP错误({status_code}): 服务器返回了错误响应，端点可能存在权限或配置问题。"

    except requests.exceptions.RequestException as e:
        return f"请求异常: {type(e).__name__}: {str(e)}"

    except Exception as e:
        return f"未知错误: {type(e).__name__}: {str(e)}"
```

### 3.3 重试机制封装

对于网络类工具，可以添加自动重试：

```python
import time
from functools import wraps
from typing import Callable

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """
    工具重试装饰器
    max_retries: 最大重试次数
    delay: 重试间隔(秒)，支持指数退避
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # 指数退避
                        time.sleep(wait_time)
                        continue
            return f"操作失败（已重试{max_retries}次）: {last_exception}"
        return wrapper
    return decorator

# 使用示例
@with_retry(max_retries=3, delay=0.5)
def unreliable_api_call(url: str) -> str:
    """可能失败的外部 API 调用"""
    response = requests.get(url, timeout=5)
    return response.json()
```

### 3.4 结构化错误响应

让 LLM 更好地理解和处理错误：

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class ToolError:
    """标准化的工具错误格式"""
    error_type: str      # 错误分类
    message: str         # 人类可读的错误描述
    suggestion: str      # 给 LLM 的建议（如何修复/替代方案）
    recoverable: bool    # 是否可以重试

    def to_string(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

@tool
def database_query(sql: str) -> str:
    """执行数据库查询（仅允许 SELECT 语句）"""
    sql_upper = sql.strip().upper()

    # 权限检查
    if not sql_upper.startswith('SELECT'):
        error = ToolError(
            error_type="PERMISSION_DENIED",
            message=f"仅支持 SELECT 查询，不允许执行: {sql.strip()[:20]}...",
            suggestion="请将操作改为只读的 SELECT 查询语句",
            recoverable=True
        )
        return error.to_string()

    # SQL 注入检测（简化版）
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE', '--']
    if any(kw in sql_upper for kw in dangerous_keywords):
        error = ToolError(
            error_type="SECURITY_VIOLATION",
            message="检测到潜在的不安全 SQL 操作",
            suggestion="请移除危险关键字，确保查询为纯读取操作",
            recoverable=False
        )
        return error.to_string()

    # 执行查询...
    return "查询结果: ..."
```

---

## 4. 运行时值传递给工具（Injected Args）

有些工具参数不应该由 LLM 决定，而是需要在运行时从外部注入——例如用户ID、会话上下文、配置参数等。

### 4.1 场景分析

```python
@tool
def get_user_preference(user_id: str) -> str:
    """
    获取用户偏好设置
    问题：user_id 不应该由 LLM 猜测！应该从登录态/Session 中获取
    """
    pass

@tool
def read_file(path: str, allowed_directories: list[str]) -> str:
    """
    读取文件内容
    问题：allowed_directories 是安全策略参数，应由系统注入
    """
    pass

@tool
def query_database(conn_string: str, sql: str) -> str:
    """
    执行数据库查询
    问题：conn_string 是敏感凭证，绝不能让 LLM 看到
    """
    pass
```

### 4.2 Injected Args 机制

LangGraph 的 ToolNode 支持 `injected_args` 参数，用于指定哪些参数应从运行时 Context 中注入：

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Annotated
from langgraph.config import RunnableConfig

# 使用 Annotated 标记注入参数
@tool
def get_user_orders(
    user_id: Annotated[str, "injected"],  # 标记为注入参数
    status_filter: str = "all"            # 普通 LLM 决定的参数
) -> str:
    """获取用户的订单列表"""
    # user_id 将从运行时 config 中注入
    return f"用户 {user_id} 的{status_filter}订单: ..."

tools_with_injection = [get_user_orders]

# 创建 ToolNode 并指定注入来源
tool_node = ToolNode(
    tools_with_injection,
    # injected_args 可以是一个字典或 callable
    injected_args=lambda config: {
        "user_id": config["configurable"].get("user_id", "anonymous")
    }
)
```

### 4.3 完整注入示例

```python
from datetime import datetime
from typing import Annotated, Optional

@tool
def context_aware_search(
    query: str,
    user_id: Annotated[str, "injected"],
    session_id: Annotated[str, "injected"],
    current_time: Annotated[str, "injected"],
    user_tier: Annotated[str, "injected"],
) -> str:
    """
    带上下文的搜索功能
    会根据用户等级和时间进行个性化过滤
    """
    tier_bonus = {"premium": "[优先]", "standard": "", "free": "[基础]"}
    return (
        f"[{current_time}] {tier_bonus.get(user_tier, '')} "
        f"用户 {user_id} (会话 {session_id}) 搜索 '{query}' 的结果: ..."
    )

@tool
def file_operations(
    action: str,              # LLM 决定：read/write/list
    path: str,                # LLM 决定：目标路径
    content: Optional[str] = None,  # LLM 决定：写入内容（可选）
    workspace: Annotated[str, "injected"] = None,  # 注入：工作目录
    permissions: Annotated[list, "injected"] = None,  # 注入：权限列表
) -> str:
    """安全的文件操作工具"""
    if action == "write" and "write" not in (permissions or []):
        return "错误：当前用户没有写入权限"
    # ...
    return f"操作完成: {action} {path}"

def create_injected_config(config: RunnableConfig) -> dict:
    """构建注入参数的工厂函数"""
    configurable = config.get("configurable", {})
    return {
        "user_id": configurable.get("user_id", "anon"),
        "session_id": configurable.get("session_id", "default"),
        "current_time": datetime.now().isoformat(),
        "user_tier": configurable.get("user_tier", "free"),
        "workspace": configurable.get("workspace", "/tmp/sandbox"),
        "permissions": configurable.get("permissions", ["read"]),
    }

# 使用
tool_node = ToolNode(
    [context_aware_search, file_operations],
    injected_args=create_injected_config
)

# 调用时传入用户上下文
result = app.invoke(
    {"messages": [HumanMessage(content="帮我查一下最近的订单")]},
    config={
        "configurable": {
            "thread_id": "sess_001",
            "user_id": "user_12345",
            "session_id": "sess_abc",
            "user_tier": "premium",
            "workspace": "/home/user/docs",
            "permissions": ["read", "write"],
        }
    }
)
```

### 4.4 注入 vs 默认值对比

| 特性 | Injected Args | 默认参数值 |
|------|---------------|------------|
| **值来源** | 运行时 Config | 编译时固定 |
| **每次调用可变？** | 是（随 Config 变化） | 否（固定不变） |
| **安全性** | 高（LLM 看不到） | 低（LLM 可能在 prompt 中看到） |
| **适用场景** | 用户身份、权限、时间等动态值 | 常量配置、回退默认值 |
| **LLM 是否感知** | 否（完全透明） | 是（出现在 schema 中） |

---

## 5. 从工具更新图状态

默认情况下，工具执行的结果会被追加到 `messages` 字段中。但在某些场景下，我们希望工具能更新 State 的其他字段。

### 5.1 默认行为回顾

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]       # 默认追加
    search_results: Annotated[list, add]  # 我们想让工具也更新这个字段
    current_task: str                     # 或者更新当前任务状态
```

### 5.2 方法一：通过 Reducer 让工具输出影响多字段

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
import json

class ExtendedAgentState(TypedDict):
    messages: Annotated[list, add]
    extracted_data: Annotated[dict, lambda old, new: {**old, **new}]
    task_status: str

# 自定义 ToolNode 包装器
class StateAwareToolNode(ToolNode):
    """能够同时更新多个 State 字段的 ToolNode"""

    def _execute_tools(self, messages, *args, **kwargs):
        # 先执行原始的工具调用逻辑
        original_result = super()._execute_tools(messages, *args, **kwargs)

        # 解析工具消息，提取额外的状态更新
        additional_state = {}
        for msg in original_result.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.content.startswith("{"):
                try:
                    data = json.loads(msg.content)
                    if "_state_update" in data:
                        additional_state.update(data["_state_update"])
                except json.JSONDecodeError:
                    pass

        # 合并到结果中
        if additional_state:
            original_result.update(additional_state)

        return original_result
```

### 5.3 方法二：工具返回结构化数据配合后处理节点

```python
@tool
def deep_research(topic: str) -> str:
    """
    深度研究一个主题，返回结构化结果
    返回 JSON 格式，包含研究结果和元数据
    """
    research_result = {
        "summary": f"关于 {topic} 的研究发现...",
        "key_points": ["要点1", "要点2", "要点3"],
        "sources": ["source1.com", "source2.com"],
        "confidence": 0.85,
        "_metadata": {
            "research_depth": "deep",
            "sources_count": 5,
            "processing_time_ms": 2300
        }
    }
    return json.dumps(research_result, ensure_ascii=False)

def post_process_tools(state: ExtendedAgentState):
    """后处理节点：从工具结果中提取结构化数据"""
    new_state = {}
    messages = state.get("messages", [])

    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if "summary" in data:
                    new_state.setdefault("extracted_data", {})["summary"] = data["summary"]
                if "key_points" in data:
                    new_state.setdefault("extracted_data", {})["points"] = data["key_points"]
                if "_metadata" in data:
                    new_state.setdefault("research_meta", []).append(data["_metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

    return new_state if new_state else {}

# 图中增加后处理节点
graph.add_node("post_process", post_process_tools)
graph.add_edge("tools", "post_process")
graph.add_edge("post_process", "chatbot")
```

### 5.4 方法三：利用 ReturnMessages 自定义工具包装

```python
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """搜索结果的标准化输出模型"""
    summary: str = Field(description="搜索摘要")
    urls: list[str] = Field(description="相关链接")
    relevance_score: float = Field(description="相关性评分 0-1")

class StructuredSearchTool(BaseTool):
    """返回结构化数据的自定义工具基类"""
    name = "structured_search"
    description = "搜索并返回结构化结果"

    def _run(self, query: str) -> dict:
        # 执行实际搜索...
        raw_results = self._do_search(query)

        # 返回结构化的 dict（而非字符串）
        # LangGraph 会将其序列化为 ToolMessage
        return {
            "content": f"找到 {len(raw_results)} 条结果",
            "structured_output": SearchResult(
                summary="搜索摘要...",
                urls=["url1", "url2"],
                relevance_score=0.9
            ).model_dump(),
            # 这个字段可以被专门的 reducer 处理
            "_state_updates": {
                "last_search_query": query,
                "search_performed_at": datetime.now().isoformat()
            }
        }

    def _do_search(self, query: str) -> list:
        # 实际搜索实现
        pass
```

---

## 6. 管理大量工具的策略

当 Agent 需要集成几十甚至上百个工具时，如何有效管理成为关键挑战。

### 6.1 分组策略（Grouping）

按功能域对工具进行分组，每组使用独立的 ToolNode：

```python
# ===== 文件操作组 =====
file_tools = [
    read_file,
    write_file,
    list_directory,
    delete_file,
    move_file,
]
file_tool_node = ToolNode(file_tools)

# ===== 数据库操作组 =====
db_tools = [
    execute_sql_query,
    get_table_schema,
    export_to_csv,
    import_from_csv,
]
db_tool_node = ToolNode(db_tools)

# ===== Web API 组 =====
web_tools = [
    search_web,
    fetch_url_content,
    send_email,
    call_webhook,
]
web_tool_node = ToolNode(web_tools)

# ===== 通信组 =====
comm_tools = [
    send_slack_message,
    create_jira_ticket,
    notify_team,
]
comm_tool_node = ToolNode(comm_tools)

# 图中的路由
def route_by_domain(state):
    """根据意图路由到不同的工具组"""
    intent = classify_intent(state["messages"][-1].content)
    routing_map = {
        "file_ops": "file_tools",
        "database": "db_tools",
        "web_api": "web_tools",
        "communication": "comm_tools",
    }
    return routing_map.get(intent, "__end__")

graph.add_conditional_edges("router", route_by_domain, {
    "file_tools": "file_tools",
    "db_tools": "db_tools",
    "web_api": "web_tools",
    "comm_tools": "comm_tools",
    "__end__": END,
})
```

### 6.2 动态加载策略（Dynamic Loading）

不是一次性加载所有工具，而是根据任务动态选择：

```python
class DynamicToolRegistry:
    """动态工具注册表"""

    def __init__(self):
        self._all_tools = {}  # 全量工具库
        self._categories = {}  # 分类索引

    def register(self, category: str, tool):
        """注册工具到分类"""
        self._all_tools[tool.name] = tool
        self._categories.setdefault(category, []).append(tool.name)

    def get_tools_for_task(self, task_description: str, max_tools: int = 10) -> list:
        """根据任务描述选择最相关的工具子集"""
        # 简单的关键词匹配（实际项目可以用 embedding 相似度）
        relevant = []
        for category, tools in self._categories.items():
            if self._category_matches(task_description, category):
                for tool_name in tools:
                    relevant.append(self._all_tools[tool_name])

        # 限制数量避免 token 消耗过大
        return relevant[:max_tools]

    def _category_matches(self, text: str, category: str) -> bool:
        keywords = {
            "file": ["文件", "目录", "读写", "保存", "file", "directory"],
            "database": ["数据库", "查询", "表", "SQL", "database", "query"],
            "web": ["搜索", "网页", "URL", "API", "web", "search"],
            "code": ["代码", "运行", "调试", "执行", "code", "run"],
        }
        category_keywords = keywords.get(category.lower(), [])
        return any(kw in text.lower() for kw in category_keywords)


# 使用方式
registry = DynamicToolRegistry()

# 启动时注册所有工具
registry.register("file", read_file)
registry.register("file", write_file)
registry.register("database", execute_sql_query)
registry.register("web", search_web)
# ... 注册更多

# 运行时动态选择
def dynamic_chatbot(state):
    task_desc = state["messages"][-1].content
    selected_tools = registry.get_tools_for_task(task_desc)

    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(selected_tools)
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 注意：ToolNode 也需要动态创建
def create_dynamic_tool_node(state):
    task_desc = state["messages"][-1].content
    selected_tools = registry.get_tools_for_task(task_desc)
    return ToolNode(selected_tools)
```

### 6.3 权限控制策略（Access Control）

通过 Context 控制不同用户可用的工具：

```python
from enum import Enum
from dataclasses import dataclass

class UserRole(Enum):
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class PermissionConfig:
    role: UserRole
    allowed_tools: set[str]
    rate_limits: dict[str, int]  # tool_name -> max_calls_per_session

# 定义各角色的权限
ROLE_PERMISSIONS = {
    UserRole.GUEST: PermissionConfig(
        role=UserRole.GUEST,
        allowed_tools={"search_web", "get_weather"},
        rate_limits={"search_web": 5}
    ),
    UserRole.USER: PermissionConfig(
        role=UserRole.USER,
        allowed_tools={
            "search_web", "get_weather", "calculator",
            "read_file", "list_directory"
        },
        rate_limits={"search_web": 20, "read_file": 50}
    ),
    UserRole.ADMIN: PermissionConfig(
        role=UserRole.ADMIN,
        allowed_tools={t.name for t in all_tools},  # 全部工具
        rate_limits={}  # 无限制
    ),
}

class PermissionAwareToolNode(ToolNode):
    """带权限控制的 ToolNode"""

    def __init__(self, all_tools, permission_fn=None):
        super().__init__(all_tools)
        self.permission_fn = permission_fn

    def _filter_tools(self, config):
        """根据权限过滤可用工具"""
        if self.permission_fn:
            allowed = self.permission_fn(config)
            return [t for t in self.tools if t.name in allowed]
        return self.tools

def get_user_permissions(config: RunnableConfig) -> set[str]:
    """从 config 中提取用户权限"""
    role = config.get("configurable", {}).get("user_role", "GUEST")
    permission = ROLE_PERMISSIONS.get(UserRole(role), ROLE_PERMISSIONS[UserRole.GUEST])
    return permission.allowed_tools

# 创建带权限的 ToolNode
permissioned_tool_node = PermissionAwareToolNode(
    all_tools=all_tools,
    permission_fn=get_user_permissions
)
```

### 6.4 工具注册表模式（Registry Pattern）

建立统一的工具注册和管理中心：

```python
class ToolRegistry:
    """
    企业级工具注册表
    支持版本管理、依赖声明、健康检查
    """

    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._version_history: dict[str, list] = {}

    def register(self, tool, *, version: str = "1.0.0",
                 dependencies: list[str] = None,
                 tags: list[str] = None,
                 deprecated: bool = False):
        """注册工具及其元数据"""
        self._tools[tool.name] = tool
        self._metadata[tool.name] = {
            "version": version,
            "tags": tags or [],
            "deprecated": deprecated,
            "registered_at": datetime.now().isoformat(),
        }
        self._dependencies[tool.name] = dependencies or []
        self._version_history.setdefault(tool.name, []).append({
            "version": version,
            "date": datetime.now().isoformat()
        })

    def get(self, name: str) -> Any:
        """获取工具"""
        tool = self._tools.get(name)
        if tool and self._metadata[name]["deprecated"]:
            print(f"⚠️ 工具 {name} 已弃用，建议迁移到替代方案")
        return tool

    def resolve_dependencies(self, names: list[str]) -> list:
        """解析依赖关系，返回拓扑排序后的工具列表"""
        resolved = []
        resolving = set()

        def resolve(name):
            if name in resolved:
                return
            if name in resolving:
                raise CircularDependencyError(name)
            resolving.add(name)
            for dep in self._dependencies.get(name, []):
                resolve(dep)
            resolving.discard(name)
            resolved.append(name)

        for name in names:
            resolve(name)

        return [self._tools[n] for n in resolved if n in self._tools]

    def health_check(self) -> dict:
        """对所有工具进行健康检查"""
        results = {}
        for name, tool in self._tools.items():
            try:
                # 尝试轻量级调用验证工具可用性
                test_input = self._generate_test_input(tool)
                tool.invoke(test_input)
                results[name] = {"status": "healthy"}
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
        return results

    def list_by_tag(self, tag: str) -> list:
        """按标签筛选工具"""
        return [
            self._tools[name]
            for name, meta in self._metadata.items()
            if tag in meta["tags"] and not meta["deprecated"]
        ]

    def get_stats(self) -> dict:
        """获取注册表统计信息"""
        total = len(self._tools)
        deprecated = sum(1 for m in self._metadata.values() if m["deprecated"])
        return {
            "total_tools": total,
            "active": total - deprecated,
            "deprecated": deprecated,
            "tags_used": len(set(t for m in self._metadata.values() for t in m["tags"]))
        }


# 使用示例
registry = ToolRegistry()

registry.register(search_web, tags=["web", "external-api"])
registry.register(get_weather, tags=["utility", "external-api"])
registry.register(calculator, tags=["math", "builtin"])

# 按需获取
web_tools = registry.list_by_tag("web")
math_tools = registry.list_by_tag("math")

# 健康检查
health = registry.health_check()
for name, status in health.items():
    print(f"{name}: {status['status']}")
```

---

## 7. 工具集成最佳实践清单

### 7.1 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 一个工具只做一件事 | `get_weather` 和 `forecast_weather` 分开 |
| **清晰命名** | 名字即文档 | 用 `search_academic_papers` 而非 `search` |
| **详细描述** | description 要具体 | 不要写"搜索"，要写"搜索学术论文数据库" |
| **类型明确** | 参数要有类型注解 | `query: str` 比 `query` 好 |
| **输入验证** | 内部做参数校验 | 拒绝非法输入而不是崩溃 |
| **幂等性** | 同样输入产生同样输出 | 缓存友好，重试安全 |

### 7.2 性能优化建议

```python
# 1. 大型工具集使用 embedding 进行语义匹配选择
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_tool_selection(query: str, tools: list, top_k: int = 10) -> list:
    """基于语义相似度选择最相关的工具"""
    query_embedding = embedder.encode(query)

    tool_embeddings = embedder.encode([t.description for t in tools])
    similarities = cosine_similarity([query_embedding], tool_embeddings)[0]

    # 选择 top_k 最相似的
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [tools[i] for i in top_indices]

# 2. 工具结果缓存
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def cached_search(query_hash: str) -> str:
    """带缓存的搜索工具"""
    actual_query = decode_hash(query_hash)  # 实际实现
    return do_actual_search(actual_query)

# 3. 异步工具优先
@tool
async def async_fetch(url: str) -> str:
    """异步 HTTP 请求"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

### 7.3 监控与日志

```python
import logging
import time
from functools import wraps

logger = logging.getLogger("tool_monitoring")

def monitor_tool_execution(tool_func):
    """工具执行监控装饰器"""
    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = tool_func.__name__

        logger.info(f"TOOL_START | {tool_name} | args={args}")

        try:
            result = tool_func(*args, **kwargs)
            duration = time.time() - start_time

            logger.info(
                f"TOOL_SUCCESS | {tool_name} | "
                f"duration={duration:.3f}s | "
                f"result_length={len(str(result))}"
            )

            # 发送到监控系统
            send_metrics(tool_name, duration, success=True)

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"TOOL_ERROR | {tool_name} | "
                f"duration={duration:.3f}s | "
                f"error={type(e).__name__}: {e}"
            )
            send_metrics(tool_name, duration, success=False, error=str(e))
            raise

    return wrapper
```

---

## 本章小结

本章深入探讨了 LangGraph 中工具集成的进阶技巧，核心知识点包括：

1. **ToolNode vs 手动调用**：ToolNode 以极简 API 提供可靠、内置并行、统一错误处理的工具执行能力，是生产环境的首选方案
2. **错误处理三层体系**：全局 `raise_on_error` 控制、工具内部 try-catch、标准化错误响应格式，构成完整的容错机制
3. **Injected Args 机制**：解决运行时动态参数注入需求，使敏感信息和上下文数据对 LLM 透明传递
4. **多字段状态更新**：通过自定义 Reducer、后处理节点或结构化返回值，打破工具只能更新 messages 的限制
5. **大规模工具管理**：分组隔离、动态加载、权限控制、注册表模式四大策略应对工具数量膨胀

掌握这些进阶技巧后，你将能够构建出工具丰富、架构清晰、安全可控的企业级 Agent 系统。

---

> **参考来源**：黑马程序员 LangChain 课程 - BV178w1z7EHQ（第3章 Agent 进阶 - T14 LangGraph 工具集成进阶）
