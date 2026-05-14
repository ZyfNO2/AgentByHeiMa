# T02 - State 自定义

## 1. 默认 AgentState 的局限

在使用 `create_agent()` 创建 Agent 时，如果不指定 `state_schema` 参数，系统会使用内置的默认 `AgentState`。这个默认 State 的结构非常简单——仅包含一个 `messages` 字段用于存储对话历史。

### 1.1 默认 AgentState 的结构

```python
# LangChain 内置的默认 AgentState（简化展示）
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """默认 Agent 状态——只有 messages"""
    messages: list[BaseMessage]
```

这种极简设计在简单的对话场景下完全够用，但在实际生产环境中往往面临以下局限：

### 1.2 四大局限性分析

| 局限 | 具体表现 | 影响 |
|------|----------|------|
| **无法记录调用统计** | 不知道 LLM 被调用了几次、每次耗时多少 | 无法做性能监控和计费 |
| **无法追踪错误状态** | 工具调用失败只能看日志，无法在 State 中累积 | 难以实现自动重试和错误恢复 |
| **无法保存中间结果** | 复杂的多步骤任务中，中间计算结果无处存放 | 重复计算、效率低下 |
| **无法表达业务语义** | 只有通用的 messages 列表，缺乏领域特定的字段 | 代码可读性差、维护困难 |

### 1.3 一个具体的痛点场景

假设我们在开发一个**数据分析 Agent**，用户可以要求它"分析销售数据并生成报告"。这个过程涉及多个步骤：

```
用户: "分析上月销售数据，找出Top3产品"

期望的状态跟踪：
├── 当前阶段: data_analysis（而非只有一堆消息）
├── 已调用工具: [load_data, filter_data, aggregate, rank]（4次）
├── 中间结果: filtered_rows=1523, total_revenue=¥892K
├── 耗时统计: llm_calls=3, total_time=12.7s
└── 错误记录: []（无错误）

默认 State 能提供的：
└── messages: [用户消息, AI消息(含tool_call), 工具结果, AI消息, ...]
    （一团乱麻，需要从消息中反推一切）
```

显然，我们需要一种更有结构的 State 来承载这些信息。

---

## 2. 自定义 State 步骤详解

### 2.1 基础自定义方法

自定义 State 的核心是继承或定义一个 `TypedDict`，并在其中声明所需的字段：

```python
from langchain.agents import AgentState
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class MyAgentState(AgentState):
    """
    自定义 Agent 状态

    继承自 AgentState 以保留 messages 字段的默认行为，
    同时扩展业务所需的额外字段。
    """
    messages: list[BaseMessage]              # 继承默认：对话历史（必须保留）

    # --- 业务统计字段 ---
    call_count: NotRequired[int] = 0         # 工具/LLM 调用累计次数
    total_time: NotRequired[float] = 0.0     # 累计执行耗时（秒）

    # --- 错误处理字段 ---
    errors: NotRequired[list[str]] = []      # 错误信息累积列表
    retry_count: NotRequired[int] = 0        # 重试计数

    # --- 业务语义字段 ---
    user_intent: NotRequired[str] = ""       # 解析出的用户意图
    current_stage: NotRequired[str] = ""     # 当前执行阶段
    intermediate_results: NotRequired[dict] = {}  # 中间计算结果缓存
```

### 2.2 关键语法解析：NotRequired

`NotRequired` 是 Python 3.11+ 的 `typing` 模块引入的特性（旧版本可通过 `typing_extensions` 使用），用于标记 TypedDict 中的可选字段。

#### NotRequired vs Required 对比

```python
from typing import Required, NotRequired
from typing_extensions import TypedDict

class DemoState(TypedDict):
    """演示 Required 和 NotRequired 的区别"""

    # Required：该字段必须存在，缺失时直接报 KeyError
    required_field: Required[str]

    # NotRequired：该字段可以不存在，访问时需处理缺失情况
    optional_field: NotRequired[str]

    # 带默认值的字段（推荐用法）
    field_with_default: NotRequired[int] = 42
```

| 特性 | `Required[T]` | `NotRequired[T]` | `NotRequired[T] = default` |
|------|---------------|------------------|---------------------------|
| **是否必须存在** | 是 | 否 | 否（不存在时用默认值） |
| **缺失时行为** | 直接抛出 KeyError | 访问时 KeyError | 返回默认值 |
| **类型检查严格度** | 严格 | 宽松 | 宽松 |
| **适用场景** | 核心必填字段 | 可能有也可能没有的字段 | 大部分自定义业务字段 |
| **推荐指数** | 仅用于 messages 等核心字段 | 少量使用 | **强烈推荐** |

### 2.3 推荐的自定义 State 模板

```python
from datetime import datetime
from enum import Enum
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class TaskStage(str, Enum):
    """任务执行阶段枚举——增强类型安全"""
    INTENT_RECOGNITION = "intent_recognition"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    RESULT_SYNTHESIS = "result_synthesis"
    COMPLETED = "completed"
    FAILED = "failed"


class ProductionAgentState(TypedDict):
    """
    生产级 Agent State 模板

    设计原则：
    1. 所有自定义字段均使用 NotRequired + 默认值
    2. 使用枚举替代字符串表示有限集合
    3. 按功能域分组组织字段
    """

    # ======== 核心域（继承自 AgentState 的行为）========
    messages: list[BaseMessage]

    # ======== 统计监控域 ========
    call_count: NotRequired[int] = 0
    llm_token_usage: NotRequired[dict] = {}       # {"input": 100, "output": 50}
    total_execution_time: NotRequired[float] = 0.0
    stage_start_time: NotRequired[float] = 0.0    # 当前阶段的开始时间戳

    # ======== 任务控制域 ========
    current_stage: NotRequired[TaskStage] = TaskStage.INTENT_RECOGNITION
    user_intent: NotRequired[str] = ""
    task_priority: NotRequired[int] = 1           # 1=普通, 2=高, 3=紧急

    # ======== 错误恢复域 ========
    errors: NotRequired[list[dict]] = []          # [{"stage": "...", "error": "...", "time": ...}]
    retry_count: NotRequired[int] = 0
    max_retries: NotRequired[int] = 3

    # ======== 结果缓存域 ========
    intermediate_results: NotRequired[dict] = {}
    final_result: NotRequired[str] = ""

    # ======== 审计追踪域 ========
    created_at: NotRequired[str] = ""             # ISO格式时间戳
    updated_at: NotRequired[str] = ""
```

---

## 3. 在节点中读写自定义 State

定义好 State 之后，关键问题是如何在 Agent 的各个节点（Node）中读取和更新这些字段。

### 3.1 LangGraph 节点的数据流模型

在 LangGraph 中，每个节点的函数签名遵循统一的模式：

```python
def node_function(state: YourStateType) -> dict:
    """
    节点函数的标准模式

    Args:
        state: 当前状态的完整快照（输入）

    Returns:
        dict: 包含需要更新的字段的字典（增量更新）
              返回空字典 {} 表示不做任何修改
    """
    # 1. 从 state 中读取所需数据
    current_value = state.get("field_name", default_value)

    # 2. 执行业务逻辑
    new_value = do_something(current_value)

    # 3. 返回需要更新的字段（只返回变化的字段）
    return {
        "field_name": new_value,
        "another_field": another_new_value,
    }
```

### 3.2 读取 State 数据

```python
def my_processing_node(state: ProductionAgentState):
    """在节点中读取自定义 State 字段"""

    # 方式一：使用 .get() 方法（推荐，安全）
    call_count = state.get("call_count", 0)
    current_stage = state.get("current_stage", TaskStage.INTENT_RECOGNITION)
    errors = state.get("errors", [])

    # 方式二：直接键访问（仅确定字段存在时使用）
    messages = state["messages"]  # messages 总是存在的

    # 方式三：使用 .get() 处理嵌套字典
    token_info = state.get("llm_token_usage", {})
    input_tokens = token_info.get("input", 0)

    print(f"[调试] 当前阶段: {current_stage}, 已调用 {call_count} 次")

    # ... 业务逻辑 ...

    return {}  # 暂时不更新任何内容
```

### 3.3 写入/更新 State 数据

```python
import time

def track_metrics_node(state: ProductionAgentState):
    """统计型节点——记录调用指标"""

    start_time = time.time()

    # 模拟一些工作
    result = perform_heavy_computation()

    elapsed = time.time() - start_time

    # 读取当前值
    current_count = state.get("call_count", 0)
    current_total = state.get("total_execution_time", 0.0)
    current_errors = state.get("errors", [])

    # 返回增量更新
    return {
        "call_count": current_count + 1,
        "total_execution_time": current_total + elapsed,
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "last_computation": result,
        },
    }


def error_handling_node(state: ProductionAgentState, error: Exception):
    """错误处理节点——累积错误信息"""

    from datetime import datetime

    current_errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # 记录新的错误
    new_error_entry = {
        "stage": state.get("current_stage", "unknown"),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
    }

    # 判断是否超过最大重试次数
    should_retry = retry_count < max_retries

    return {
        "errors": current_errors + [new_error_entry],
        "retry_count": retry_count + 1,
        "current_stage": TaskStage.FAILED if not should_retry else state.get("current_stage"),
    }
```

### 3.4 State 更新的重要规则

```
规则一：返回增量字典，不是替换整个 State
        ✅ return {"call_count": count + 1}
        ❌ return new_state_dict  （这会丢失其他字段！）

规则二：未返回的字段保持不变
        return {"call_count": 5}  →  其他字段原样保留

规则三：不要尝试在函数内直接修改 state 参数
        ❌ state["call_count"] += 1  （无效操作）
        ✅ return {"call_count": state["call_count"] + 1}

规则四：列表/字典类型的字段需要手动合并
        ✅ return {"errors": old_errors + [new_error]}
        ❌ return {"errors": new_error}  （覆盖而非追加）
```

---

## 4. 实战案例：统计型 State

下面构建一个完整的可运行示例，实现 Agent 运行过程中的全面统计监控。

### 4.1 完整代码示例

```python
"""
统计型 State 完整实战示例
功能：追踪 Agent 运行过程中的调用次数、耗时、错误等信息
"""

import time
import json
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义带统计能力的 State
# ============================================================
class StatsAgentState(TypedDict):
    """带统计功能的 Agent State"""
    messages: list[BaseMessage]

    # --- 统计字段 ---
    tool_call_count: NotRequired[int] = 0
    llm_call_count: NotRequired[int] = 0
    total_tool_time: NotRequired[float] = 0.0
    total_llm_time: NotRequired[float] = 0.0

    # --- 错误追踪 ---
    failed_tools: NotRequired[list[str]] = []

    # --- 调用链路 ---
    call_chain: NotRequired[list[dict]] = []   # [{"tool": "name", "time": 1.2, "success": True}]


# ============================================================
# 2. 定义工具（模拟外部服务调用）
# ============================================================
@tool
def search_database(query: str) -> str:
    """搜索数据库中的信息"""
    time.sleep(0.3)  # 模拟网络延迟
    return f"找到关于 '{query}' 的 5 条结果"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    time.sleep(0.1)  # 模拟计算延迟
    try:
        result = eval(expression)  # 实际项目中请勿使用 eval
        return f"计算结果: {result}"
    except Exception as e:
        raise ValueError(f"表达式无效: {e}")


@tool
def fetch_api(endpoint: str) -> str:
    """从外部 API 获取数据"""
    time.sleep(0.5)  # 模拟 API 延迟
    if "fail" in endpoint.lower():
        raise ConnectionError(f"无法连接到 {endpoint}")
    return f"API {endpoint} 返回了数据"


tools = [search_database, calculate, fetch_api]


# ============================================================
# 3. 定义节点函数
# ============================================================

def llm_node(state: StatsAgentState) -> dict:
    """模拟 LLM 调用节点"""
    start = time.time()
    time.sleep(0.2)  # 模拟 LLM 推理时间
    elapsed = time.time() - start

    count = state.get("llm_call_count", 0)
    total = state.get("total_llm_time", 0.0)

    # 模拟 LLM 决定调用工具
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage) and "搜索" in last_msg.content:
        return {
            "llm_call_count": count + 1,
            "total_llm_time": total + elapsed,
            "messages": state["messages"] + [
                AIMessage(content="", tool_calls=[
                    {"name": "search_database", "args": {"query": last_msg.content}, "id": "tc_1"}
                ])
            ],
        }
    return {
        "llm_call_count": count + 1,
        "total_llm_time": total + elapsed,
        "messages": state["messages"] + [
            AIMessage(content="我已理解您的问题，这是我的回答。")
        ],
    }


def tool_node(state: StatsAgentState) -> dict:
    """工具执行节点——包含统计逻辑"""
    start = time.time()

    # 获取最后一个 AI 消息中的工具调用
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
        return {}

    tc = last_msg.tool_calls[0]
    tool_name = tc["name"]
    tool_args = tc["args"]

    try:
        # 查找并执行对应的工具
        tool_fn = next((t for t in tools if t.name == tool_name), None)
        if tool_fn:
            result = tool_fn.invoke(tool_args)
            success = True
        else:
            result = f"未知工具: {tool_name}"
            success = False

    except Exception as e:
        result = f"工具执行错误: {e}"
        success = False

    elapsed = time.time() - start

    # 更新统计信息
    tool_count = state.get("tool_call_count", 0)
    total_tool_time = state.get("total_tool_time", 0.0)
    failed = state.get("failed_tools", [])
    chain = state.get("call_chain", [])

    updates = {
        "tool_call_count": tool_count + 1,
        "total_tool_time": total_tool_time + elapsed,
        "call_chain": chain + [{
            "tool": tool_name,
            "args": tool_args,
            "time": round(elapsed, 3),
            "success": success,
        }],
        "messages": state["messages"] + [
            {"role": "tool", "name": tool_name, "content": result, "tool_call_id": tc["id"]}
        ],
    }

    if not success:
        updates["failed_tools"] = failed + [tool_name]

    return updates


def stats_reporter_node(state: StatsAgentState) -> dict:
    """统计报告节点——输出汇总信息"""
    report = {
        "=== Agent 运行统计报告 ===": None,
        "LLM 调用次数": state.get("llm_call_count", 0),
        "工具调用次数": state.get("tool_call_count", 0),
        "LLM 总耗时(ms)": round(state.get("total_llm_time", 0) * 1000, 2),
        "工具总耗时(ms)": round(state.get("total_tool_time", 0) * 1000, 2),
        "失败的工具": state.get("failed_tools", []) or "无",
        "调用链路": state.get("call_chain", []),
    }

    print("\n" + "=" * 50)
    for k, v in report.items():
        if v is not None:
            print(f"  {k}: {v}")
        else:
            print(f"\n  {k}")
    print("=" * 50 + "\n")

    return {}  # 不修改 State，只输出报告


# ============================================================
# 4. 构建并运行 Graph
# ============================================================
def build_stats_agent():
    """构建带统计功能的 Agent Graph"""

    graph = StateGraph(StatsAgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_node("stats_reporter", stats_reporter_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        lambda s: "tools" if s["messages"][-1].tool_calls else "stats_reporter",
        {"tools": "tools", "stats_reporter": "stats_reporter"},
    )
    graph.add_edge("tools", "llm")  # 工具执行完回到 LLM
    graph.add_edge("stats_reporter", END)

    return graph.compile()


# ============================================================
# 5. 运行测试
# ============================================================
if __name__ == "__main__":
    agent = build_stats_agent()

    # 测试正常流程
    result = agent.invoke({
        "messages": [HumanMessage(content="搜索人工智能最新进展")]
    })

    # 输出最终统计
    print("\n最终 State 中的统计数据:")
    print(f"  LLM 调用次数: {result.get('llm_call_count', 0)}")
    print(f"  工具调用次数: {result.get('tool_call_count', 0)}")
    print(f"  总耗时: {result.get('total_llm_time', 0) + result.get('total_tool_time', 0):.3f}s")

    # 测试带错误的流程
    print("\n--- 测试错误场景 ---\n")
    result2 = agent.invoke({
        "messages": [HumanMessage(content="调用 fail_api 接口")]
    })
```

### 4.2 预期输出示例

```
==================================================
  === Agent 运行统计报告 ===
  LLM 调用次数: 2
  工具调用次数: 1
  LLM 总耗时(ms): 401.25
  工具总耗时(ms): 302.15
  失败的工具: 无
  调用链路: [
    {"tool": "search_database", "time": 0.302, "success": True}
  ]
==================================================

最终 State 中的统计数据:
  LLM 调用次数: 2
  工具调用次数: 1
  总耗时: 0.703s
```

---

## 5. State 与 Reducer 配合使用

在 LangGraph 中，对于列表类型的 State 字段（如 `messages`、`errors`），通常需要配合 Reducer 来确保数据被正确地**追加**而非**覆盖**。

### 5.1 什么是 Reducer？

Reducer 是一个函数，定义了当节点返回某字段的新值时，如何与已有值合并：

```python
from operator import add
from langgraph.graph import add_messages

# 对于列表类型字段，常用的 Reducer 有：

# add_messages：LangGraph 专用的消息追加 reducer（智能去重）
# add：简单的列表拼接 reducer
# lambda old, new: old + [new]：自定义追加逻辑
# lambda old, new: new：覆盖（默认行为，慎用）
```

### 5.2 在 State 定义中指定 Reducer

```python
from langgraph.graph import add_messages

class ReducerDemoState(TypedDict):
    # messages 使用专用 reducer（LangGraph 推荐）
    messages: Annotated[list[BaseMessage], add_messages]

    # errors 使用自定义追加 reducer
    errors: Annotated[list[str], operator.add]

    # call_chain 使用自定义 reducer（追加字典元素）
    call_chain: Annotated[list[dict], lambda old, new: old + ([new] if new else [])]

    # 普通字段不需要 reducer（默认覆盖行为）
    call_count: int
```

### 5.3 Reducer 选择指南

| 字段类型 | 推荐 Reducer | 说明 |
|----------|-------------|------|
| `messages` | `add_messages` | LangGraph 内置，智能处理消息去重和更新 |
| 错误列表 | `operator.add` 或自定义 | 简单追加即可 |
| 日志/审计列表 | `operator.add` | 保持时间顺序 |
| 计数器/累加值 | **不需要 Reducer** | 在节点中读取旧值+1后返回新值 |
| 最后更新值 | **不需要 Reducer** | 覆盖行为正是我们需要的 |
| 配置字典 | 自定义 merge 函数 | 深合并而非浅覆盖 |

---

## 6. State 设计最佳实践

### 6.1 字段设计原则

```
原则一：最小化原则
├── 只放当前 Agent 确实需要追踪的数据
├── 不要预想未来可能用到而提前加字段（YAGNI 原则）
└── 示例：如果不需要审计追踪，就不要加 created_at / updated_at

原则二：类型安全原则
├── 使用 TypedDict 或 Pydantic 做类型约束
├── 优先使用枚举代替字符串常量
└── 嵌套结构要有明确的类型注解

原则三：扁平优于嵌套
├── 避免过深的嵌套（不超过 2 层）
├── 如果嵌套不可避免，考虑拆分为独立的 Store namespace
└── 示例：❌ state["user"]["profile"]["preferences"]["style"]
          ✅ state["response_style"]

原则四：命名一致性
├── 使用一致的命名风格（snake_case）
├── 布尔字段以 is_/has_/can_ / should_ 开头
├── 时间字段以 _at / _time 结尾
└── 计数字段以 _count 结尾
```

### 6.2 常见 State 模式速查表

| 场景 | 推荐字段 | 类型 | Reducer |
|------|----------|------|---------|
| 简单对话 Agent | `messages` | `list[BaseMessage]` | `add_messages` |
| 需要调用统计 | `call_count`, `total_time` | `int`, `float` | 无 |
| 需要错误恢复 | `errors`, `retry_count` | `list[dict]`, `int` | `add` / 无 |
| 多步骤任务 | `current_stage`, `step_results` | `Enum`, `dict` | 无 |
| 需要审计日志 | `audit_log` | `list[dict]` | `add` |
| 流式输出 | `partial_response` | `str` | 无（覆盖） |

### 6.3 反模式警示

```python
# 反模式一：State 变成万能字典
class BadState(TypedDict):
    everything: dict  # 把所有东西都塞进一个 dict —— 失去了类型安全

# 反模式二：State 存储不应存的数据
class BadState2(TypedDict):
    messages: list[BaseMessage]
    api_key: str           # 安全风险！应该放在 Context
    db_connection: object  # 不可序列化！应该放在 Context
    user_history: list     # 应该放在 Store

# 反模式三：State 字段过多（超过 15 个）
class BloatedState(TypedDict):
    messages: list[BaseMessage]
    field_1: str           # 太多字段说明设计有问题
    field_2: str           # 考虑拆分到 Store
    field_3: str
    # ... 还有 20 个字段
```

---

## 7. 本章小结

本章深入讲解了 LangChain Agent 中 State 自定义的完整方法论：

1. **默认 AgentState 局限明显**：仅有 `messages` 字段无法满足生产环境的统计、错误追踪、中间结果缓存等需求。

2. **自定义 State 的核心步骤**：定义 TypedDict 类、使用 `NotRequired` + 默认值标注可选字段、在 `create_agent()` 中通过 `state_schema` 参数传入。

3. **节点中读写 State 的规范**：使用 `.get()` 安全读取、返回增量字典进行更新、列表类型字段配合 Reducer 使用。

4. **设计原则至关重要**：遵循最小化、类型安全、扁平化、命名一致性四大原则，避免 State 臃肿和职责混乱。

> **下一节预告**：T03 将介绍 Store（长期记忆）机制，学习如何在跨会话的维度持久化和检索用户数据。

---

## 参考来源

- 视频课程：黑马程序员 LangChain 教程 [BV178w1z7EHQ](https://www.bilibili.com/video/BV178w1z7EHQ)
- 配套文档：飞书文档 [IBPWwU7sUiuvbMkMYGacZxU8nsg](https://sid9kndrs6s.feishu.cn/wiki/IBPWwU7sUiuvbMkMYGacZxU8nsg)
- LangGraph State 文档：https://langchain-ai.github.io/langgraph/concepts/low_level/#state
- TypedDict 官方文档：https://docs.python.org/3/library/typing.html#typedict
