# T01 - Runtime 机制总览

## 1. 什么是 Runtime？

Runtime 是理解 Agent 内部运行状态的**核心框架**。在 LangChain 的 Agent 开发体系中，`create_agent()` 函数并非孤立运行，其底层架构建立在 **LangGraph 的 Runtime** 之上。

简单来说，Runtime 是 Agent 运行时的"操作系统"——它负责管理 Agent 执行过程中的所有状态流转、数据存储和上下文传递。理解 Runtime，就等于掌握了 Agent 的"神经系统"。

### 1.1 Runtime 与 LangGraph 的关系

```
┌─────────────────────────────────────────────────┐
│                  create_agent()                   │
│                                                   │
│  ┌───────────────────────────────────────────┐   │
│  │           LangGraph Runtime               │   │
│  │                                           │   │
│  │  ┌──────┐  ┌──────┐  ┌──────────────┐    │   │
│  │  │State │  │Store │  │   Context    │    │   │
│  │  └──────┘  └──────┘  └──────────────┘    │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

LangChain 的 `create_agent()` 本质上是对 LangGraph Runtime 的高级封装，开发者无需直接操作底层的 Graph 编程模型，但理解 Runtime 机制对于构建复杂、可维护的 Agent 至关重要。

---

## 2. 三大核心概念

Runtime 由三个核心概念组成：**State（状态）**、**Store（存储）**、**Context（上下文）**。三者各司其职，共同构成完整的运行时体系。

### 2.1 核心概念对照表

| 概念 | 说明 | 生命周期 | 类比 | 数据特征 |
|------|------|----------|------|----------|
| **State** | 短期记忆，存储当前对话信息、任务状态 | 单次请求（per-thread） | 工作记忆 / 草稿纸 | 动态变化、频繁读写 |
| **Store** | 长期记忆，包含用户偏好、失败经验等 | 跨会话（cross-session） | 长期记忆 / 档案柜 | 持久化、低频更新 |
| **Context** | 运行时上下文，传递配置参数 | 单次请求（per-invocation） | 环境配置 / 身份证 | 只读、静态不变 |

### 2.2 三者的定位对比

```
                    ┌─────────────────────────────────┐
                    │         Agent Runtime            │
                    │                                  │
  ┌─────────┐      │  ┌─────────┐  ┌─────────┐       │
  │  外部   │──────┼─>│ Context │  │  State  │<────┼──┐
  │  输入   │      │  │ (配置)  │  │ (状态)  │       │  │
  └─────────┘      │  └────┬────┘  └────┬────┘       │  │
                    │       │            │            │  │
                    │       v            v            │  │
                    │  ┌─────────┐  ┌─────────┐       │  │
                    │  │  Tool   │  │   LLM   │       │  │
                    │  └────┬────┘  └────┬────┘       │  │
                    │       │            │            │  │
                    │       └──────┬─────┘            │  │
                    │              v                   │  │
                    │        ┌─────────┐              │  │
                    │        │  Store  │<─────────────┘  │
                    │        │ (持久化)│                  │
                    │        └─────────┘                  │
                    └─────────────────────────────────┘
```

- **Context**：从外部注入，只读，整个请求周期内不变
- **State**：在节点间流转，可读可写，驱动 Agent 状态机运转
- **Store**：按需访问，跨会话持久化，提供长期记忆能力

---

## 3. 为什么需要区分三种"记忆"？

在传统的 Agent 开发中，开发者容易将所有数据混在一起处理——对话历史、用户偏好、系统配置全部塞进同一个字典或对象中。这种做法看似简单，实则埋下了诸多隐患。

### 3.1 混用带来的问题

#### 问题一：状态污染（State Pollution）

```python
# 错误示范：把所有东西都塞进 State
state = {
    "messages": [...],           # 对话历史 -- 合理
    "user_id": "u_123",          # 用户ID -- 不应变化！
    "api_key": "sk-xxx",         # API Key -- 安全隐患！
    "user_preference": {...},    # 用户偏好 -- 应该持久化
}
```

当 `user_id` 或 `api_key` 这样的配置数据被放入 State 后：
- 每次 LLM 调用都会携带这些不必要的数据，增加 token 消耗
- 如果某个节点意外修改了这些字段，会导致难以排查的 bug
- State 在请求结束后即销毁，用户偏好等需要持久化的数据会丢失

#### 问题二：安全风险（Security Risk）

将敏感信息（API Key、数据库连接串、Token）放入 State 意味着：
- 这些数据会被发送给 LLM Provider（即使不在 prompt 中显式使用）
- State 可能被序列化到日志或追踪系统中
- 多租户场景下存在数据泄露风险

#### 问题三：测试困难（Testing Difficulty）

当配置、状态和持久化数据混在一起时：
- 单元测试需要 mock 大量无关字段
- 无法独立测试业务逻辑和环境配置
- Mock 对象变得臃肿且脆弱

### 3.2 正确的三层分离

| 数据类型 | 放置位置 | 原因 |
|----------|----------|------|
| 对话消息、中间结果、调用计数 | **State** | 每次对话不同，动态变化 |
| 用户偏好、学习记录、失败经验 | **Store** | 跨会话保持，需要持久化 |
| API Key、用户身份、功能开关 | **Context** | 运行时不变，安全隔离 |

---

## 4. Runtime 在 createAgent 中的体现

LangChain 的 `create_agent()` API 直接暴露了 Runtime 的三大核心概念，让开发者可以在创建 Agent 时进行精细化配置。

### 4.1 完整的 Agent 创建示例

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel

# ============================================================
# 第一步：定义自定义 State Schema
# ============================================================
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class MyAgentState(TypedDict):
    """自定义 Agent 状态"""
    messages: list[BaseMessage]           # 继承默认：对话历史
    call_count: NotRequired[int]          # 新增：工具调用计数
    errors: NotRequired[list[str]]        # 新增：错误累积列表

# ============================================================
# 第二步：定义 Context Schema（运行时配置）
# ============================================================
class MyContext(BaseModel):
    """运行时上下文配置"""
    user_id: str                          # 用户唯一标识
    user_name: str                        # 用户显示名
    is_premium: bool = False              # 是否 VIP 用户
    max_tool_calls: int = 10              # 最大工具调用次数限制

# ============================================================
# 第三步：准备 Store（长期记忆）
# ============================================================
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # 生产环境建议用 SqliteStore

# ============================================================
# 第四步：定义工具（可访问 Runtime）
# ============================================================
@tool
async def get_weather(city: str, runtime):
    """查询指定城市的天气"""
    # 通过 runtime 访问 Context
    user_name = runtime.context.user_name

    # 通过 runtime 访问 Store（读取用户偏好）
    pref = await runtime.store.aget(
        ("user_prefs", runtime.context.user_id),
        "response_style"
    )
    style = pref.value if pref else "detailed"

    # 通过 runtime 访问并更新 State（由框架自动处理）
    # 这里只是示意，实际 State 更新在节点返回值中完成

    return f"{user_name}，{city}今天的天气是晴天（{style}模式）"

# ============================================================
# 第五步：创建 Agent
# ============================================================
agent = create_agent(
    model="openai:gpt-4o",               # 模型选择
    tools=[get_weather],                  # 工具列表
    state_schema=MyAgentState,            # 自定义 State 结构
    store=store,                          # 长期记忆存储
    context_schema=MyContext,             # 运行时上下文结构
    system_prompt="你是一个专业的天气助手",
)

# ============================================================
# 第六步：调用 Agent 时传入 Context
# ============================================================
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]},
    context={
        "user_id": "u_123456",
        "user_name": "张三",
        "is_premium": True,
    }
)
print(result["messages"][-1].content)
```

### 4.2 各参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | str | 是 | 模型标识符，支持 OpenAI、Anthropic 等 |
| `tools` | list[Tool] | 是 | Agent 可调用的工具列表 |
| `state_schema` | TypedDict | 否 | 自定义 State 结构，默认为内置 AgentState |
| `store` | BaseStore | 否 | 长期记忆存储后端 |
| `context_schema` | BaseModel | 否 | 运行时上下文的 Pydantic 模型 |
| `system_prompt` | str | 否 | 系统提示词 |

---

## 5. Runtime 数据流全景图

下面用一个完整的流程说明一次 Agent 调用中三者的协作过程：

```
用户请求: "帮我查一下北京天气"
                │
                ▼
     ┌──────────────────────┐
     │   1. 注入 Context    │
     │  user_id="u_123"    │
     │  is_premium=True    │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │  2. 初始化 State     │
     │  messages=[用户消息] │
     │  call_count=0       │
     │  errors=[]          │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────────┐
     │   3. LLM 决策节点       │
     │   读取 State + Context  │
     │   决定调用 get_weather  │
     └──────────┬───────────────┘
                │
                ▼
     ┌──────────────────────────┐
     │   4. 工具执行节点        │
     │   runtime.context.*     │ ◄── 读取 Context（用户身份）
     │   runtime.store.get()   │ ◄── 读取 Store（用户偏好）
     │   更新 State            │ ◄── 写入 State（call_count+1）
     └──────────┬───────────────┘
                │
                ▼
     ┌──────────────────────────┐
     │   5. LLM 生成回复        │
     │   结合工具结果生成答案   │
     │   更新 State.messages    │
     └──────────┬───────────────┘
                │
                ▼
     ┌──────────────────────────┐
     │   6. 可选：写回 Store    │
     │   保存交互经验/偏好     │
     └──────────┬───────────────┘
                │
                ▼
          返回结果给用户
```

---

## 6. 关键设计原则

### 6.1 最小权限原则

每个组件只应该访问它需要的数据：

| 组件 | 可访问 | 不可访问 |
|------|--------|----------|
| **LLM 节点** | State（messages）、Context（部分） | Store（原始数据）、Context（敏感字段） |
| **Tool 函数** | Context、Store（受限）、State（只读） | State（写入由框架管理） |
| **Middleware** | Context（完整）、State（可拦截修改） | Store（通常不需要） |

### 6.2 关注点分离原则

```
┌────────────────────────────────────────────────────┐
│                    关注点分离                       │
├──────────────┬───────────────┬──────────────────────┤
│   Context    │     State     │        Store         │
│   配置层     │    状态层     │       持久层         │
├──────────────┼───────────────┼──────────────────────┤
│ "我是谁？"   │ "我在做什么？"│ "我记得什么？"       │
│ "能用什么？" │ "做到哪了？"  │ "以前学到了什么？"   │
│ "有什么限制？"│ "出了什么错？"│ "用户喜欢什么？"     │
└──────────────┴───────────────┴──────────────────────┘
```

---

## 7. 本章小结

本章介绍了 LangChain Agent Runtime 的核心机制：

1. **Runtime 是 Agent 的运行基石**：`create_agent()` 底层基于 LangGraph Runtime 构建，理解 Runtime 等于掌握 Agent 的内部运作原理。

2. **三大核心概念各司其职**：State 管理单次对话的动态状态，Store 提供跨会话的长期记忆，Context 传递运行时的静态配置。三者清晰分离，避免状态污染和安全风险。

3. **正确分离带来显著收益**：通过合理划分数据的归属层级，可以降低 token 消耗、提升安全性、简化测试逻辑，并为后续的功能扩展预留空间。

> **下一节预告**：T02 将深入讲解如何自定义 State 结构，突破默认 AgentState 的局限，实现更丰富的状态管理能力。

---

## 参考来源

- 视频课程：黑马程序员 LangChain 教程 [BV178w1z7EHQ](https://www.bilibili.com/video/BV178w1z7EHQ)
- 配套文档：飞书文档 [IBPWwU7sUiuvbMkMYGacZxU8nsg](https://sid9kndrs6s.feishu.cn/wiki/IBPWwU7sUiuvbMkMYGacZxU8nsg)
- LangGraph 官方文档：https://langchain-ai.github.io/langgraph/concepts/runtime/
