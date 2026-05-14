# T03 - Store 长期记忆

## 1. Store vs State 的本质区别

在学习 Store 之前，最关键的问题是理解它与 State 的本质差异。很多初学者容易混淆这两个概念，因为它们都可以用来"记住"数据。但从架构视角来看，二者服务于完全不同的目的。

### 1.1 核心差异对照表

| 维度 | State | Store |
|------|-------|-------|
| **作用域** | 单次对话（single thread） | 跨所有对话（all threads） |
| **生命周期** | 随请求结束而销毁 | 持久化到磁盘/数据库，进程重启后仍存在 |
| **持久化方式** | 内存中（in-memory） | SQLite / PostgreSQL / Redis / 自定义后端 |
| **典型内容** | 对话历史、当前任务状态、中间计算结果 | 用户偏好、学习记录、失败经验、用户画像 |
| **访问方式** | `state["key"]` / `state.get("key")` | `runtime.store.get()` / `runtime.store.put()` / `runtime.store.search()` |
| **写入频率** | 高频（每个节点都可能更新） | 低频（按需写入） |
| **数据量** | 小（单次对话的上下文） | 可大可小（取决于积累量） |
| **一致性要求** | 强一致（单线程顺序执行） | 最终一致（允许短暂延迟） |
| **类比** | 工作记忆 / 草稿纸 | 长期记忆 / 档案柜 |

### 1.2 形象类比：人的记忆系统

```
┌─────────────────────────────────────────────────────────┐
│                   人类记忆系统类比                         │
├──────────────────────┬──────────────────────────────────┤
│                      │                                   │
│   ┌──────────────┐   │   ┌──────────────────────────┐   │
│   │   感觉记忆    │   │   │      长期记忆 (Store)     │   │
│   │  (Context)    │   │   │                          │   │
│   │               │   │   │  ┌────────────────────┐  │   │
│   │  "我现在是谁"  │   │   │  │  陈述性记忆         │  │   │
│   │  "环境怎样"   │   │   │  │  (用户偏好/画像)    │  │   │
│   └──────────────┘   │   │  ├────────────────────┤  │   │
│                      │   │  │  程序性记忆         │  │   │
│   ┌──────────────┐   │   │  │  (技能/习惯)       │  │   │
│   │   工作记忆    │   │   │  ├────────────────────┤  │   │
│   │   (State)    │   │   │  │  经验记忆           │  │   │
│   │               │   │   │  │  (成功/失败经验)   │  │   │
│   │  "我在做什么"  │   │   │  └────────────────────┘  │   │
│   │  "做到哪了"   │   │   │                          │   │
│   └──────────────┘   │   └──────────────────────────┘   │
│                      │                                   │
└──────────────────────┴──────────────────────────────────┘
```

- **Context** = 感觉记忆：当前时刻的环境感知（我是谁、我在哪），转瞬即逝但必需
- **State** = 工作记忆：正在处理的任务的临时信息（类似于心算时的草稿纸）
- **Store** = 长期记忆：沉淀下来的知识和经验（类似于人脑中储存的技能和事实）

---

## 2. 为什么需要长期记忆？

没有长期记忆的 Agent 就像一个永远失忆的人——每次对话都是初次见面，无法从过去的交互中学习和改进。

### 2.1 三大核心需求

#### 需求一：个性化偏好记忆

```
第一次对话：
  用户: "以后回答尽量简短一点，不要太啰嗦"
  Agent: "好的，我已经记住了您的偏好 ✓"

... 一周后 ...

第N次对话：
  用户: "解释一下量子计算"
  Agent: "量子计算利用量子比特...（简洁版，3句话讲清）"  ← 自动应用偏好
```

没有 Store，Agent 无法在第二次对话中记住用户的偏好设置。

#### 需求二：积累式学习（从错误中学习）

```
第一次对话：
  用户: "我的名字叫李明，不叫小明！"
  Agent: "抱歉！已更正：您叫【李明】 ✓"

... 后续所有对话 ...

第N次对话：
  User: "你好"
  Agent: "您好，李明！有什么可以帮您的？"  ← 再也不会叫错
```

#### 需求三：跨会话的连续体验

```
项目进度追踪场景：

Day 1:
  用户: "我在做一个电商推荐系统的项目"
  Agent: "记下了！项目：电商推荐系统 ✓"

Day 5:
  用户: "上次那个项目的进展怎么样了？"
  Agent: "您提到的【电商推荐系统】项目，
         上次讨论还处于需求分析阶段。
         目前有什么新的进展吗？"  ← 跨会话记忆关联
```

### 2.2 技术价值总结

| 能力 | 没有 Store | 有 Store |
|------|-----------|----------|
| 用户偏好 | 每次重新告知 | 自动记住并应用 |
| 错误纠正 | 重复犯错 | 一次纠正，永久生效 |
| 上下文延续 | 每次从头开始 | 自然衔接前文 |
| 用户画像 | 无法建立 | 随交互不断丰富 |
| 个性化体验 | 千篇一人 | 因人而异 |

---

## 3. Store 的基础 API

LangGraph 的 Store 提供了一组简洁的 CRUD 操作接口，支持同步和异步两种调用方式。

### 3.1 核心 API 速查

| 操作 | 同步方法 | 异步方法 | 说明 |
|------|----------|----------|------|
| **存入** | `store.put(namespace, key, value)` | `await store.aput(...)` | 写入或更新一条数据 |
| **读取** | `store.get(namespace, key)` | `await store.aget(...)` | 读取一条数据 |
| **删除** | `store.delete(namespace, key)` | `await store.adelete(...)` | 删除一条数据 |
| **搜索** | `store.search(namespace, filter)` | `await store.asearch(...)` | 按条件搜索数据 |
| **批量操作** | `store.put_many(items)` | `await store.aput_many(...)` | 批量写入 |

### 3.2 基本使用模式

```python
from langgraph.store.memory import InMemoryStore

# 创建 Store 实例
store = InMemoryStore()

# ======== 存入数据 (put) ========
# put(namespace, key, value) → 返回 StoreItem
item = store.put(
    ("user_prefs", "user_123"),     # namespace: 元组形式的命名空间
    "response_style",               # key: 数据项的键名
    {"style": "concise", "length": "short"},  # value: 要存储的数据（通常是 dict）
)
print(f"写入成功，key={item.key}")

# ======== 读取数据 (get) ========
# get(namespace, key) → 返回 StoreItem 或 None
result = store.get(("user_prefs", "user_123"), "response_style")

if result:
    print(f"读取成功: {result.value}")
    # result.value == {"style": "concise", "length": "short"}
else:
    print("数据不存在")

# ======== 删除数据 (delete) ========
store.delete(("user_prefs", "user_123"), "response_style")
print("删除成功")

# 验证删除
result = store.get(("user_prefs", "user_123"), "response_style")
print(result)  # None
```

### 3.3 StoreItem 的结构

无论是 `put` 还是 `get` 返回的都是 `StoreItem` 对象，其结构如下：

```python
@dataclass
class StoreItem:
    namespace: tuple       # 命名空间元组，如 ("user_prefs", "user_123")
    key: str               # 数据项的键名，如 "response_style"
    value: dict            # 实际存储的数据
    created_at: str        # 创建时间（ISO 格式）
    updated_at: str        # 最后更新时间（ISO 格式）
```

---

## 4. InMemoryStore（内存存储）

### 4.1 概述与适用场景

`InMemoryStore` 是 LangGraph 提供的最简单的 Store 实现，所有数据保存在进程内存中。

| 特性 | 说明 |
|------|------|
| **后端** | Python 字典 (`dict`) |
| **持久化** | 无（进程重启后数据丢失） |
| **性能** | 极高（内存读写） |
| **并发** | 不支持多进程共享 |
| **适用场景** | 开发、测试、原型验证、一次性脚本 |

### 4.2 完整使用示例

```python
"""
InMemoryStore 完整使用示例
适用于开发/测试环境
"""

import asyncio
from langgraph.store.memory import InMemoryStore


async def demo_inmemory_store():
    store = InMemoryStore()

    # ---- 1. 写入用户偏好 ----
    await store.aput(
        ("user_prefs", "u_001"),
        "display_settings",
        {"theme": "dark", "language": "zh-CN", "font_size": 14}
    )

    await store.aput(
        ("user_prefs", "u_001"),
        "response_style",
        {"tone": "professional", "detail_level": "medium"}
    )

    # ---- 2. 写入学习记录 ----
    await store.aput(
        ("learning", "u_001"),
        "corrections",
        {
            "correction_count": 3,
            "items": [
                {"date": "2025-01-15", "wrong": "叫用户小明", "correct": "叫用户李明"},
                {"date": "2025-02-20", "wrong": "用英文回答", "correct": "用中文回答"},
            ]
        }
    )

    # ---- 3. 读取单个条目 ----
    style = await store.aget(("user_prefs", "u_001"), "response_style")
    print(f"用户风格偏好: {style.value}")

    # ---- 4. 搜索命名空间下的所有数据 ----
    results = await store.asearch(("user_prefs", "u_001"))
    print(f"\n用户 u_001 的所有偏好设置:")
    for item in results:
        print(f"  [{item.key}] = {item.value}")

    # ---- 5. 更新数据（put 相同 key 会覆盖）----
    await store.aput(
        ("user_prefs", "u_001"),
        "response_style",
        {"tone": "casual", "detail_level": "brief"}  # 完全替换
    )

    updated = await store.aget(("user_prefs", "u_001"), "response_style")
    print(f"\n更新后的风格: {updated.value}")
    # {"tone": "casual", "detail_level": "brief"}

    # ---- 6. 删除数据 ----
    await store.adelete(("user_prefs", "u_001"), "display_settings")
    deleted = await store.aget(("user_prefs", "u_001"), "display_settings")
    print(f"删除后查询: {deleted}")  # None


# 运行示例
if __name__ == "__main__":
    asyncio.run(demo_inmemory_store())
```

### 4.3 InMemoryStore 的局限性

```
局限性清单：
├── 进程重启 → 全部丢失（最大的问题）
├── 内存占用 → 数据量大时有 OOM 风险
├── 无法跨进程共享 → 多 worker 部署时各自独立
├── 无法跨机器共享 → 分布式部署不可用
└── 无查询索引 → search 是全量扫描，大数据量下性能差

结论：仅适合开发和测试，生产环境请使用 SqliteStore 或其他持久化方案
```

---

## 5. SqliteStore（SQLite 持久化）

### 5.1 概述与优势

`SqliteStore` 将数据持久化到 SQLite 数据库文件中，是生产环境单机部署的首选方案。

| 特性 | 说明 |
|------|------|
| **后端** | SQLite 数据库文件 (.db) |
| **持久化** | 是（数据写入磁盘文件） |
| **性能** | 高（SQLite 本身非常轻量高效） |
| **并发** | 支持（SQLite 的 WAL 模式允许多读） |
| **依赖** | 无需安装额外的数据库服务 |
| **适用场景** | 生产环境、单机部署、边缘设备 |

### 5.2 基本使用

```python
from langgraph.store.sqlite import SqliteStore

# 创建 SQLite Store（数据持久化到 .db 文件）
store = SqliteStore(
    conn_string="./my_agent_store.db"   # 数据库文件路径
)

# API 与 InMemoryStore 完全一致
await store.aput(("prefs", "user_1"), "theme", {"value": "dark"})
result = await store.aget(("prefs", "user_1"), "theme")
print(result.value)  # {"value": "dark"}

# 重启程序后数据依然存在！
```

### 5.3 SqliteStore vs InMemoryStore 对比

| 维度 | InMemoryStore | SqliteStore |
|------|--------------|-------------|
| **持久化** | 否 | 是 |
| **重启后数据** | 丢失 | 保留 |
| **额外依赖** | 无 | 无（Python 内置 sqlite3） |
| **写入速度** | ~0.01ms | ~1-5ms（磁盘 I/O） |
| **读取速度** | ~0.001ms | ~0.1-1ms |
| **数据容量** | 受限于内存 | 受限于磁盘（可达 TB 级） |
| **适合阶段** | 开发/测试 | 生产/预发布 |

---

## 6. Store 的命名空间设计

命名空间（Namespace）是 Store 中组织数据的核心机制，类似于文件系统中的目录结构。

### 6.1 命名空间的结构

Store 中的每条数据通过 `(namespace, key)` 二元组唯一定位：

```
namespace: ("user_prefs", "user_123")   ← 元组，可以有多层
key:       "response_style"             ← 字符串
value:     {"style": "concise"}         ← 字典（通常）
```

### 6.2 推荐的命名空间分层策略

```
顶层分类（第一层）          含义
─────────────────────────────────────────
("user_prefs", uid)       用户偏好设置
("user_profile", uid)     用户画像信息
("learning", uid)         学习/纠正记录
("experience", uid)       成功经验积累
("session_history", uid)  历史会话摘要
("project", pid)          项目相关数据
("team", tid)             团队/组织数据
("_system", _)            系统内部数据（下划线开头）
```

### 6.3 命名空间设计最佳实践

```python
# ======== 好的设计示例 ========

# 用户维度的数据——以 user_id 作为第二层
namespace_user_prefs = ("user_prefs", "user_abc123")
namespace_user_learning = ("learning", "user_abc123")

# 项目维度的数据——以 project_id 作为第二层
namespace_project = ("project", "proj_001")

# 系统级数据——固定标识
namespace_system_config = ("system", "global_config")


# ======== 避免的反模式 ========

# 反模式一：扁平命名空间（所有数据混在一起）
# ("data", "user_123_prefs")     ← 难以区分类型
# ("data", "user_123_learning")  ← 搜索时噪音太多

# 反模式二：过深的嵌套（超过 3 层）
# ("org", "dept", "team", "project", "task", "subtask_1")  ← 过于复杂

# 反模式三：命名不一致
# ("user-prefs", "u123")   ← 用了连字符
# ("userPrefs", "u456")    ← 用了驼峰
# ("user_preferences", "789")  ← 又换了一种
# 应统一使用 snake_case
```

### 6.4 命名空间的搜索能力

```python
# 搜索特定命名空间下的所有数据
results = await store.asearch(("user_prefs", "user_123"))
for item in results:
    print(f"{item.key}: {item.value}")

# 可以对搜索结果进行过滤
results = await store.asearch(
    ("learning", "user_123"),
    filter=lambda item: "correction" in item.key
)
```

---

## 7. 在 Tool 中访问 Store

在实际开发中最常见的场景是在 Tool 函数中读写 Store，让 Agent 具备真正的"记忆能力"。

### 7.1 基本访问方式

LangChain 的 Tool 函数可以通过 `runtime` 参数获取对 Store 的访问权：

```python
from langchain_core.tools import tool

@tool
async def save_user_preference(preference: str, category: str, runtime):
    """
    保存用户偏好设置到长期记忆

    Args:
        preference: 偏好值
        category: 偏好类别（如 style/theme/language）
        runtime: LangGraph 运行时对象（自动注入）
    """
    # 从 Context 中获取当前用户 ID
    user_id = runtime.context.get("user_id", "anonymous")

    # 写入 Store
    await runtime.store.aput(
        ("user_prefs", user_id),
        category,
        {"value": preference, "updated_at": "2025-01-20T10:00:00"}
    )

    return f"已保存您的偏好设置：[{category}] = {preference}"


@tool
async def recall_user_preference(category: str, runtime):
    """
    从长期记忆中回忆用户的偏好设置

    Args:
        category: 要查询的偏好类别
        runtime: LangGraph 运行时对象（自动注入）
    """
    user_id = runtime.context.get("user_id", "anonymous")

    # 从 Store 读取
    item = await runtime.store.aget(("user_prefs", user_id), category)

    if item and item.value:
        return f"根据我的记忆，您的[{category}]偏好是：{item.value['value']}"
    else:
        return f"我没有找到关于[{category}]的偏好记录。您可以告诉我您的偏好，我会记住的。"
```

### 7.2 完整的记忆型 Tool 组合

```python
"""
完整的"记忆工具组"示例
包含：记住、回忆、遗忘 三个工具
"""

from langchain_core.tools import tool
from datetime import datetime


@tool
async def remember(fact: str, category: str = "general", runtime):
    """
    记住用户告诉你的事实或偏好。

    当用户说"记住..."、"下次..."、"我喜欢..."、"我不喜欢..."时使用此工具。

    Args:
        fact: 要记住的内容
        category: 记忆分类（如 preference/experience/fact）
        runtime: 运行时对象
    """
    user_id = runtime.context.get("user_id", "default")

    # 获取已有的记忆列表
    existing = await runtime.store.aget(("memory", user_id), category)

    memories = existing.value.get("items", []) if existing else []

    # 追加新记忆
    new_memory = {
        "content": fact,
        "timestamp": datetime.now().isoformat(),
    }
    memories.append(new_memory)

    # 写回 Store
    await runtime.store.aput(
        ("memory", user_id),
        category,
        {"items": memories, "count": len(memories)}
    )

    return f"好的，我已经记住了：「{fact}」（归类为 {category}，共 {len(memories)} 条记忆）"


@tool
async def recall(category: str = "all", runtime):
    """
    回忆之前记住的内容。

    当用户问"你还记得吗"、"我之前说过什么"、"我的偏好是什么"时使用此工具。

    Args:
        category: 要回忆的分类，"all" 表示全部
        runtime: 运行时对象
    """
    user_id = runtime.context.get("user_id", "default")

    if category == "all":
        # 搜索该用户的所有记忆
        results = await runtime.store.asearch(("memory", user_id))
        output_parts = []
        for item in results:
            memories = item.value.get("items", [])
            for m in memories:
                output_parts.append(f"[{item.key}] {m['content']} ({m['timestamp'][:10]})")

        if output_parts:
            return "我记住的内容有：\n" + "\n".join(output_parts)
        else:
            return "我还没有记住关于您的任何事情。"
    else:
        item = await runtime.store.aget(("memory", user_id), category)
        if item and item.value.get("items"):
            memories = item.value["items"]
            content = "\n".join(f"- {m['content']}" for m in memories)
            return f"[{category}] 分类下的记忆：\n{content}\n共 {len(memories)} 条"
        else:
            return f"我没有在 [{category}] 分类下找到任何记忆。"


@tool
async def forget(fact: str, runtime):
    """
    忘记指定的记忆内容。

    当用户说"忘了这个"、"删除记忆"、"不要再记住"时使用此工具。

    Args:
        fact: 要忘记的内容（支持模糊匹配）
        runtime: 运行时对象
    """
    user_id = runtime.context.get("user_id", "default")
    results = await runtime.store.asearch(("memory", user_id))

    forgotten = []
    for item in results:
        memories = item.value.get("items", [])
        filtered = [m for m in memories if fact.lower() not in m["content"].lower()]

        removed_count = len(memories) - len(filtered)
        if removed_count > 0:
            await runtime.store.aput(
                ("memory", user_id),
                item.key,
                {"items": filtered, "count": len(filtered)}
            )
            forgotten.append(f"[{item.key}] 删除了 {removed_count} 条记忆")

    if forgotten:
        return "已完成遗忘操作：\n" + "\n".join(forgotten)
    else:
        return f"没有找到包含「{fact}」的记忆，可能本来就未曾记住。"


# 将三个工具组合使用
memory_tools = [remember, recall, forget]
```

### 7.3 在 createAgent 中集成

```python
from langchain.agents import create_agent
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

class AgentContext(BaseModel):
    user_id: str

# 创建持久化 Store
store = SqliteStore(conn_string="./agent_memory.db")

# 创建带有记忆能力的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=memory_tools,           # 记忆工具组
    store=store,                  # 长期记忆后端
    context_schema=AgentContext,  # 用于识别用户身份
    system_prompt="""
    你是一个有长期记忆能力的助手。
    你可以通过 remember/recall/forget 工具来管理你对用户的记忆。
    当用户提供个人信息或偏好时，主动使用 remember 工具记住。
    """,
)

# 调用时传入用户身份
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "记住我喜欢喝咖啡"}]},
    context={"user_id": "user_zhangsan"},
)
```

---

## 8. 长期记忆设计模式

根据不同的业务需求，Store 中的数据可以按照几种经典模式来组织。

### 8.1 三大记忆模式

| 模式 | 目的 | 典型内容 | 保留策略 |
|------|------|----------|----------|
| **偏好记忆** (Preference Memory) | 个性化用户体验 | 语言风格、回答长度、主题兴趣 | 长期保留，用户可修改 |
| **学习记忆** (Learning Memory) | 从错误中持续改进 | 纠正过的错误、用户反馈的负面案例 | 长期保留，定期清理过期项 |
| **经验记忆** (Experience Memory) | 积累成功策略 | 有效的问题解决路径、高效的工具组合 | 定期评估，淘汰低质量项 |

### 8.2 各模式的 Schema 设计

#### 偏好记忆 Schema

```python
# Namespace: ("prefs", user_id)

preference_schemas = {
    "communication": {
        "language": "zh-CN",           # 语言偏好
        "style": "concise",            # concise / detailed / casual / formal
        "max_paragraphs": 3,           # 最大段落数
        "use_emoji": False,            # 是否使用表情符号
        "technical_depth": "medium",   # low / medium / high
    },

    "display": {
        "format": "markdown",          # 输出格式
        "include_sources": True,       # 是否包含引用来源
        "code_theme": "dark",          # 代码块主题
    },

    "interests": {
        "topics": ["AI", "编程", "投资"],  # 兴趣话题
        "avoid_topics": ["娱乐八卦"],      # 避免的话题
        "domain_expertise": "软件开发",     # 用户专业领域
    },
}
```

#### 学习记忆 Schema

```python
# Namespace: ("learning", user_id)

learning_schema = {
    "corrections": {
        "items": [
            {
                "date": "2025-01-15",
                "trigger": "用户说'我叫李明不叫小明'",
                "wrong_behavior": "称呼用户为小明",
                "correct_behavior": "称呼用户为李明",
                "confidence": 1.0,          # 确信度（0~1）
                "times_applied": 5,          # 已应用次数
            }
        ],
        "total_corrections": 12,
    },

    "negative_feedback": {
        "items": [
            {
                "date": "2025-02-01",
                "feedback": "回答太长了",
                "category": "verbosity",
                "action_taken": "切换到 concise 模式",
            }
        ],
    },
}
```

#### 经验记忆 Schema

```python
# Namespace: ("experience", user_id) 或 ("experience", "global")

experience_schema = {
    "successful_patterns": {
        "items": [
            {
                "pattern_id": "pat_001",
                "scenario": "数据分析请求",
                "approach": "先 search_database → 再 calculate → 最后 summarize",
                "success_rate": 0.95,
                "avg_time_saved": "40%",
                "usage_count": 23,
                "last_used": "2025-03-10",
            }
        ],
    },

    "tool_efficiency": {
        "tool_name": "fetch_api",
        "total_calls": 150,
        "success_rate": 0.87,
        "avg_latency_ms": 520,
        "common_failures": ["timeout", "connection_refined"],
        "recommended_alternative": "cached_fetch_api",
    },
}
```

### 8.3 记忆的生命周期管理

```
                    时间轴 →

  创建 ──→ 活跃使用 ──→ 冷却期 ──→ 归档/清理
   │         │            │           │
   │         │            │           ├─→ 迁移到冷存储
   │         │            │           └─→ 直接删除（低价值数据）
   │         │            │
   │         │            └─→ 降低权重（排序时靠后）
   │         │
   │         └─→ 高频访问，保持热状态
   │
   └─→ 初始权重 = 1.0

管理策略：
├── 偏好记忆：永不自动删除，仅用户主动修改
├── 学习记忆：90天未触发的纠正项降权，180天归档
├── 经验记忆：成功率低于60%的模式自动淘汰
└── 定期（每月）运行清理脚本压缩存储
```

---

## 9. 本章小结

本章全面介绍了 LangGraph Store（长期记忆）机制：

1. **Store 与 State 本质不同**：State 是单次对话的工作记忆，Store 是跨会话持久化的长期记忆。选择哪种取决于数据的生命周期需求。

2. **两种主要存储后端**：`InMemoryStore` 适合开发测试（零配置、高性能但不持久），`SqliteStore` 适合生产环境（持久化、轻量、无额外依赖）。

3. **命名空间设计是关键**：采用 `(namespace_tuple, key)` 的二元组寻址方式，按用户/功能/类型分层组织数据，避免冲突和混乱。

4. **Tool 中访问 Store 很自然**：通过 `runtime.store` API 在工具函数中读写长期记忆，配合 `remember/recall/forget` 模式可以实现真正有记忆能力的 Agent。

5. **三种经典记忆模式**：偏好记忆（个性化）、学习记忆（纠错改进）、经验记忆（策略积累），各有适用的 Schema 设计和管理策略。

> **下一节预告**：T04 将介绍 Context（运行时上下文），学习如何安全地传递配置数据和用户身份信息。

---

## 参考来源

- 视频课程：黑马程序员 LangChain 教程 [BV178w1z7EHQ](https://www.bilibili.com/video/BV178w1z7EHQ)
- 配套文档：飞书文档 [IBPWwU7sUiuvbMkMYGacZxU8nsg](https://sid9kndrs6s.feishu.cn/wiki/IBPWwU7sUiuvbMkMYGacZxU8nsg)
- LangGraph Store 文档：https://langchain-ai.github.io/langgraph/concepts/memory/
- SQLite 官方文档：https://www.sqlite.org/docs.html
