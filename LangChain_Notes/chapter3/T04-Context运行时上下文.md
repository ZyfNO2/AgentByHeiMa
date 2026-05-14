# T04 - Context 运行时上下文

## 1. Context 是什么？

Context 是 Runtime 三大核心概念中的第三个，也是最容易被误解的一个。它的本质是：**运行期间不变的静态配置数据**。

如果说 State 是"草稿纸"（随时涂改），Store 是"档案柜"（长期保存），那么 Context 就是"身份证"——在整个请求过程中始终不变，标识着"这次运行的基本环境是什么"。

### 1.1 Context 的核心特征

| 特征 | 描述 |
|------|------|
| **不变性** | 在单次 invoke/stream 调用过程中不可被修改 |
| **注入时机** | 在调用 Agent 时由外部通过 `context` 参数传入 |
| **生命周期** | 随请求开始而创建，随请求结束而销毁 |
| **典型内容** | 用户身份、数据库连接、API Key、功能开关、租户信息 |
| **安全等级** | 最高——不会被发送给 LLM，仅在运行时可用 |
| **访问范围** | Tool 函数、Middleware、条件边（conditional edges）均可访问 |

### 1.2 Context 不是什么？

```
Context 不是 State：
├── Context 不会在节点间被修改和传递
├── Context 不参与状态机的状态转换
└── Context 不是用来存储业务计算结果的

Context 不是 Store：
├── Context 不会持久化到磁盘
├── Context 不跨会话保持
└── Context 不是用来存储用户偏好的

Context 是什么？
├── Context 是"我是谁、我在哪、我能用什么"的运行时宣言
├── Context 是安全地传递敏感配置信息的通道
└── Context 是 Middleware 动态调整行为的依据
```

---

## 2. 定义 ContextSchema

Context 的数据结构通过 Pydantic `BaseModel` 来定义，享受完整的类型校验和数据验证能力。

### 2.1 基本定义方式

```python
from pydantic import BaseModel, Field


class MyContext(BaseModel):
    """
    Agent 运行时上下文

    使用 Pydantic BaseModel 定义，获得：
    - 自动类型校验
    - 默认值支持
    - 序列化/反序列化
    - IDE 自动补全
    """

    # ======== 必填字段 ========
    user_id: str = Field(..., description="用户唯一标识符")
    user_name: str = Field(..., description="用户显示名称")

    # ======== 可选字段（带默认值） ========
    is_premium: bool = Field(default=False, description="是否为付费会员")
    user_role: str = Field(default="user", description="用户角色: admin/user/guest")

    # ======== 配置类字段 ========
    db_connection_str: str = Field(
        default="sqlite:///default.db",
        description="数据库连接字符串"
    )
    max_tool_calls: int = Field(default=10, ge=1, le=100, description="最大工具调用次数")
    enable_logging: bool = Field(default=True, description="是否启用详细日志")

    # ======== 功能开关 ========
    feature_flags: dict = Field(
        default_factory=lambda: {
            "experimental_feature": False,
            "beta_mode": False,
        },
        description="功能开关配置"
    )
```

### 2.2 Pydantic Field 的常用参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `default` | Any | 默认值 | `default="hello"` |
| `default_factory` | Callable | 默认值工厂（用于可变类型） | `default_factory=list` |
| `...` | Ellipsis | 标记为必填字段 | `Field(...)` |
| `description` | str | 字段描述（用于生成文档） | `description="用户ID"` |
| `ge` | number | 大于等于（Greater or Equal） | `ge=0` |
| `le` | number | 小于等于（Less or Equal） | `le=100` |
| `gt` | number | 严格大于 | `gt=0` |
| `lt` | number | 严格小于 | `lt=1000` |
| `pattern` | str | 正则匹配（字符串） | `pattern=r"^u_\w+$"` |

### 2.3 常见的 Context 模板

#### 多租户 SaaS 应用

```python
class SaasContext(BaseModel):
    """SaaS 多租户场景的 Context"""
    tenant_id: str                              # 租户 ID
    tenant_name: str                            # 租户名称
    user_id: str                                # 用户 ID
    user_role: str = "member"                   # owner/admin/member/guest
    plan_tier: str = "free"                     # free/pro/enterprise
    rate_limit_rpm: int = 60                    # 每分钟请求限制
    allowed_models: list[str] = ["gpt-4o-mini"] # 允许使用的模型列表
    data_region: str = "us-east-1"              # 数据区域
```

#### 企业内部工具

```python
class EnterpriseContext(BaseModel):
    """企业内部工具的 Context"""
    employee_id: str                            # 工号
    department: str                             # 部门
    security_clearance: int = 1                 # 安全等级 1-5
    cost_center: str                            # 成中心
    manager_id: str = ""                        # 直属主管工号
    working_hours_only: bool = True             # 是否仅在工作时间响应
    approval_required_for: list[str] = []       # 需要审批的操作类型
```

#### 个人助理

```python
class PersonalAssistantContext(BaseModel):
    """个人助理场景的 Context"""
    user_id: str
    device_type: str = "desktop"                # desktop/mobile/tablet
    location_timezone: str = "Asia/Shanghai"    # 时区
    current_time: str = ""                      # 当前时间（调用方注入）
    notification_enabled: bool = True           # 是否允许推送通知
    preferred_language: str = "zh-CN"           # 偏好语言
```

---

## 3. 在 createAgent 中使用 Context

### 3.1 创建时绑定 ContextSchema

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel


# ======== 第一步：定义 Context Schema ========
class WeatherAgentContext(BaseModel):
    user_id: str
    user_name: str
    location: str = "北京"          # 默认位置
    temperature_unit: str = "celsius"  # celsius / fahrenheit


# ======== 第二步：创建 Agent 并绑定 Context ========
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather_tool, get_forecast_tool],
    context_schema=WeatherAgentContext,   # 绑定 Context 结构定义
    system_prompt="你是专业的天气助手，根据用户的位置提供准确的天气信息。",
)
```

### 3.2 调用时传入 Context 数据

```python
# ======== 第三步：调用时注入 Context ========

# 方式一：invoke（同步调用）
result = agent.invoke(
    {"messages": [{"role": "user", "content": "今天天气怎么样？"}]},
    context={
        "user_id": "u_001",
        "user_name": "张三",
        "location": "上海",
        "temperature_unit": "celsius",
    }
)

# 方式二：ainvoke（异步调用）
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "明天会下雨吗？"}]},
    context={
        "user_id": "u_001",
        "user_name": "张三",
        "location": "上海",
    }
)

# 方式三：stream（流式调用）
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "这周末天气如何？"}]},
    context={
        "user_id": "u_001",
        "user_name": "张三",
        "location": "杭州",
    }
):
    print(chunk, end="")
```

### 3.3 Context 数据校验

由于 Context 基于 Pydantic 定义，传入不符合 schema 的数据时会自动触发校验错误：

```python
# 正确调用
agent.invoke(
    {"messages": [...]},
    context={"user_id": "u_001", "user_name": "张三"}  # OK
)

# 缺少必填字段 → ValidationError
agent.invoke(
    {"messages": [...]},
    context={"user_id": "u_001"}  # 缺少 user_name!
)
# pydantic.ValidationError: Field required [type=missing, input={'user_id': 'u_001'}, location=('user_name',)]

# 类型不匹配 → ValidationError
agent.invoke(
    {"messages": [...]},
    context={"user_id": 123, "user_name": "张三"}  # user_id 应该是 str!
)
# pydantic.ValidationError: Input should be a valid string [type=string_type, input_value=123, ...]
```

---

## 4. 在 Tool 中读取 Context

Tool 函数可以通过 `runtime` 参数访问 Context 数据，这使得工具可以根据用户身份、配置等信息做出差异化行为。

### 4.1 基本 Context 读取

```python
from langchain_core.tools import tool


@tool
def get_personalized_greeting(runtime):
    """
    获取个性化的问候语
    根据 Context 中的用户信息生成定制问候
    """
    # 从 runtime.context 读取用户信息
    name = runtime.context.user_name
    user_id = runtime.context.user_id
    is_premium = runtime.context.is_premium

    # 根据用户属性差异化响应
    if is_premium:
        greeting = f"尊敬的 VIP 会员 {name}，您好！很高兴为您服务。"
    else:
        greeting = f"您好，{name}！"

    return f"{greeting}（用户ID: {user_id}）"


@tool
def query_database(query: str, runtime):
    """
    查询数据库（使用 Context 中的连接信息）
    """
    # 从 Context 获取数据库配置
    db_conn = runtime.context.db_connection_str
    user_role = runtime.context.user_role

    # 权限检查
    if user_role != "admin" and "DELETE" in query.upper():
        return "错误：您没有执行删除操作的权限。"

    # 使用配置的连接字符串执行查询
    # (实际项目中这里会是真实的数据库操作)
    return f"使用连接 [{db_conn}] 执行查询: {query}"


@tool
def check_feature_access(feature_name: str, runtime):
    """
    检查用户是否有权使用某功能
    """
    flags = runtime.context.feature_flags

    if feature_name in flags:
        enabled = flags[feature_name]
        status = "已开启" if enabled else "未开启"
        return f"功能 [{feature_name}] 状态：{status}"
    else:
        return f"功能 [{feature_name}] 不存在于当前配置中。"
```

### 4.2 Context 在 Tool 中的典型用途

| 用途 | Context 字段 | Tool 中的使用方式 |
|------|-------------|------------------|
| **用户识别** | `user_id`, `user_name` | 个性化问候、操作审计 |
| **权限控制** | `user_role`, `security_clearance` | 条件性地允许/拒绝某些操作 |
| **资源路由** | `data_region`, `tenant_id` | 连接到正确的数据库分片 |
| **功能开关** | `feature_flags` | 启用/禁用实验性功能 |
| **限流配额** | `rate_limit_rpm`, `plan_tier` | 检查是否超出使用限额 |
| **环境适配** | `device_type`, `location_timezone` | 调整输出格式和时区显示 |

### 4.3 完整示例：带权限控制的文件操作工具

```python
"""
带权限控制的工具组示例
通过 Context 实现细粒度的访问控制
"""

from langchain_core.tools import tool
from datetime import datetime
import os


@tool
async def read_file(path: str, runtime):
    """
    读取文件内容

    Args:
        path: 文件路径
        runtime: 运行时对象
    """
    user_id = runtime.context.user_id
    role = runtime.context.user_role
    allowed_dirs = runtime.context.allowed_directories

    # 权限检查：管理员可以访问任意路径
    if role != "admin":
        # 非管理员只能访问白名单目录
        real_path = os.path.realpath(path)
        if not any(real_path.startswith(os.path.realpath(d)) for d in allowed_dirs):
            return f"权限不足：您只能在以下目录中操作：{allowed_dirs}"

    # 实际读取操作
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件内容（前500字符）：\n{content[:500]}"
    except FileNotFoundError:
        return f"错误：文件不存在 - {path}"
    except PermissionError:
        return f"错误：无读取权限 - {path}"


@tool
async def write_file(path: str, content: str, runtime):
    """
    写入文件内容

    Args:
        path: 文件路径
        content: 要写入的内容
        runtime: 运行时对象
    """
    role = runtime.context.user_role

    # 只允许管理员和编辑者写入
    if role not in ("admin", "editor"):
        return f"权限不足：您的角色 [{role}] 没有文件写入权限。"

    # 审计日志
    audit_log = {
        "user": runtime.context.user_id,
        "action": "WRITE",
        "path": path,
        "timestamp": datetime.now().isoformat(),
        "content_length": len(content),
    }
    print(f"[AUDIT] {audit_log}")  # 实际中应写入审计系统

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件：{path}（{len(content)} 字符）"
    except Exception as e:
        return f"写入失败：{e}"


@tool
async def list_directory(path: str = ".", runtime):
    """
    列出目录内容

    Args:
        path: 目录路径（默认当前目录）
        runtime: 运行时对象
    """
    user_id = runtime.context.user_id
    show_hidden = runtime.context.show_hidden_files

    try:
        entries = os.listdir(path)
        if not show_hidden:
            entries = [e for e in entries if not e.startswith(".")]

        return f"目录 [{path}] 的内容：\n" + "\n".join(
            f"  {'📁' if os.path.isdir(os.path.join(path, e)) else '📄'} {e}"
            for e in sorted(entries)
        )
    except PermissionError:
        return f"无权限访问目录：{path}"
    except FileNotFoundError:
        return f"目录不存在：{path}"


# 定义配套的 Context
class FileSystemContext(BaseModel):
    user_id: str
    user_role: str = "viewer"           # admin/editor/viewer
    allowed_directories: list[str] = ["/safe/workspace"]
    show_hidden_files: bool = False


# 组装 Agent
file_agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_file, write_file, list_directory],
    context_schema=FileSystemContext,
    system_prompt="你是一个文件管理助手。你可以帮助用户读取、写入和浏览文件。",
)


# 不同角色的调用示例
async def demo():
    # 管理员调用——拥有完全权限
    admin_result = await file_agent.ainvoke(
        {"messages": [{"role": "user", "content": "列出 /etc 的内容"}]},
        context={
            "user_id": "admin_01",
            "user_role": "admin",
            "allowed_directories": ["/"],
            "show_hidden_files": True,
        }
    )

    # 普通用户调用——受白名单限制
    user_result = await file_agent.ainvoke(
        {"messages": [{"role": "user", "content": "读取 /etc/passwd"}]},
        context={
            "user_id": "user_01",
            "user_role": "viewer",
            "allowed_directories": ["/safe/workspace"],
            "show_hidden_files": False,
        }
    )
```

---

## 5. 在 Middleware 中使用 Context

Middleware（中间件）是 Agent 请求处理管道中的钩子函数，可以利用 Context 实现强大的动态行为调整。

### 5.1 Middleware 概述

```
请求流入 → [Middleware 1] → [Middleware 2] → ... → Agent Core → ... → 响应流出
                                    ↑
                              可访问 Context
                              可修改 State
                              可决定是否继续/中断
```

### 5.2 基于 Context 的动态 System Prompt

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_dynamic_system_prompt(context):
    """
    根据 Context 动态构建 System Prompt

    这是 Middleware 的典型用法——根据用户属性调整 Agent 行为
    """
    base_prompt = "你是一个有用的 AI 助手。"

    # 根据用户角色添加特定指令
    role_instructions = {
        "admin": "\n你拥有管理员权限，可以帮助执行管理操作。",
        "developer": "\n你在与一名开发者对话，可以使用技术术语。",
        "beginner": "\n你在与一名新手对话，请用简单易懂的语言解释概念。",
    }

    instruction = role_instructions.get(context.user_role, "")

    # 根据会员等级调整详细程度
    detail_hint = ""
    if context.is_premium:
        detail_hint = "\nVIP 用户偏好详细的回答。"
    else:
        detail_hint = "\n请保持回答简洁。"

    return base_prompt + instruction + detail_hint


# 在 Middleware 中使用
def context_aware_middleware(state, config):
    """根据 Context 动态调整 System Prompt 的 Middleware"""
    context = config.get("configurable", {}).get("context")

    if context:
        dynamic_prompt = build_dynamic_system_prompt(context)
        # 将动态 prompt 注入到 state 中
        return {"system_prompt_override": dynamic_prompt}

    return {}
```

### 5.3 基于 Context 的条件性工具过滤

```python
def filter_tools_by_context(tools, context):
    """
    根据 Context 中的配置过滤可用工具

    用途：
    - 免费用户不能使用高级工具
    - 特定区域的用户不能使用某些数据源
    - Beta 功能仅对特定用户开放
    """
    filtered = []

    for tool in tools:
        # 检查功能开关
        if hasattr(tool, 'required_feature'):
            feature = tool.required_feature
            if not context.feature_flags.get(feature, False):
                continue  # 功能未开启，跳过此工具

        # 检查最低计划等级
        if hasattr(tool, 'min_plan_tier'):
            tier_order = {"free": 0, "pro": 1, "enterprise": 2}
            if tier_order.get(context.plan_tier, 0) < tier_order.get(tool.min_plan_tier, 0):
                continue  # 计划等级不够，跳过

        filtered.append(tool)

    return filtered


# 示例：标记工具的需求
@tool(required_feature="advanced_analysis", min_plan_tier="pro")
def deep_analysis(data: str):
    """深度数据分析（仅 Pro 及以上用户可用）"""
    # ...
    pass
```

### 5.4 基于 Context 的限流和配额管理

```python
import time
from collections import defaultdict

# 内存中的简单限流器（生产环境应使用 Redis）
_rate_limit_tracker = defaultdict(list)


def rate_limit_middleware(state, config):
    """
    基于 Context 的速率限制 Middleware

    检查用户是否超出了请求频率限制
    """
    context = config.get("configurable", {}).get("context")
    if not context:
        return {}

    user_id = context.user_id
    limit_rpm = getattr(context, 'rate_limit_rpm', 60)
    window = 60  # 60秒窗口

    now = time.time()
    requests = _rate_limit_tracker[user_id]

    # 清除窗口外的旧记录
    requests[:] = [t for t in requests if now - t < window]

    if len(requests) >= limit_rpm:
        wait_time = requests[0] + window - now
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"请求过于频繁。请在 {wait_time:.1f} 秒后重试。"
                }
            ],
            "_block": True  # 自定义标志，指示终止处理
        }

    # 记录本次请求
    requests.append(now)

    return {}  # 正常继续
```

---

## 6. Context vs State vs Store 选择决策树

在实际开发中，一个常见困惑是："这个数据到底该放哪里？" 下面提供一套清晰的决策流程。

### 6.1 决策树

```
你需要存储一段数据。请问：

① 这段数据会在运行过程中被修改吗？
│
├─ 是 → ★ 放入 State ★
│   （State 就是专门用来管理动态变化的业务状态的）
│
└─ 否 → ② 这段数据是配置/身份信息，还是需要跨会话记住的用户数据？
         │
         ├─ 是配置/身份信息 → ★ 放入 Context ★
         │   （API Key、用户ID、功能开关、数据库连接串等）
         │
         └─ 是需要跨会话记住的数据 → ★ 放入 Store ★
             （用户偏好、学习记录、历史经验等）
```

### 6.2 具体场景判断表

| 数据示例 | 会变化吗？ | 配置还是记忆？ | 放哪里？ |
|---------|-----------|---------------|---------|
| 对话消息列表 | 是（不断追加） | — | **State** |
| 当前执行步骤 | 是（逐步推进） | — | **State** |
| LLM 调用计数 | 是（每次+1） | — | **State** |
| 错误列表 | 是（可能追加） | — | **State** |
| 用户 ID | 否 | 身份 | **Context** |
| API Key | 否 | 配置 | **Context** |
| 数据库连接串 | 否 | 配置 | **Context** |
| 功能开关 | 否 | 配置 | **Context** |
| 用户喜欢的回答风格 | 否 | 需要跨会话记住 | **Store** |
| 用户纠正过的错误 | 否 | 需要跨会话记住 | **Store** |
| 历史成功的策略 | 否 | 需要跨会话记住 | **Store** |
| 当前时间戳 | 每次调用不同 | 配置（由调用方注入） | **Context** |

### 6.3 边界情况的讨论

#### 问：用户的"当前地理位置"放哪里？

```
答：取决于"当前"的定义

情况A：用户在本次对话中主动提供了位置，且本次对话中不会变
     → Context（作为用户身份的一部分传入）

情况B：Agent 通过 GPS 工具实时获取位置，每次调用可能不同
     → State（作为动态数据在节点间传递）

情况C：用户的"常住地址"（home address）
     → Store（跨会话持久化的用户画像数据）
```

#### 问：用户的"会员到期时间"放哪里？

```
答：Context

原因：
1. 到期时间本身不会在运行过程中被 Agent 修改
2. 它属于用户账户的配置信息
3. 每次调用时从数据库查询最新的传入即可
4. 如果放入 State，可能导致用过期的数据做决策
```

#### 问："本次对话中用户已经用了多少配额"放哪里？

```
答：State

原因：
1. 这个数字在对话过程中不断变化
2. 它是对话级别的临时统计
3. 但最终的累计值可能会定期同步到 Store（用于跨会话的配额管理）
```

---

## 7. 三者协作的完整架构

### 7.1 协作关系全景

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Runtime 完整架构                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    External World                        │   │
│  │                                                          │   │
│  │  用户请求 ──→ ┌──────────────┐                           │   │
│  │             │  Context 注入  │ ← 用户ID/API Key/配置     │   │
│  │             └──────┬───────┘                           │   │
│  └────────────────────┼────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Agent Graph（状态机）                      │  │
│  │                                                           │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐             │  │
│  │   │  Node A  │───▶│  Node B │───▶│  Node C │             │  │
│  │   └────┬────┘    └────┬────┘    └────┬────┘             │  │
│  │        │              │              │                   │  │
│  │        ▼              ▼              ▼                   │  │
│  │   ╔═══════════════════════════════════════╗              │  │
│  │   ║           State（状态池）              ║              │  │
│  │   ║  messages / call_count / errors / ... ║              │  │
│  │   ╚══════════════════╤═══════════════════╝              │  │
│  │                      │ 在节点间流动                        │  │
│  └──────────────────────┼──────────────────────────────────┘  │
│                         │                                      │
│  ┌──────────────────────┼──────────────────────────────────┐  │
│  │                        │                                 │  │
│  │  ┌────────────────────▼────────────────────────────┐   │  │
│  │  │              Store（长期记忆）                    │   │  │
│  │  │                                                 │   │  │
│  │  │  ("user_prefs", uid) → {style, language, ...}   │   │  │
│  │  │  ("learning", uid)   → {corrections, ...}       │   │  │
│  │  │  ("experience", uid) → {patterns, ...}          │   │  │
│  │  │                                                 │   │  │
│  │  │  Backend: SqliteStore / InMemoryStore / Custom  │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │            Context（运行时配置）                  │   │  │
│  │  │                                                  │   │  │
│  │  │  user_id / api_key / db_conn / feature_flags    │   │  │
│  │  │  （只读，全局可见，请求结束时销毁）                │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  数据流向：                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                                                       │   │
│  │  Context ──→ Tool/Middleware（只读访问）               │   │
│  │  State   ──↔→ Node（读写，驱动状态机）                 │   │
│  │  Store   ──↔→ Tool（按需读写，跨会话持久）             │   │
│  │                                                       │   │
│  │  Tool 可以同时访问：✓ Context  ✓ State(读)  ✓ Store   │   │
│  │  Node 可以同时访问：✓ Context  ✓ State(读写)          │   │
│  │  Middleware 可以访问：✓ Context  ✓ State(可拦截修改)   │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 各组件的访问权限矩阵

| 组件 | 读 Context | 写 Context | 读 State | 写 State | 读 Store | 写 Store |
|------|:---------:|:---------:|:-------:|:-------:|:--------:|:--------:|
| **LLM Node** | 部分 | 不可 | 可 | 可（返回值） | 通常不 | 通常不 |
| **Tool 函数** | 可 | 不可 | 可（只读） | 不可（由框架管） | 可 | 可 |
| **Middleware** | 可 | 不可 | 可 | 可（拦截修改） | 可 | 可 |
| **Conditional Edge** | 可 | 不可 | 可 | 不可 | 可 | 不可 |

### 7.3 数据流向的实际例子

```
场景：VIP 用户查询天气，Agent 记住他的偏好

Step 1: 用户发起请求
┌──────────────────────────────────────────────────┐
│ Request: "查询北京天气，以后都用简洁模式回答我"    │
│ Context: {user_id: "vip_001", is_premium: true}  │
└──────────────────────────┬───────────────────────┘
                           ▼
Step 2: Agent 进入 Router 节点
┌──────────────────────────────────────────────────┐
│ 读取 Context: is_premium=True → 选择高级模型      │
│ 读取 State: messages=[用户消息]                   │
│ 返回 State: {next_action: "call_weather_tool"}   │
└──────────────────────────┬───────────────────────┘
                           ▼
Step 3: Tool 执行节点 (get_weather)
┌──────────────────────────────────────────────────┐
│ 读取 Context: user_id="vip_001"                   │
│ 读取 Store: ("prefs","vip_001") → 发现无偏好记录  │
│ 执行: 调用天气API获取北京天气                     │
│ 返回: "北京今天晴，25°C"                          │
└──────────────────────────┬───────────────────────┘
                           ▼
Step 4: Agent 进入 Memory 节点 (remember 工具)
┌──────────────────────────────────────────────────┐
│ 读取 Context: user_id="vip_001"                   │
│ 写入 Store: ("prefs","vip_001", "style")          │
│          → {value: "concise", source: "user_req"} │
│ 返回: "已记住您的偏好：简洁模式 ✓"                │
└──────────────────────────┬───────────────────────┘
                           ▼
Step 5: LLM 生成最终回复
┌──────────────────────────────────────────────────┐
│ 读取 State: messages + 工具结果                   │
│ 读取 Context: is_premium=True → 详细程度=高       │
│ 生成: "北京今日晴，气温25°C，适宜外出。偏好已保存。"│
│ 更新 State: messages 追加 AI 回复                  │
└──────────────────────────┬───────────────────────┘
                           ▼
Step 6: 返回结果给用户
┌──────────────────────────────────────────────────┐
│ Response: "北京今日晴，气温25°C，适宜外出。         │
│           偏好已保存。"                            │
│ (Context 和 State 随请求结束而销毁)                │
│ (Store 中的偏好数据持久保留)                       │
└──────────────────────────────────────────────────┘

... 一周后，同一用户再次对话 ...

Step N: 用户发起新请求
┌──────────────────────────────────────────────────┐
│ Request: "上海明天天气如何？"                     │
│ Context: {user_id: "vip_001", is_premium: true}  │
│                                                  │
│ Agent 读取 Store: ("prefs","vip_001","style")     │
│            → 发现 {value: "concise"}              │
│ 自动应用简洁风格回答 ✓                            │
└──────────────────────────────────────────────────┘
```

---

## 8. 本章小结

本章深入讲解了 Runtime 中的最后一个核心概念——Context（运行时上下文）：

1. **Context 是不变的配置载体**：它在请求开始时注入，整个运行期间只读不被修改，随请求结束而销毁。适合放置用户身份、API Key、功能开关等静态配置数据。

2. **Pydantic BaseModel 定义 Schema**：Context 的结构通过 Pydantic 定义，享受自动类型校验、默认值、IDE 补全等便利。使用 `Field()` 可以添加约束条件和描述信息。

3. **Tool 和 Middleware 都能访问 Context**：Tool 通过 `runtime.context` 读取，Middleware 通过 `config` 参数获取。可用于权限控制、个性化行为、动态 Prompt、限流等多种场景。

4. **三者选择的决策树**：数据会变 → State；数据不变且是配置/身份 → Context；数据不变但需跨会话记住 → Store。清晰的边界划分是构建健壮 Agent 的基础。

5. **三者协同构成完整架构**：Context 注入配置，State 驱动状态机，Store 提供持久记忆。各组件有不同的访问权限，共同支撑起一个既有记忆能力又安全可控的 Agent 系统。

---

## 第3章（Agent 进阶）总回顾

至此，T01-T04 已经完整覆盖了 LangChain Agent Runtime 的三大核心概念：

| 主题 | 核心内容 | 关键能力 |
|------|----------|----------|
| **T01 - Runtime 总览** | State / Store / Context 三大概念的定位与关系 | 理解 Agent 运行的底层架构 |
| **T02 - State 自定义** | TypedDict 定义、NotRequired 字段、节点读写、Reducer | 扩展 Agent 的状态管理能力 |
| **T03 - Store 长期记忆** | InMemoryStore / SqliteStore、命名空间设计、记忆模式 | 实现 Agent 的跨会话记忆能力 |
| **T04 - Context 上下文** | Pydantic Schema、Tool/Middleware 中的使用、权限控制 | 安全传递配置，实现差异化行为 |

这三者的协同工作是构建**生产级 Agent** 的基石——没有它们，Agent 只不过是一个无状态的简单对话机器人而已。

---

## 参考来源

- 视频课程：黑马程序员 LangChain 教程 [BV178w1z7EHQ](https://www.bilibili.com/video/BV178w1z7EHQ)
- 配套文档：飞书文档 [IBPWwU7sUiuvbMkMYGacZxU8nsg](https://sid9kndrs6s.feishu.cn/wiki/IBPWwU7sUiuvbMkMYGacZxU8nsg)
- LangGraph Context 文档：https://langchain-ai.github.io/langgraph/concepts/runtime/
- Pydantic 官方文档：https://docs.pydantic.dev/latest/
