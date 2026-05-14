# 13-LangChain入门-认识LangChain

> 视频来源：[黑马程序员 2026最新版LangChain+LangGraph开发实战](https://www.bilibili.com/video/BV178w1z7EHQ?p=14)
> 时长：12:52 | 章节：第2章 LangChain入门 → 第1节 LangChain核心组件

---

## 1. 什么是 LangChain

### 1.1 一句话定义

**LangChain 是一个用于开发 AI 智能体（Agent）的应用框架/平台。**

它由 **Harrison Chase** 于 **2022年10月** 创建（比 ChatGPT 发布还早一个月），核心理念是：

> **把 LLM 相关的各种组件"链接"（Chain）在一起，简化 LLM 应用开发的难度。**

### 1.2 官方定位

| 项目 | 说明 |
|------|------|
| **官网** | https://www.langchain.com/ |
| **官方定义** | LangChain provides the engineering platform and open source frameworks developers use to build, test, and deploy reliable AI agents. |
| **文档** | https://docs.langchain.com/ |
| **GitHub Stars** | 90,000+（截至2026年） |
| **月下载量** | 数百万次 |

### 1.3 核心理解

```
LangChain ≠ 大语言模型（LLM）
LangChain = LLM 的"操作系统" / "开发工具箱"
```

**类比理解**：
- LLM 就像一个只有大脑的"天才"——语言能力超强，但：
  - ❌ 无法读取你的本地文件
  - ❌ 无法调用外部 API/工具
  - ❌ 无法记住多轮对话上下文
  - ❌ 无法标准化地复用提示词和业务逻辑
- **LangChain 就是给这个天才装上手脚、眼睛、耳朵和记忆**
- 让你用最少的代码，快速搭建复杂的 AI 应用

---

## 2. 为什么需要 LangChain？

### 2.1 直接调 LLM API 的痛点

| 痛点 | 具体表现 |
|------|----------|
| **失忆** | 单次对话结束上下文清零，无法多轮连贯交互 |
| **知识孤岛** | 被困在预训练数据中，无法访问本地文档/数据库 |
| **行动无力** | 只能输出文本，无法调用 API、查天气、操作文件 |
| **开发繁琐** | 每次搭建应用都要重复写"对话管理、数据加载、工具调用" |
| **模型锁定** | 各厂商 API 格式不同，切换模型成本高 |

### 2.2 LangChain 的解决方案

| 场景 | 直接调 API | LangChain 方案 |
|------|-----------|---------------|
| 多轮对话 + 工具调用 | 手动维护循环逻辑 | `create_agent` 自动管理 |
| 读取 PDF/网页 | 自己实现文档切分、向量化 | `Document Loaders` + `Text Splitters` |
| 调用外部 API | 硬编码函数调用逻辑 | `@tool` 装饰器自动注册 |
| 复杂业务流程 | 多个大模型调用，代码耦合高 | LangGraph 有向图编排 |
| 模型切换 | 各厂商 API 格式不同 | 统一接口，一行代码切换 |

---

## 3. LangChain 架构体系

### 3.1 不是单一框架，而是完整平台

LangChain 并不仅仅是一个 Python 包，而是一整个**智能体开发生态系统**，包含多个层级：

```
┌─────────────────────────────────────┐
│         应用层 Application          │
│    LangGraph（Agent 编排与状态管理） │
├─────────────────────────────────────┤
│         组件层 Components           │
│  Chains · Agents · Retrieval · Tools │
├─────────────────────────────────────┤
│         集成层 Integration          │
│   第三方模型 · 向量库 · 工具 · API   │
│      (langchain-community)          │
├─────────────────────────────────────┤
│         核心层 Core                 │
│   基础抽象 · LCEL · Runnable 接口    │
│       (langchain-core)              │
└─────────────────────────────────────┘
```

### 3.2 分层架构详解

| 层级 | 包名 | 职责 |
|------|------|------|
| **langchain-core** | 基础抽象与 LCEL | 组件协同的核心，所有组件的基类定义 |
| **langchain-community** | 社区集成 | 第三方模型、工具、向量库等 200+ 集成 |
| **langchain** | 核心框架 | Chains、Agents、Retrieval 等核心业务组件 |
| **langgraph** | 工作流引擎 | 编排多个节点，负责工作流调度与状态跳转 |

### 3.3 核心包安装策略

LangChain 2026 采用**按需安装**策略：

```bash
# 核心 + OpenAI
pip install langchain "langchain[openai]"

# 核心 + Anthropic
pip install langchain "langchain[anthropic]"

# 核心 + Google Gemini
pip install langchain "langchain[google-genai]"

# 本地模型（Ollama）
pip install langchain langchain-ollama

# 全功能（学习环境推荐）
pip install langchain "langchain[all]"
```

---

## 4. LangChain 核心模块一览

### 4.1 六大核心模块

| 模块 | 英文 | 作用 | 对应前置知识 |
|------|------|------|-------------|
| **模型** | Models | 封装 LLM 调用，统一接口 | 第12集 手写 API 调用 |
| **提示词** | Prompts | 提示模板管理，变量替换 | 第13集 提示工程 |
| **记忆** | Memory | 对话历史记录管理 | — |
| **链** | Chains | 串联多个组件的工作流 | — |
| **工具** | Tools | 外部能力封装（Function Calling） | 第14-15集 |
| **智能体** | Agents | 自主决策循环 | 第22集 |

### 4.2 一句话公式

```
LangChain = LLM调用封装 + 提示模板 + 记忆 + 链式调用 + 工具生态
```

它把"手写 LLM 调用"的重复逻辑封装起来，让你专注**业务逻辑**。

---

## 5. 快速体验：三行代码创建 Agent

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取某个城市的天气"""
    return f"{city} 今天晴天，气温 22°C。"

agent = create_agent(
    model="openai:gpt-4o",        # 模型标识：厂商:模型名
    tools=[get_weather],            # 注册工具函数
    system_prompt="你是一个天气助手",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})
print(result["messages"][-1].content_blocks)
```

**关键点解析**：
- `create_agent()` —— LangChain 的核心入口，一行即可创建 Agent
- Agent 底层构建在 **LangGraph** 之上
- 支持持久化执行、人工审核、流式输出等高级特性
- `@tool` 装饰器自动将普通函数注册为可调用工具

---

## 6. 版本注意事项

### 6.1 锁定版本号

LangChain 更新非常快，**强烈建议锁定版本号**：

```bash
pip install langchain==0.3.14 \
            langchain-openai==0.3.0 \
            langchain-community==0.3.14
```

遇到 API 废弃时查阅迁移指南：
https://python.langchain.com/docs/versions/migrating_versions/

### 6.2 常见坑点

| 坑 | 说明 |
|----|------|
| **版本不兼容** | 锁定版本号，不要混用 v0.1 和 v0.3 的 API |
| **base_url 漏写** | 用 DeepSeek/豆包等非 OpenAI 模型必须设置 base_url |
| **api_key 硬编码** | 用环境变量或 `.env` 文件，不要明文写在代码里 |
| **旧教程失效** | 2026年的课程用 v0.3+ 语法，别抄 2024 年的旧代码 |

---

## 7. 本集小结

> **LangChain = 给大模型装上手脚和记忆的开发框架。**
>
> 它通过模块化设计（Models / Prompts / Memory / Chains / Tools / Agents），让开发者像搭积木一样快速组合出复杂的 AI 应用。
>
> 下一集将学习：**快速入门和 Agent 原理**——深入理解 Agent 的工作机制。

---

*参考来源：飞书文档 [J5CVwnkY7i3qPwk4TCecuYvCnOe](https://my.feishu.cn/wiki/J5CVwnkY7i3qPwk4TCecuYvCnOe) | 黑马程序员 B站课程 BV178w1z7EHQ*
