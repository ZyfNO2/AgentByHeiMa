# T15 - 多 Agent 系统设计 (Multi-Agent System Design)

## 1. 从单 Agent 到多 Agent 的演进

### 1.1 单 Agent 的能力边界

单 Agent 架构（一个 LLM + 一组工具）在许多场景下已经足够强大：

```
┌─────────────────────────────┐
│         Single Agent        │
│                             │
│  ┌───────┐    ┌──────────┐ │
│  │  LLM   │◄──│  Tools   │ │
│  │(大脑)  │──►│(手脚)    │ │
│  └───────┘    └──────────┘ │
│       ▲                    │
│       │ State              │
└───────┴────────────────────┘
```

**单 Agent 擅长的场景**：
- 单一领域的问答与任务执行（如客服、写作助手）
- 工具调用链较短的任务（通常 < 5 步）
- 响应速度要求高的实时交互
- 开发和原型验证阶段

**单 Agent 遇到的瓶颈**：

| 瓶颈 | 表现 | 根因 |
|------|------|------|
| **上下文窗口限制** | 长对话或复杂任务导致信息丢失 | LLM 有 token 上限 |
| **角色冲突** | 同一个 Prompt 要兼顾多种角色 | 单一 System Prompt 无法最优适配所有场景 |
| **工具集过大** | 几十个工具让 LLM 选择困难 | 工具选择准确率随数量增加而下降 |
| **专业深度不足** | 通用模型在特定领域表现平庸 | 缺乏领域专精的微调或知识 |
| **串行效率低** | 本可并行的步骤被迫顺序执行 | 单 Agent 只有一个"思考流" |

### 1.2 多 Agent 的核心价值

多 Agent 系统（Multi-Agent System, MAS）通过**分工协作**突破单 Agent 的限制：

```
┌──────────────────────────────────────────────────────────┐
│                   Multi-Agent System                      │
│                                                          │
│   ┌─────────┐                                            │
│   │Manager  │ ◄── 协调、分配、整合                        │
│   │(主管)   │                                            │
│   └────┬────┘                                            │
│        │ Send / 路由                                     │
│   ┌────┼────┬────────┐                                   │
│   ▼    ▼    ▼        ▼                                   │
│ ┌────┐┌────┐┌────┐ ┌────┐                                │
│ │Agent││Agent││Agent│ │Agent│                              │
│ │ A  ││ B  ││ C  │ │ D  │                                │
│ │(搜索)│(代码)│(数学)│(写作)│                              │
│ └────┘└────┘└────┘ └────┘                                │
│                                                          │
│   每个 Agent: 独立 LLM + 专业工具 + 专属 Prompt           │
└──────────────────────────────────────────────────────────┘
```

**多 Agent 的优势**：

- **专业化**：每个 Agent 可以有针对性的 System Prompt 和工具集
- **并行化**：独立任务可以同时执行，大幅缩短总耗时
- **模块化**：各 Agent 可独立开发、测试、迭代和扩展
- **容错性**：单个 Agent 失败不影响整体系统
- **可解释性**：清晰的职责划分使决策过程更透明

---

## 2. 多 Agent 架构模式

LangGraph 支持多种多 Agent 编排模式，每种模式适用于不同的任务特征。

### 2.1 四大主流模式对比

| 模式 | 英文名 | 结构 | 通信方式 | 典型适用场景 |
|------|--------|------|----------|--------------|
| **主管-委托模式** | Supervisor Pattern | Manager → Workers | Manager 分析后分发任务 | 任务分解、意图路由 |
| **顺序流水线模式** | Sequential Pipeline | A → B → C → D | 上一步输出作为下一步输入 | 固定流程的数据处理链 |
| **辩论/评审模式** | Debate/Review | Agents 互相评审 | 共享 State + 批评建议 | 决策优化、质量控制 |
| **竞争模式** | Competition | 多 Agent 竞争同一任务 | 各自独立执行，取最优结果 | 容错、质量最大化 |

### 2.2 模式一：主管-委托模式（Supervisor Pattern）

这是最常用也最灵活的多 Agent 模式。

**架构图**：
```
用户请求
    │
    ▼
┌─────────────┐
│  Supervisor  │ ← "大脑"：理解需求，拆分任务，分配给专家
│  (主管Agent) │
└──────┬──────┘
       │ Send("expert_name", state)
       ├──────────────────────────────────────┐
       ▼              ▼               ▼        ▼
   ┌──────┐     ┌──────┐      ┌──────┐   ┌──────┐
   │Researcher│   │Coder │      │Math  │   │Writer│
   │(搜索专家)│   │代码专家│      │数学专家│  │写作专家│
   └──────┘     └──────┘      └──────┘   └──────┘
       │              │               │         │
       └──────────────┴───────────────┴─────────┘
                          │
                          ▼
                  ┌─────────────┐
                  │  Supervisor  │ ← 整合所有专家的结果
                  │  (汇总输出)  │
                  └──────┬──────┘
                         │
                         ▼
                      最终回答
```

**核心实现机制——Send API**：

```python
from langgraph.types import Send

def supervisor_node(state):
    """
    主管节点：分析当前状态，决定将任务发送给哪个专家
    返回 Send 对象列表，每个 Send 指定目标节点和要传递的状态
    """
    last_message = state["messages"][-1]
    user_query = last_message.content

    # 使用轻量级 LLM 进行意图分类
    intent = classify_intent(user_query)

    # 根据 intent 分发到不同的专家 Agent
    if intent == "web_search":
        return [Send("researcher", state)]
    elif intent == "code_generation":
        return [Send("coder", state)]
    elif intent == "math_problem":
        return [Send("math_expert", state)]
    elif intent == "writing_task":
        return [Send("writer", state)]
    else:
        # 默认由通用助手处理
        return [Send("generalist", state)]


# 图构建中使用条件边
from langgraph.graph import START, END

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_agent)
graph.add_node("coder", coder_agent)
graph.add_node("math_expert", math_agent)
graph.add_node("writer", writer_agent)
graph.add_node("generalist", generalist_agent)

# 关键：使用 Send 实现动态路由
graph.add_conditional_edges(
    "supervisor",
    supervisor_node,
    {
        "researcher": "researcher",
        "coder": "coder",
        "math_expert": "math_expert",
        "writer": "writer",
        "generalist": "generalist",
    }
)

# 所有专家完成后回到主管进行汇总
for expert in ["researcher", "coder", "math_expert", "writer", "generalist"]:
    graph.add_edge(expert, "supervisor")
```

### 2.3 模式二：顺序流水线模式（Sequential Pipeline）

适用于有固定处理流程的场景：

```python
# 定义流水线中的各个阶段 Agent

def data_collector(state):
    """阶段1：数据收集"""
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([search_web, fetch_api])
    response = model.invoke([
        SystemMessage(content="你是数据收集专家。根据需求收集相关信息。"),
        *state["messages"]
    ])
    return {"messages": [response], "stage": "collected"}

def data_analyzer(state):
    """阶段2：数据分析"""
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([calculator, statistics])
    response = model.invoke([
        SystemMessage(content="你是数据分析专家。分析收集到的数据。"),
        *state["messages"]
    ])
    return {"messages": [response], "stage": "analyzed"}

def report_generator(state):
    """阶段3：报告生成"""
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke([
        SystemMessage(content="你是报告撰写专家。基于分析结果生成专业报告。"),
        *state["messages"]
    ])
    return {"messages": [response], "stage": "reported"}

def quality_reviewer(state):
    """阶段4：质量审核"""
    model = ChatOpenAI(model="gpt-4o-mini")
    review_prompt = f"""请审核以下报告的质量：
    - 数据准确性
    - 逻辑连贯性
    - 格式规范性

    如果通过审核返回 'APPROVED'，否则返回修改意见。
    当前内容：{state['messages'][-1].content}"""

    response = model.invoke(review_prompt)
    approved = "APPROVED" in response.content

    return {
        "messages": [response],
        "approved": approved,
        "stage": "reviewed"
    }

# 构建流水线图
pipeline_graph = StateGraph(AgentState)

pipeline_graph.add_node("collect", data_collector)
pipeline_graph.add_node("analyze", data_analyzer)
pipeline_graph.add_node("generate", report_generator)
pipeline_graph.add_node("review", quality_reviewer_reviewer)

# 线性连接
pipeline_graph.add_edge(START, "collect")
pipeline_graph.add_edge("collect", "analyze")
pipeline_graph.add_edge("analyze", "generate")
pipeline_graph.add_edge("generate", "review")

# 审核不通过则回退到生成环节
pipeline_graph.add_conditional_edges(
    "review",
    lambda s: "generate" if not s.get("approved") else END,
    {"generate": "generate", "__end__": END}
)

app = pipeline_graph.compile()
```

**流水线执行示意**：
```
输入: "分析2024年Q1的销售额趋势并生成报告"

[Collect] ──→ 收集了 Q1 销售原始数据 (3个API, 12条记录)
     │
     ▼
[Analyze] ──→ 计算增长率、环比、预测区间 (使用了 calculator)
     │
     ▼
[Generate] ──→ 生成包含图表说明和分析结论的报告草稿
     │
     ▼
[Review] ──→ 发现数据引用有误
     │
     │ (不通过，回退)
     ▼
[Generate] ──→ 修正数据引用，重新生成报告
     │
     ▼
[Review] ──→ APPROVED ✓
     │
     ▼
   最终输出
```

### 2.4 模式三：辩论/评审模式（Debate/Review）

多个 Agent 从不同角度评审同一个输出，提升质量：

```python
class DebateState(TypedDict):
    messages: Annotated[list, add]
    proposal: str                    # 待评审的提案
    critiques: list[str]             # 各方批评意见
    debate_round: int                # 辩论轮次
    final_verdict: Optional[str]     # 最终裁决

def proposer_agent(state: DebateState) -> dict:
    """提案者：生成初始方案"""
    model = ChatOpenAI(model="gpt-4o")
    prompt = f"""基于以下需求生成方案：
{state['messages'][-1].content}

要求：
1. 方案完整且可行
2. 考虑边界情况
3. 提供实施步骤"""
    response = model.invoke(prompt)
    return {"proposal": response.content, "debate_round": 0}

def critic_technical(state: DebateState) -> dict:
    """技术评审员：从技术可行性角度审查"""
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""作为技术架构师，严格审查以下方案的技术可行性：

方案：
{state['proposal']}

请指出：
1. 技术风险点
2. 性能瓶颈
3. 可维护性问题
4. 改进建议"""
    response = model.invoke(prompt)
    return {"critiques": state.get("critiques", []) + [f"[技术视角] {response.content}"]}

def critic_business(state: DebateState) -> dict:
    """业务评审员：从商业价值角度审查"""
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""作为产品经理，从商业角度评估以下方案：

方案：
{state['proposal']}

请评估：
1. ROI（投资回报率）
2. 时间成本
3. 用户接受度
4. 竞争优势"""
    response = model.invoke(prompt)
    return {"critiques": state.get("critiques", []) + [f"[商业视角] {response.content}"]}

def critic_security(state: DebateState) -> dict:
    """安全评审员：从安全合规角度审查"""
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""作为安全专家，审查以下方案的安全风险：

方案：
{state['proposal']}

请检查：
1. 数据隐私问题
2. 合规风险
3. 安全漏洞
4. 应对措施"""
    response = model.invoke(prompt)
    return {"critiques": state.get("critiques", []) + [f"[安全视角] {response.content}"]}

def synthesizer(state: DebateState) -> dict:
    """综合者：整合各方意见形成最终版本"""
    model = ChatOpenAI(model="gpt-4o")
    critiques_text = "\n\n".join(state.get("critiques", []))
    prompt = f"""你是一个决策协调者。以下是原方案和来自三个维度的评审意见：

=== 原始方案 ===
{state['proposal']}

=== 评审意见 ===
{critiques_text}

请综合以上所有意见：
1. 总结关键改进点
2. 输出优化后的最终方案
3. 说明采纳/拒绝每条意见的理由"""
    response = model.invoke(prompt)
    return {"final_verdict": response.content, "messages": [AIMessage(content=response.content)]}


# 构建辩论图
debate_graph = StateGraph(DebateState)

debate_graph.add_node("proposer", proposer_agent)
debate_graph.add_node("critic_tech", critic_technical)
debate_graph.add_node("critic_biz", critic_business)
debate_graph.add_node("critic_sec", critic_security)
debate_graph.add_node("synthesize", synthesizer)

debate_graph.add_edge(START, "proposer")

# 提案后并行启动三个评审员（Fan-out）
debate_graph.add_edge("proposer", "critic_tech")
debate_graph.add_edge("proposer", "critic_biz")
debate_graph.add_edge("proposer", "critic_sec")

# 所有评审完成后进入综合节点（Fan-in）
debate_graph.add_edge("critic_tech", "synthesize")
debate_graph.add_edge("critic_biz", "synthesize")
debate_graph.add_edge("critic_sec", "synthesize")

debate_graph.add_edge("synthesize", END)

debate_app = debate_graph.compile()
```

### 2.5 模式四：竞争模式（Competition）

多个 Agent 同时尝试解决同一个问题，取最佳结果：

```python
class CompetitionState(TypedDict):
    problem: str
    solutions: list[dict]  # [{agent_id, solution, confidence, reasoning}]
    best_solution: Optional[str]

def competitor_a(state: CompetitionState) -> dict:
    """竞争者A：采用保守策略，追求稳妥"""
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""你是策略A的解题专家（保守派）。
优先选择经过验证的方法，确保正确性而非创新性。

问题：{state['problem']}
请给出你的解决方案及置信度评估。"""
    response = model.invoke(prompt)
    return {
        "solutions": state.get("solutions", []) + [{
            "agent_id": "conservative_bot",
            "solution": response.content,
            "strategy": "保守验证",
            "confidence": extract_confidence(response.content)
        }]
    }

def competitor_b(state: CompetitionState) -> dict:
    """竞争者B：采用激进策略，追求创新"""
    model = ChatOpenAI(model="gpt-4o")  # 用更强的模型
    prompt = f"""你是策略B的解题专家（创新派）。
尝试新颖的方法和思路，即使有一定风险。

问题：{state['problem']}
请给出你的解决方案及置信度评估。"""
    response = model.invoke(prompt)
    return {
        "solutions": state.get("solutions", []) + [{
            "agent_id": "creative_bot",
            "solution": response.content,
            "strategy": "创新探索",
            "confidence": extract_confidence(response.content)
        }]
    }

def judge(state: CompetitionState) -> dict:
    """裁判：评估各方案，选出最优解"""
    solutions = state.get("solutions", [])
    if len(solutions) < 2:
        return {"best_solution": "等待更多竞争者提交..."}

    model = ChatOpenAI(model="gpt-4o")
    solutions_display = "\n\n".join([
        f"【方案{i+1}】({s['agent_id']}, {s['strategy']}):\n{s['solution']}\n置信度: {s['confidence']}"
        for i, s in enumerate(solutions)
    ])

    prompt = f"""以下是针对同一问题的不同解决方案，请评判哪个最优：

问题：{state['problem']}

{solutions_display}

请：
1. 分别点评每个方案的优缺点
2. 选出最佳方案
3. 如果可以，融合各方案的优点给出终极方案"""

    response = model.invoke(prompt)
    return {"best_solution": response.content}

# 竞争图
comp_graph = StateGraph(CompetitionState)

comp_graph.add_node("competitor_a", competitor_a)
comp_graph.add_node("competitor_b", competitor_b)
comp_graph.add_node("judge", judge)

comp_graph.add_edge(START, "competitor_a")
comp_graph.add_edge(START, "competitor_b")  # 并行启动
comp_graph.add_edge("competitor_a", "judge")
comp_graph.add_edge("competitor_b", "judge")
comp_graph.add_edge("judge", END)

competition_app = comp_graph.compile()
```

---

## 3. 主管-委托模式实战详解

### 3.1 专家 Agent 设计规范

每个专家 Agent 应遵循统一的设计模板：

```python
from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ====== 统一的 Agent State ======
class MultiAgentState(TypedDict):
    messages: Annotated[list, add]
    task_type: str          # 当前任务类型
    intermediate_results: dict  # 中间结果存储
    final_output: str       # 最终输出


# ====== 专家 Agent 模板 ======
class ExpertAgent:
    """
    专家 Agent 基类
    提供统一的接口和生命周期管理
    """

    def __init__(self, name: str, system_prompt: str, tools: list = None,
                 model_name: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = ChatOpenAI(model=model_name)
        if tools:
            self.model = self.model.bind_tools(tools)

    def __call__(self, state: MultiAgentState) -> dict:
        """使 Agent 可被 LangGraph 节点直接调用"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"]
        ]

        response = self.model.invoke(messages)

        return {
            "messages": [response],
            "intermediate_results": {
                **state.get("intermediate_results", {}),
                self.name: {
                    "output": response.content,
                    "tool_calls": getattr(response, 'tool_calls', []),
                }
            }
        }


# ====== 具体专家实现 ======

# --- Web Researcher ---
WEB_RESEARCHER_PROMPT = """你是一个专业的网络搜索研究员。
你的任务是：
1. 理解用户的搜索需求
2. 使用搜索工具获取最新、最相关的信息
3. 对搜索结果进行筛选和总结
4. 以结构化的方式呈现发现

注意事项：
- 优先选择权威来源（官方文档、学术论文、知名媒体报道）
- 注意信息的时效性
- 如遇到矛盾的信息，一并列出
- 不要编造信息，如果搜不到就如实告知"""

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息"""
    # 实际对接搜索 API
    return f"已搜索 '{query}'，找到 {max_results} 条结果..."

@tool
def fetch_page(url: str) -> str:
    """获取指定 URL 的页面内容"""
    return f"已获取 {url} 的内容..."

web_researcher = ExpertAgent(
    name="web_researcher",
    system_prompt=WEB_RESEARCHER_PROMPT,
    tools=[search_web, fetch_page],
)


# --- Code Expert ---
CODE_EXPERT_PROMPT = """你是一个资深编程专家。
你的任务是：
1. 理解编程需求和约束条件
2. 编写高质量、可维护的代码
3. 解释代码逻辑和设计决策
4. 处理可能的错误和边界情况

编码规范：
- 遵循语言的最佳实践
- 添加必要的注释
- 考虑错误处理和异常情况
- 保持函数/方法简洁单一"""

@tool
def execute_code(code: str, language: str = "python") -> str:
    """在沙箱中执行代码并返回结果"""
    # 实际对接代码执行环境
    return f"执行 {language} 代码:\n{code}\n---\n输出: ..."

@tool
def lint_code(code: str, language: str = "python") -> str:
    """检查代码质量和潜在问题"""
    return f"代码检查完成，发现 0 个警告..."

code_expert = ExpertAgent(
    name="code_expert",
    system_prompt=CODE_EXPERT_PROMPT,
    tools=[execute_code, lint_code],
)


# --- Math Solver ---
MATH_SOLVER_PROMPT = """你是一个数学问题求解专家。
你的任务是：
1. 理解数学问题的类型和要求
2. 选择合适的解题方法和公式
3. 给出详细的推导过程
4. 验证答案的正确性

解题原则：
- 展示完整的推理步骤
- 必要时提供多种解法
- 注意数值精度和有效数字
- 对于应用题，说明假设条件"""

@tool
def symbolic_compute(expression: str) -> str:
    """进行符号计算（代数运算、微积分等）"""
    return f"符号计算 '{expression}' 的结果..."

@tool
def numerical_solve(equation: str, method: str = "newton") -> str:
    """数值求解方程"""
    return f"使用 {method} 方法求解..."

math_solver = ExpertAgent(
    name="math_solver",
    system_prompt=MATH_SOLVER_PROMPT,
    tools=[symbolic_compute, numerical_solve],
)
```

### 3.2 主管 Agent 实现

```python
SUPERVISOR_PROMPT = """你是一个智能任务调度主管。
你的职责是：
1. 分析用户的需求，识别任务类型
2. 将任务分配给最合适的专业 Agent
3. 在必要时将复杂任务拆分为子任务
4. 整合各专家的产出，形成最终回答

你可以调度的专家团队：
- web_researcher: 网络搜索研究员（适合：查找资料、获取最新信息、验证事实）
- code_expert: 编程专家（适合：写代码、调试、代码审查、技术方案设计）
- math_solver: 数学专家（适合：数学计算、统计分析、公式推导、建模）

分配规则：
- 如果任务涉及多个领域，按主要领域分配
- 如果无法判断，默认分配给最通用的 agent
- 记录分配理由以便追溯"""

def create_supervisor():
    """创建主管 Agent"""
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field

    class RoutingDecision(BaseModel):
        target_agent: str = Field(description="目标专家名称")
        reason: str = Field(description="分配理由")
        sub_tasks: list[str] = Field(default=[], description="需要拆分的子任务")

    parser = JsonOutputParser(pydantic_object=RoutingDecision)
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([])

    def supervisor_node(state: MultiAgentState) -> list[Send]:
        """主管节点：决定任务路由"""
        last_msg = state["messages"][-1]

        # 构造路由提示
        routing_prompt = f"""{SUPERVISOR_PROMPT}

当前用户消息：{last_msg.content}
历史消息摘要：{_summarize_history(state['messages'][:-1])}

请决定将此任务分配给哪位专家。以 JSON 格式输出决策。"""

        response = model.invoke([
            SystemMessage(content=routing_prompt),
            HumanMessage(content=f"请分析并分配任务: {last_msg.content}")
        ])

        # 解析决策
        try:
            decision = parser.parse(response.content)
            target = decision.target_agent
        except Exception:
            # 解析失败时的兜底逻辑
            content = response.content.lower()
            if "搜索" in content or "查" in content or "find" in content.lower():
                target = "web_researcher"
            elif "代码" in content or "程序" in content or "code" in content.lower():
                target = "code_expert"
            elif any(w in content for w in ["计算", "数学", "算", "calculate"]):
                target = "math_solver"
            else:
                target = "web_researcher"  # 默认

        # 返回 Send 对象实现动态路由
        valid_targets = {
            "web_researcher": "web_researcher",
            "code_expert": "code_expert",
            "math_solver": "math_solver",
        }

        actual_target = valid_targets.get(target, "web_researcher")
        print(f"[Supervisor] 路由决策: {last_msg.content[:30]}... → {actual_target}")

        return [Send(actual_target, state)]

    return supervisor_node


def _summarize_history(messages: list) -> str:
    """简要总结历史消息（避免超长 context）"""
    if len(messages) <= 3:
        return "无重要历史上下文"
    recent = messages[-3:]
    summary_parts = []
    for msg in recent:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
        summary_parts.append(f"{role}: {content}")
    return "; ".join(summary_parts)
```

### 3.3 完整的多 Agent 图组装

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

def build_multi_agent_system():
    """构建完整的多 Agent 系统"""

    # 创建各组件
    supervisor = create_supervisor()

    # 定义图
    graph = StateGraph(MultiAgentState)

    # 注册节点
    graph.add_node("supervisor", supervisor)
    graph.add_node("web_researcher", web_researcher)
    graph.add_node("code_expert", code_expert)
    graph.add_node("math_solver", math_solver)

    # 定义边
    graph.add_edge(START, "supervisor")

    # 主管到专家的条件边（使用 Send 动态路由）
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state,  # supervisor 已经返回 Send 列表
        {
            "web_researcher": "web_researcher",
            "code_expert": "code_expert",
            "math_solver": "math_solver",
        }
    )

    # 专家完成后回到主管（继续处理或结束）
    graph.add_edge("web_researcher", "supervisor")
    graph.add_edge("code_expert", "supervisor")
    graph.add_edge("math_solver", "supervisor")

    # 编译
    app = graph.compile()
    return app


# 使用示例
if __name__ == "__main__":
    app = build_multi_agent_system()

    # 测试1：搜索类任务
    result1 = app.invoke({
        "messages": [HumanMessage(content="帮我查一下 Python 3.13 的新特性")]
    })
    print("=== 搜索任务 ===")
    print(result1["messages"][-1].content[:200])

    # 测试2：编程类任务
    result2 = app.invoke({
        "messages": [HumanMessage(content="用 Python 写一个快速排序算法")]
    })
    print("\n=== 编程任务 ===")
    print(result2["messages"][-1].content[:200])

    # 测试3：数学类任务
    result3 = app.invoke({
        "messages": [HumanMessage(content="求积分 ∫x²eˣdx")]
    })
    print("\n=== 数学任务 ===")
    print(result3["messages"][-1].content[:200])
```

---

## 4. 多 Agent 的状态隔离与共享

### 4.1 状态隔离策略

在多 Agent 系统中，合理的状态管理至关重要：

```
┌─────────────────────────────────────────────────┐
│              Parent Graph (父图)                 │
│                                                  │
│  Shared State (共享状态):                        │
│  ├── messages: []        (全局对话历史)          │
│  ├── user_context: {}    (用户画像)              │
│  └── task_log: []        (任务执行日志)          │
│                                                  │
│  ┌─────────────────────────────────────────┐     │
│  │     SubGraph: Researcher Agent          │     │
│  │                                        │     │
│  │  Private State (私有状态):              │     │
│  │  ├── search_queries: []                 │     │
│  │  ├── found_sources: []                  │     │
│  │  └── relevance_scores: {}               │     │
│  └─────────────────────────────────────────┘     │
│                                                  │
│  ┌─────────────────────────────────────────┐     │
│  │     SubGraph: Coder Agent               │     │
│  │                                        │     │
│  │  Private State (私有状态):              │     │
│  │  ├── code_snippets: []                  │     │
│  │  ├── test_results: {}                   │     │
│  │  └── errors_found: []                   │     │
│  └─────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

**实现方式——子图封装**：

```python
from langgraph.graph import StateGraph

# ====== 子图内部私有 State ======
class ResearcherState(TypedDict):
    # 从父图继承
    messages: Annotated[list, add]
    # 子图私有字段
    search_history: list[dict]
    collected_sources: list[str]
    current_focus: str

def researcher_internal_node_1(state: ResearcherState):
    """子图内部节点，只能访问 ResearcherState"""
    pass

def researcher_internal_node_2(state: ResearcherState):
    """子图内部节点"""
    pass

# 构建子图
researcher_subgraph = StateGraph(ResearcherState)
researcher_subgraph.add_node("plan_search", researcher_plan)
researcher_subgraph.add_node("execute_search", researcher_execute)
researcher_subgraph.add_node("summarize_findings", researcher_summarize)
researcher_subgraph.add_edge(START, "plan_search")
researcher_subgraph.add_edge("plan_search", "execute_search")
researcher_subgraph.add_edge("execute_search", "summarize_findings")
researcher_subgraph.add_edge("summarize_findings", END)

# 编译子图
compiled_researcher = researcher_subgraph.compile()


# ====== 父图中嵌入子图 ======
class ParentState(TypedDict):
    messages: Annotated[list, add]
    user_preferences: dict
    # 不包含子图的私有字段！

parent_graph = StateGraph(ParentState)
parent_graph.add_node("supervisor", supervisor_fn)
parent_graph.add_node("researcher", compiled_researcher)  # 嵌入子图
parent_graph.add_node("coder", compiled_coder)

parent_graph.add_edge(START, "supervisor")
# 父图只看到子图的输入输出，看不到内部状态
parent_graph.add_conditional_edges("supervisor", route_to_expert)
parent_graph.add_edge("researcher", "supervisor")
parent_graph.add_edge("coder", "supervisor")
```

### 4.2 Store 共享长期记忆

使用 LangGraph 的 Store 机制跨会话共享数据：

```python
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# 创建 Store（生产环境用 PostgresStore）
store = InMemoryStore()

# 在 Agent 中存取共享数据
def researcher_with_memory(state):
    """带长期记忆的研究员"""

    # 读取用户偏好（跨会话持久化）
    namespace = ("user_preferences", state["user_id"])
    prefs = store.get(namespace, "preferences")

    # 根据偏好调整搜索策略
    search_strategy = prefs.value.get("search_strategy", "balanced") if prefs else "balanced"

    # ... 执行搜索 ...

    # 将发现的优质来源存入 Store（供其他 Agent 或后续会话使用)
    store.put(
        namespace=("knowledge_base", state["user_id"]),
        key=f"sources_{datetime.now().strftime('%Y%m%d')}",
        value={"sources": discovered_sources, "timestamp": datetime.now().isoformat()}
    )

    return {"messages": [response]}


# 另一个 Agent 可以访问相同的数据
def coder_with_shared_knowledge(state):
    """能访问共享知识库的程序员"""

    # 读取研究员之前积累的知识
    namespace = ("knowledge_base", state["user_id"])
    knowledge = store.search(namespace, query="")  # 获取该用户的所有知识

    relevant_context = "\n".join([
        item.value.get("sources", "") for item in knowledge.items[:5]
    ])

    # 在编码时参考这些背景知识
    prompt = f"""参考以下背景信息完成任务：
{relevant_context}

用户需求：{state['messages'][-1].content}"""
    # ...
```

### 4.3 通信协议设计

父图与子图、子图之间的通信需要明确的协议：

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_INPUT = "requires_input"

class TaskResult(BaseModel):
    """标准化的 Agent 间通信格式"""
    status: TaskStatus
    output: str
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None
    next_action: Optional[str] = None  # 建议主管的下一步操作
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

def expert_wrapper(agent_func):
    """
    装饰器：确保所有专家 Agent 返回标准化的 TaskResult
    """
    def wrapper(state):
        try:
            raw_result = agent_func(state)

            # 包装为标准化格式
            return {
                "messages": [
                    AIMessage(
                        content=json.dumps(
                            TaskResult(
                                status=TaskStatus.COMPLETED,
                                output=raw_result.get("output", ""),
                                metadata=raw_result.get("metadata", {}),
                                confidence=raw_result.get("confidence", 0.8)
                            ).model_dump(),
                            ensure_ascii=False
                        )
                    )
                ]
            }
        except Exception as e:
            return {
                "messages": [
                    AIMessage(content=json.dumps(
                        TaskResult(
                            status=TaskStatus.FAILED,
                            output="",
                            error=str(e),
                            confidence=0.0
                        ).model_dump(),
                        ensure_ascii=False
                    ))
                ]
            }

    return wrapper
```

---

## 5. 完整项目：多 Agent 研究助手

### 5.1 项目概述

构建一个端到端的**多 Agent 学术研究助手**，能够自动完成从文献检索到报告撰写的全流程。

### 5.2 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Academic Research Assistant                   │
│                                                                 │
│  用户输入: "帮我研究一下 Transformer 在 3D 视觉中的应用进展"      │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Router Agent (路由器)                   │   │
│  │  职责: 意图识别 + 任务分解 + 优先级排序                     │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │ Send                                    │
│         ┌─────────────┼─────────────┬─────────────┐             │
│         ▼             ▼             ▼             ▼             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Web      │  │ Paper    │  │ Code     │  │ Data     │        │
│  │ Researcher│  │ Reader   │  │ Analyst  │  │ Scientist│        │
│  │ 网络研究员│  │ 论文阅读 │  │ 代码分析 │  │ 数据分析 │        │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘        │
│        │             │             │             │              │
│        └─────────────┴──────┬──────┴─────────────┘              │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Synthesizer (综合器)                     │   │
│  │  职责: 整合多方结果 → 结构化知识 → 生成研究报告             │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                               │
│              │ Quality Reviewer │                               │
│              │   质量审核员      │                               │
│              └────────┬────────┘                               │
│                       │                                         │
│                       ▼                                         │
│              最终研究报告 (Markdown/PDF)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 核心代码骨架

```python
"""
多 Agent 学术研究助手 - 完整代码骨架
文件结构:
  research_assistant/
  ├── agents/
  │   ├── __init__.py
  │   ├── router.py          # Router Agent
  │   ├── web_researcher.py  # Web Researcher
  │   ├── paper_reader.py    # Paper Reader
  │   ├── code_analyst.py    # Code Analyst
  │   ├── data_scientist.py  # Data Scientist
  │   └── synthesizer.py     # Synthesizer
  ├── graph.py               # 主图定义
  ├── state.py               # State 定义
  ├── store.py               # Store 配置
  └── config.py              # 配置管理
"""

# ====== state.py ======
from typing import TypedDict, Annotated, Literal
from operator import add

class ResearchState(TypedDict):
    # 全局状态
    messages: Annotated[list, add]
    research_topic: str

    # 路由相关
    task_plan: list[dict]          # [{agent, task, priority, dependencies}]
    completed_tasks: list[str]     # 已完成的任务 ID
    failed_tasks: list[str]        # 失败的任务 ID

    # 各 Agent 的产出
    web_findings: dict             # Web Researcher 的发现
    paper_summaries: list[dict]    # Paper Reader 的论文摘要
    code_analysis: dict            # Code Analyst 的分析结果
    data_insights: dict            # Data Scientist 的数据分析

    # 最终输出
    draft_report: str              # Synthesizer 生成的初稿
    final_report: str              # 经过审核的终稿
    review_feedback: str           # 审核意见


# ====== router.py ======
ROUTER_SYSTEM_PROMPT = """你是学术研究助手的智能路由器。
收到研究主题后，你需要：

1. **分析主题**：识别涉及的关键领域
2. **制定计划**：拆分为具体的子任务
3. **确定依赖**：哪些任务可以并行，哪些必须顺序执行
4. **分配优先级**：按重要性排序

可用的专家团队：
- web_researcher: 搜索最新的研究动态、新闻、博客讨论
- paper_reader: 阅读和理解学术论文 PDF
- code_analyst: 分析 GitHub 上的开源实现
- data_scientist: 处理实验数据和基准测试结果

输出格式：JSON 格式的任务计划列表。"""

def router_agent(state: ResearchState) -> dict:
    """路由 Agent：制定研究计划"""
    model = ChatOpenAI(model="gpt-4o")

    topic = state.get("research_topic", state["messages"][-1].content)

    prompt = f"""{ROUTER_SYSTEM_PROMPT}

研究主题：{topic}

请输出详细的研究计划，包括：
1. 需要执行的子任务列表
2. 每个子任务分配给哪个专家
3. 任务之间的依赖关系
4. 建议的执行顺序"""

    response = model.invoke(prompt)

    # 解析计划（实际项目中应使用 structured output）
    task_plan = parse_task_plan(response.content)

    return {
        "task_plan": task_plan,
        "research_topic": topic,
        "messages": [AIMessage(content=f"研究计划已制定，共 {len(task_plan)} 个子任务。")]
    }


# ====== web_researcher.py ======
WEB_RESEARCHER_PROMPT = """你是学术界的网络搜索专家。
专注于为学术研究收集高质量的网络资源。

搜索策略：
1. 先用宽泛的关键词了解整体格局
2. 再逐步缩小范围深入细节
3. 关注 arXiv、Google Scholar、会议官网等学术来源
4. 同时关注技术博客和社区讨论获取实践见解

输出要求：
- 对每个发现标注来源可信度 (high/medium/low)
- 提取关键信息和原文链接
- 标注信息发布时间以确保时效性"""

@tool
def academic_search(query: str, source: str = "scholar") -> str:
    """学术资源搜索"""
    pass

@tool
def search_arxiv(topic: str, max_results: int = 10) -> str:
    """搜索 arXiv 预印本"""
    pass

@tool
def search_github(topic: str, sort_by: str = "stars") -> str:
    """搜索 GitHub 相关项目"""
    pass

def web_researcher_agent(state: ResearchState) -> dict:
    """Web Researcher Agent"""
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([
        academic_search, search_arxiv, search_github
    ])

    topic = state["research_topic"]
    tasks_for_me = [t for t in state["task_plan"] if t["agent"] == "web_researcher"]

    prompt = f"""{WEB_RESEARCHER_PROMPT}

研究主题：{topic}

分配给你的任务：
{json.dumps(tasks_for_me, ensure_ascii=False, indent=2)}

请开始执行搜索任务。"""

    response = model.invoke([
        SystemMessage(content=prompt),
        *state["messages"]
    ])

    # 处理工具调用...
    findings = process_tool_calls(response, state)

    return {
        "web_findings": findings,
        "completed_tasks": state.get("completed_tasks", []) + [t["id"] for t in tasks_for_me],
        "messages": [response]
    }


# ====== synthesizer.py ======
SYNTHESIZER_PROMPT = """你是学术研究报告的综合撰写专家。
你的任务是将各专家的产出整合为一篇高质量的研究报告。

报告结构：
1. 执行摘要 (Executive Summary)
2. 研究背景与动机
3. 主要发现与技术进展
4. 关键方法对比分析
5. 开源资源与工具盘点
6. 数据与实验洞察
7. 未来研究方向
8. 参考文献

写作风格：
- 学术严谨但不晦涩
- 数据驱动，每个论点都有来源支撑
- 适当使用表格和对比来增强可读性
- 对不确定的内容明确标注"""

def synthesizer_agent(state: ResearchState) -> dict:
    """Synthesizer Agent：整合所有研究结果"""
    model = ChatOpenAI(model="gpt-4o")  # 用更强的模型做综合

    # 汇总所有中间结果
    all_findings = {
        "web_findings": state.get("web_findings", {}),
        "paper_summaries": state.get("paper_summaries", []),
        "code_analysis": state.get("code_analysis", {}),
        "data_insights": state.get("data_insights", {}),
    }

    prompt = f"""{SYNTHESIZER_PROMPT}

研究主题：{state['research_topic']}

===== 各专家的研究成果 =====

【网络研究发现】
{json.dumps(all_findings['web_findings'], ensure_ascii=False, indent=2)}

【论文阅读摘要】
{json.dumps(all_findings['paper_summaries'], ensure_ascii=False, indent=2)}

【代码分析结果】
{json.dumps(all_findings['code_analysis'], ensure_ascii=False, indent=2)}

【数据分析洞察】
{json.dumps(all_findings['data_insights'], ensure_ascii=False, indent=2)}

请基于以上材料撰写完整的研究报告。"""

    response = model.invoke(prompt)

    return {
        "draft_report": response.content,
        "messages": [AIMessage(content="研究报告初稿已完成。")]
    }


# ====== graph.py (主图组装) ======
def build_research_assistant():
    """构建完整的多 Agent 研究助手系统"""

    from agents import (
        router_agent,
        web_researcher_agent,
        paper_reader_agent,
        code_analyst_agent,
        data_scientist_agent,
        synthesizer_agent,
        reviewer_agent,
    )

    graph = StateGraph(ResearchState)

    # 注册所有节点
    graph.add_node("router", router_agent)
    graph.add_node("web_researcher", web_researcher_agent)
    graph.add_node("paper_reader", paper_reader_agent)
    graph.add_node("code_analyst", code_analyst_agent)
    graph.add_node("data_scientist", data_scientist_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("reviewer", reviewer_agent)

    # 定义边的逻辑
    graph.add_edge(START, "router")

    # 路由器并行分发给各专家（Fan-out）
    graph.add_conditional_edges("router", route_from_plan, {
        "web_researcher": "web_researcher",
        "paper_reader": "paper_reader",
        "code_analyst": "code_analyst",
        "data_scientist": "data_scientist",
    })

    # 所有专家完成后进入综合器（Fan-in）
    graph.add_edge("web_researcher", "synthesizer")
    graph.add_edge("paper_reader", "synthesizer")
    graph.add_edge("code_analyst", "synthesizer")
    graph.add_edge("data_scientist", "synthesizer")

    # 综合完成后进入审核
    graph.add_edge("synthesizer", "reviewer")

    # 审核可能需要回退修改
    graph.add_conditional_edges("reviewer", check_approval, {
        "synthesizer": "synthesizer",  # 不通过则修改
        "__end__": END,                # 通过则结束
    })

    return graph.compile(checkpointer=MemorySaver())


def route_from_plan(state: ResearchState) -> list[Send]:
    """根据路由计划生成 Send 列表"""
    plan = state.get("task_plan", [])
    sends = []
    for task in plan:
        sends.append(Send(task["agent"], state))
    return sends


def check_approval(state: ResearchState) -> str:
    """检查审核是否通过"""
    feedback = state.get("review_feedback", "")
    if "APPROVED" in feedback or "通过" in feedback:
        return "__end__"
    return "synthesizer"


# ====== 使用入口 ======
if __name__ == "__main__":
    app = build_research_assistant()

    # 启动研究任务
    result = app.invoke(
        {
            "messages": [HumanMessage(
                content="帮我全面研究一下 Vision-Language Model 在医学影像诊断中的最新进展"
            )],
            "research_topic": "",
        },
        config={"configurable": {"thread_id": "research_session_001"}}
    )

    # 输出最终报告
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.get("final_report", result.get("draft_report", "No report generated")))
```

---

## 本章小结

本章系统地讲解了多 Agent 系统的设计原理与实战方法，核心要点包括：

1. **演进逻辑**：从单 Agent 到多 Agent 是应对复杂性增长的必然选择，解决了上下文限制、角色冲突、工具膨胀等问题
2. **四大架构模式**：主管-委托（灵活路由）、顺序流水线（固定流程）、辩论评审（质量保障）、竞争模式（容错优选），各有适用场景
3. **Send API 核心**：`Send(target_node, state)` 是 LangGraph 实现动态路由和多实例并发的关键原语
4. **状态管理策略**：通过子图封装实现隔离、通过 Store 实现共享、通过标准化通信协议保证协作可靠性
5. **完整项目落地**：以学术研究助手为例，展示了从架构设计到代码实现的完整路径

多 Agent 系统是 Agent 应用走向规模化、专业化的关键技术方向，值得深入掌握和实践。

---

> **参考来源**：黑马程序员 LangChain 课程 - BV178w1z7EHQ（第3章 Agent 进阶 - T15 多 Agent 系统设计）
