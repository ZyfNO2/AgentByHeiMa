# T10 - MapReduce 模式

> **课程来源**：黑马程序员 LangChain 课程 第3章 Agent 进阶
> **视频参考**：BV178w1z7EHQ
> **T 编号**：T10

---

## 1. MapReduce 在 LangGraph 中的应用

### 1.1 什么是 MapReduce？

MapReduce 是一种经典的**分布式计算范式**，源自 Google 的同名论文（Dean & Ghemawat, 2004）。在 LangGraph 的上下文中，它被重新诠释为一种**对列表数据进行批量并行处理**的模式：

| 阶段 | 传统 MapReduce | LangGraph 中的 MapReduce |
|------|---------------|--------------------------|
| **Map（映射）** | 对数据分片执行相同函数 | 对列表中每个元素调用同一个节点/子图 |
| **Shuffle（洗牌）** | 按键分组 | 通常省略或隐式处理 |
| **Reduce（归约）** | 聚合各组结果 | 合并所有 Map 输出为最终结果 |

### 1.2 为什么在 Agent 中需要 MapReduce？

```
场景：用户上传了 50 篇文档，想问"这些文档主要讲了什么？"

❌ 串行方式：
  文档1 → 分析 → 文档2 → 分析 → ... → 文档50 → 分析 → 汇总
  耗时：50 × 单次分析时间 = 可能数分钟

✅ MapReduce 方式：
  [文档1] ─┐
  [文档2] ─┤
  [...]   ├──→ 并行分析（Map）──→ 汇总聚合（Reduce）
  [文档50]─┘
  耗时：≈ max(各次分析) + 汇总时间 = 可能只需十几秒
```

### 1.3 典型应用场景

| 场景 | Map 操作 | Reduce 操作 |
|------|---------|-------------|
| **批量文档分析** | 对每篇文档做摘要/分类/提取 | 拼接摘要 / 投票分类 / 合并实体 |
| **多源数据采集** | 从每个数据源拉取数据 | 去重、排序、融合 |
| **多模型集成** | 用不同 LLM 分别回答同一问题 | 综合投票 / 选择最佳答案 |
| **大规模 RAG** | 对每个文档块分别检索和打分 | 重排序后取 Top-K |
| **批量内容审核** | 对每条内容分别检查安全性 | 汇总所有违规项 |

---

## 2. LangGraph 中的 Map 实现

### 2.1 核心方法：`graph.map()`

LangGraph 提供了 `map()` 方法，可以将一个节点或子图**映射到输入列表的每个元素上**，自动实现并行执行：

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from operator import add

# ==================== 定义 State ====================
class MapInputState(TypedDict):
    documents: List[str]       # 待处理的文档列表
    question: str              # 用户的问题

class MapOutputState(TypedDict):
    # 每个 Map worker 的输出会通过 add reducer 追加到此列表
    document_answers: Annotated[List[str], add]

class OverallState(MapInputState, MapOutputState):
    final_answer: str          # Reduce 后的最终答案


# ==================== 定义 Map 节点函数 ====================
async def document_analyzer(state):
    """
    Map 阶段的工作函数
    注意：state 中包含的是单个文档的数据（由 map() 自动分发）
    """
    doc = state.get("document", "")
    question = state.get("question", "")

    # 这里实际会调用 LLM 对单个文档进行问答
    # answer = await llm.ainvoke(f"根据以下文档回答问题...\n文档: {doc}\n问题: {question}")
    answer = f"[基于文档片段]: 关于'{question}'，该文档提到..."

    return {"document_answers": [answer]}


# ==================== 构建图并使用 map() ====================
def build_mapreduce_graph():
    graph = StateGraph(OverallState)

    # 添加 Map 节点
    graph.add_node("analyzer", document_analyzer)

    # 添加 Reduce 节点
    async def reducer_node(state):
        answers = state.get("document_answers", [])
        # Reduce 逻辑：综合所有片段答案生成最终回答
        combined = "\n".join([f"(来源{i+1}): {a}" for i, a in enumerate(answers)])
        final = f"综合{len(answers)}个文档的分析结果:\n{combined}"
        return {"final_answer": final}

    graph.add_node("reducer", reducer_node)

    # 使用 map() 配置并行处理
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "reducer")
    graph.add_edge("reducer", END)

    app = graph.compile()

    # 关键：使用 map() 方法运行
    return app


# ==================== 运行示例 ====================
async def main():
    app = build_mapreduce_graph()

    # 准备输入：10 个文档 + 1 个问题
    docs = [f"这是第 {i+1} 篇文档的内容..." for i in range(10)]
    input_data = {
        "documents": docs,
        "question": "这些文档的核心观点是什么？"
    }

    # 使用 amap() 执行 MapReduce
    result = await app.amap(
        input_data,
        {"name": "analyzer"},       # 要 map 到的节点名
        {},                          # 运行配置（可选）
    )

    print(result["final_answer"])
```

### 2.2 `map()` 与 `amap()` 方法签名详解

```python
# 同步版本
result = graph.map(
    input: dict,                    # 包含列表字段的输入 State
    runnable: Union[str, Runnable], # 目标节点名或 Runnable 对象
    config: Optional[RunnableConfig] = None,  # 运行配置
) -> dict

# 异步版本（推荐）
result = await graph.amap(
    input: dict,
    runnable: Union[str, Runnable],
    config: Optional[RunnableConfig] = None,
) -> dict
```

### 2.3 参数详细说明

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| **input** | `dict` | 是 | 包含待映射列表的完整 State | `{"documents": [...], "question": "..."}` |
| **runnable** | `str` 或 `Runnable` | 是 | Map 的目标——节点名字符串或子图对象 | `"analyzer"` 或 `sub_graph_app` |
| **config** | `RunnableConfig` | 否 | 运行配置，可含 `thread_id`、`callbacks` 等 | `{"configurable": {"thread_id": "t-001"}}` |

---

## 3. Map 的工作原理深入

### 3.1 数据分发机制

当调用 `map(input, "node_name")` 时，LangGraph 内部执行以下操作：

```
输入 State:
{
    "documents": ["doc1", "doc2", "doc3"],   ← 列表字段（被 map）
    "question": "什么是AI？",                   ← 标量字段（广播给所有分支）
}

                    map() 内部处理
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    分支1 State      分支2 State      分支3 State
    {                {                {
      document:"doc1",  document:"doc2",  document:"doc3",
      question:"什么是AI？" question:"什么是AI？" question："什么是AI？"
    }                }                }
           │               │              
           ▼               ▼               ▼
     [analyzer]       [analyzer]       [analyzer]
           │               │              
           ▼               ▼               ▼
    {"answers":["..."]}{"answers":["..."]}{"answers":["..."]}
           │               │ │
           └───────────────┘ │
                  add reducer (自动拼接)
                             │
                             ▼
          {"document_answers": ["...", "...", "..."]}
```

**关键规则**：
1. **列表字段** → 拆解为元素，每个元素分配到一个分支
2. **标量字段** → 自动**广播**到所有分支（每个分支都收到完整的值）
3. **输出字段** → 通过 reducer 合并回父级 State

### 3.2 如何指定哪个字段是 Map 的目标？

LangGraph 通过以下规则确定 Map 字段：

```python
class MyState(TypedDict):
    items: List[str]            # List 类型字段 → 默认作为 Map 目标
    context: str                # 标量 → 广播
    results: Annotated[List[int], add]  # 带 reducer 的列表 → 收集输出
```

- **默认行为**：State 中第一个 `List` 类型（非 Annotated）的字段被视为 Map 输入
- **显式指定**：可以通过配置参数明确指定要 map 的字段名

### 3.3 Map 子图的 State 设计模式

对于更复杂的 Map 任务，可以使用子图作为 Map 单元：

```python
# ====== 子图 State（单个元素的处理单元）======
class ItemProcessState(TypedDict):
    item: str                   # 当前处理的单项数据
    config_param: int           # 广播的配置参数
    analysis_result: str        # 该项的处理结果
    confidence_score: float     # 置信度

# ====== 子图定义 ======
item_graph = StateGraph(ItemProcessState)
item_graph.add_node("analyze", analyze_item_node)
item_graph.add_node("score", score_result_node)
item_graph.add_edge(START, "analyze")
item_graph.add_edge("analyze", "score")
item_graph.add_edge("score", END)
item_app = item_graph.compile()


# ====== 主图使用子图进行 Map ======
main_graph = StateGraph(MainState)
main_graph.add_node("processor", item_app)  # 子图作为节点
# ... 其他配置

# map 时传入子图
result = await main_graph.amap(
    {"items": large_list, "config_param": 42},
    item_app,  # 直接传子图应用
)
```

---

## 4. Reduce：聚合 Map 结果

### 4.1 Reduce 的本质

Reduce 阶段的任务是将 Map 阶段产生的**多个部分结果**合并为一个**有意义的整体结果**。不同的业务场景需要不同的聚合策略。

### 4.2 常见 Reduce 策略

#### 策略一：拼接合并（Concatenation）

最简单的策略，将所有结果按顺序拼接：

```python
def concat_reduce(existing, new):
    """拼接所有结果"""
    if existing is None:
        return new if isinstance(new, list) else [new]
    if isinstance(new, list):
        return existing + new
    return existing + [new]

# 适用场景：文档摘要收集、搜索结果汇总
```

**输出示例**：
```
输入: ["摘要A", "摘要B", "摘要C"]
输出: "摘要A\n摘要B\n摘要C"
```

#### 策略二：投票决策（Voting）

用于分类任务，选择出现次数最多的结果：

```python
from collections import Counter

def voting_reduce(existing, new):
    """投票：选票数最多的类别"""
    votes = (existing or []) + ([new] if not isinstance(new, list) else new)
    counter = Counter(votes)
    return counter.most_common(1)[0][0]

# 适用场景：多模型情感分析、内容分类
```

**输出示例**：
```
输入: ["正面", "正面", "负面", "正面"]
输出: "正面"  （3票 vs 1票）
```

#### 策略三：加权平均（Weighted Average）

用于数值型结果，按置信度加权：

```python
def weighted_avg_reduce(existing, new):
    """加权平均聚合"""
    if existing is None:
        return {"total": new["value"] * new["weight"], "weight_sum": new["weight"]}

    existing["total"] += new["value"] * new["weight"]
    existing["weight_sum"] += new["weight"]
    return existing

# 最终计算: result["total"] / result["weight_sum"]

# 适用场景：多模型评分聚合、置信度加权
```

#### 策略四：LLM 摘要（LLM Summarization）

用 LLM 对所有部分结果进行智能综合：

```python
async def llm_summarize_reducer(state):
    """调用 LLM 综合所有 Map 结果"""
    partial_results = state.get("partial_answers", [])

    system_prompt = """你是一个信息综合专家。
    下面是来自多个独立源的分析结果，请综合它们生成一个全面、一致的最终答案。
    要求：保留关键信息，消除矛盾，逻辑连贯。"""

    user_prompt = "请综合以下分析结果:\n\n" + "\n---\n".join([
        f"[来源{i+1}]: {r}" for i, r in enumerate(partial_results)
    ])

    # final_answer = await llm.ainvoke([
    #     SystemMessage(content=system_prompt),
    #     HumanMessage(content=user_prompt)
    # ])
    final_answer = f"[LLM 综合] 基于 {len(partial_results)} 个来源的综合分析..."

    return {"final_answer": final_answer}

# 适用场景：多源研究报告、多方观点综合
```

#### 策略五：去重融合（Dedup & Fusion）

用于搜索结果等可能重复的场景：

```python
def dedup_fusion_reduce(existing, new):
    """去重融合：按标题相似度去重"""
    if existing is None:
        return [new]

    # 简单演示：按 title 去重（实际可用语义相似度）
    seen_titles = {item["title"] for item in existing}
    if new["title"] not in seen_titles:
        existing.append(new)
    return existing

# 适用场景：多搜索引擎结果融合
```

### 4.3 Reduce 策略选择指南

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 文档摘要收集 | 拼接 | 保留全部信息供后续参考 |
| 多模型分类 | 投票 | 集成学习中的经典做法 |
| 数值评分 | 加权平均 | 充分利用置信度信息 |
| 最终报告生成 | LLM 摘要 | 需要"理解"后的综合而非机械拼接 |
| 搜索结果 | 去重融合 | 不同源可能有重叠结果 |

---

## 5. 完整实战：批量文档问答系统

### 5.1 系统架构

```
┌─────────────────────────────────────────────────────┐
│                     用户输入                         │
│         问题: "这家公司的核心技术优势是什么？"         │
│         相关文档: [doc1, doc2, ..., doc10]          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Map 阶段（并行）                        │
│                                                      │
│   doc1 ──→ [RAG Analyzer 1] ──→ answer_1            │
│   doc2 ──→ [RAG Analyzer 2] ──→ answer_2            │
│   ...         (并行执行)           ...                │
│   doc10 ─→ [RAG Analyzer 10] ──→ answer_10          │
│                                                      │
│   每个 Analyzer: 检索相关片段 → LLM 生成基于该文档的回答 │
└──────────────────────┬──────────────────────────────┘
                       │ 全部完成
                       ▼
┌─────────────────────────────────────────────────────┐
│              Reduce 阶段                              │
│                                                      │
│   [Synthesizer Node]                                 │
│   输入: [answer_1, answer_2, ..., answer_10]         │
│   处理:                                              │
│     1. 过滤掉"文档中无相关信息"的空回答               │
│     2. 按信息密度排序                                │
│     3. 调用 LLM 综合生成最终答案                      │
│   输出: 最终综合回答                                  │
└─────────────────────────────────────────────────────┘
```

### 5.2 完整代码实现

```python
"""
MapReduce 批量文档问答系统 - 完整实现
功能：对 N 个相关文档并行执行 RAG 问答，然后综合结果
"""

import asyncio
import time
from typing import Annotated, TypedDict, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
# 第一部分：State 定义
# ============================================================

class DocumentItem(TypedDict):
    """单个文档的数据结构"""
    content: str               # 文档正文
    source: str                # 来源标识
    metadata: Dict             # 元数据（日期、作者等）

class MapWorkerState(TypedDict):
    """Map 工作节点的状态（接收单个文档）"""
    document: DocumentItem     # 当前处理的文档
    question: str              # 用户问题（广播字段）

class MapWorkerOutput(TypedDict):
    """Map 工作节点的输出"""
    partial_answers: Annotated[List[Dict], add]
    # 每个 worker 返回一个 dict，add reducer 将它们拼接到列表

class OverallState(TypedDict):
    """全局状态"""
    documents: List[DocumentItem]     # 输入：文档列表（Map 目标）
    question: str                     # 输入：用户问题
    partial_answers: List[Dict]       # 中间：各文档的部分回答
    final_answer: str                 # 输出：最终综合回答
    statistics: Dict                  # 输出：统计信息

# ============================================================
# 第二部分：Map 阶段 - 文档分析器
# ============================================================

async def document_rag_analyzer(state: MapWorkerState):
    """
    Map Worker: 对单个文档执行 RAG 问答
    实际项目中这里会：
    1. 对文档做 chunking 和 embedding
    2. 向量检索相关片段
    3. 构建 prompt 并调用 LLM
    """
    doc = state["document"]
    question = state["question"]

    # 模拟处理延迟
    await asyncio.sleep(0.3 + (hash(doc["source"]) % 100) / 200)

    # 模拟 RAG 分析结果
    content_preview = doc["content"][:50] + "..."

    # 判断该文档是否与问题相关（模拟）
    is_relevant = len(doc["content"]) > 20  # 简单模拟

    if not is_relevant:
        return {"partial_answers": [{
            "source": doc["source"],
            "relevant": False,
            "answer": None,
            "reason": "文档内容与问题不相关"
        }]}

    # 模拟 LLM 生成的基于该文档的回答
    partial_answer = {
        "source": doc["source"],
        "relevant": True,
        "answer": (
            f"根据《{doc['source']}》的分析："
            f"{content_preview}中提到了与'{question}'相关的关键技术点。"
        ),
        "confidence": 0.75 + (hash(doc["source"]) % 25) / 100,
        "key_points": ["技术点A", "技术点B", "技术点C"]
    }

    print(f"  [Map] {doc['source']}: {'✓ 相关' if is_relevant else '✗ 无关'}")

    return {"partial_answers": [partial_answer]}


# ============================================================
# 第三部分：Reduce 阶段 - 结果综合器
# ============================================================

async def answer_synthesizer(state: OverallState):
    """
    Reducer Node: 综合所有 Map 结果
    """
    partials = state.get("partial_answers", [])
    question = state["question"]

    # Step 1: 过滤有效回答
    relevant_answers = [p for p in partials if p.get("relevant", False)]

    # Step 2: 按置信度排序
    relevant_answers.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # Step 3: 构建综合 Prompt（实际中调用 LLM）
    sources_info = "\n".join([
        f"- [{a['source']}] (置信度: {a['confidence']:.0%}): {a['answer'][:80]}..."
        for a in relevant_answers
    ])

    # 模拟 LLM 综合
    final = f"""关于「{question}」的综合分析：

共分析了 {len(partials)} 个文档，其中 {len(relevant_answers)} 个包含相关信息。

主要发现：
{chr(10).join([f'  {i+1}. {a["source"]}: {a["answer"][:60]}...' for i, a in enumerate(relevant_answers[:5])])}

结论：综合以上 {len(relevant_answers)} 个信息源，可以得出...
（此处应为 LLM 生成的完整综合回答）
"""

    # 统计信息
    stats = {
        "total_documents": len(partials),
        "relevant_count": len(relevant_answers),
        "irrelevant_count": len(partials) - len(relevant_answers),
        "avg_confidence": sum(a.get("confidence", 0) for a in relevant_answers) / max(len(relevant_answers), 1),
    }

    print(f"\n  [Reduce] 统计: {stats['relevant_count']}/{stats['total_documents']} 个文档相关")

    return {
        "final_answer": final,
        "statistics": stats,
    }

# ============================================================
# 第四部分：构建图
# ============================================================

def build_doc_qa_mapreduce():
    """构建 MapReduce 文档问答图"""

    graph = StateGraph(OverallState)

    # Map 节点
    graph.add_node("doc_analyzer", document_rag_analyzer)

    # Reduce 节点
    graph.add_node("synthesizer", answer_synthesizer)

    # 边连接
    graph.add_edge(START, "doc_analyzer")
    graph.add_edge("doc_analyzer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ============================================================
# 第五部分：性能对比测试
# ============================================================

async def run_mapreduce(documents, question):
    """MapReduce 方式运行"""
    app = build_doc_qa_mapreduce()

    start = time.perf_counter()
    result = await app.amap(
        {"documents": documents, "question": question},
        {"name": "doc_analyzer"},
    )
    elapsed = time.perf_counter() - start

    return result, elapsed


async def run_serial(documents, question):
    """串行方式运行（对比基线）"""
    app = build_doc_qa_mapreduce()

    start = time.perf_counter()
    results = []
    for doc in documents:
        r = await app.ainvoke({"documents": [doc], "question": question})
        results.append(r)
    elapsed = time.perf_counter() - start

    # 手动合并
    all_partials = []
    for r in results:
        all_partials.extend(r.get("partial_answers", []))

    from copy import deepcopy
    merged_state = deepcopy(results[0]) if results else {}
    merged_state["partial_answers"] = all_partials
    merged_state = await answer_synthesizer(merged_state)

    return merged_state, elapsed


async def main():
    # 准备测试数据
    documents = [
        {"content": f"这是第 {i+1} 篇技术文档的详细内容，讨论了公司核心技术的各个方面..."
         f"{'更多技术细节...' * (i % 3 + 1)}",
         "source": f"技术文档_v{i+1}.pdf",
         "metadata": {"date": f"2024-{(i%12)+1}-01"}}
        for i in range(15)
    ]
    question = "这家公司的核心技术优势是什么？"

    print("=" * 60)
    print("MapReduce 批量文档问答系统 - 性能对比测试")
    print("=" * 60)
    print(f"\n文档数量: {len(documents)}")
    print(f"用户问题: {question}\n")

    # 测试 MapReduce
    print("--- MapReduce 方式 ---")
    mr_result, mr_time = await run_mapreduce(documents, question)
    print(f"\n耗时: {mr_time:.2f}s")
    print(f"统计: {mr_result['statistics']}")

    # 测试串行
    print("\n--- 串行方式（对比） ---")
    serial_result, serial_time = await run_serial(documents, question)
    print(f"\n耗时: {serial_time:.2f}s")
    print(f"统计: {serial_result['statistics']}")

    # 性能对比总结
    print("\n" + "=" * 60)
    speedup = serial_time / mr_time if mr_time > 0 else float('inf')
    print(f"加速比: {speedup:.1f}x")
    print(f"MapReduce: {mr_time:.2f}s vs 串行: {serial_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 预期输出

```
============================================================
MapReduce 批量文档问答系统 - 性能对比测试
============================================================

文档数量: 15
用户问题: 这家公司的核心技术优势是什么？

--- MapReduce 方式 ---
  [Map] 技术文档_v1.pdf: ✓ 相关
  [Map] 技术文档_v2.pdf: ✓ 相关
  ...
  [Map] 技术文档_v15.pdf: ✓ 相关

  [Reduce] 统计: 15/15 个文档相关

耗时: 0.82s
统计: {'total_documents': 15, 'relevant_count': 15, ...}

--- 串行方式（对比）---
  ...

耗时: 5.23s
...

============================================================
加速比: 6.4x
MapReduce: 0.82s vs 串行: 5.23s
============================================================
```

---

## 6. 高级话题与注意事项

### 6.1 资源消耗管理

`map()` 会为列表中的**每个元素创建一个子线程（thread）**：

| 元素数量 | 子线程数 | 内存消耗建议 | 建议 |
|---------|---------|-------------|------|
| 1~10 | 1~10 | 通常无压力 | 直接 map |
| 10~100 | 10~100 | 中等 | 关注并发限制 |
| 100~1000 | 100~1000 | 较高 | **必须分批** |
| 1000+ | 1000+ | 很高 | 考虑其他架构 |

### 6.2 分批处理策略

当元素数量过大时，应采用分批 MapReduce：

```python
import math

async def batched_mapreduce(graph, items, question, batch_size=20):
    """分批 MapReduce：避免同时启动过多子线程"""

    all_results = []
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(items))
        batch = items[start:end]

        print(f"处理批次 {batch_idx+1}/{num_batches} ({len(batch)} 个元素)")

        batch_result = await graph.amap(
            {"documents": batch, "question": question},
            {"name": "analyzer"},
        )
        all_results.extend(batch_result.get("partial_answers", []))

    # 最终全局 Reduce
    final_state = {"partial_answers": all_results, "question": question}
    final = await global_reducer(final_state)
    return final
```

### 6.3 错误处理与容错

单个元素的失败不应影响整个 MapReduce 流程：

```python
async def robust_document_analyzer(state: MapWorkerState):
    """带错误恢复的 Map Worker"""
    doc = state["document"]
    try:
        # 正常处理逻辑...
        result = await process_document(doc, state["question"])
        return {"partial_answers": [result]}
    except Exception as e:
        # 返回错误标记而非抛异常
        error_entry = {
            "source": doc.get("source", "unknown"),
            "relevant": False,
            "error": str(e),
            "answer": None,
        }
        print(f"[警告] 处理 {doc.get('source')} 失败: {e}")
        return {"partial_answers": [error_entry]}
        # 注意：不要 raise！否则可能导致整个 map 失败
```

### 6.4 进度追踪

对于长时间运行的 MapReduce，进度追踪非常重要：

```python
async def tracked_mapreduce(graph, items, question):
    """带进度回调的 MapReduce"""
    total = len(items)
    completed = 0

    def on_complete(result):
        nonlocal completed
        completed += 1
        progress = completed / total * 100
        print(f"\r进度: [{completed}/{total}] ({progress:.0f}%)", end="")

    # 使用 callbacks 或自定义机制追踪进度
    result = await graph.amap(
        {"documents": items, "question": question},
        {"name": "analyzer"},
    )
    print(f"\n完成! 共处理 {total} 个项目")
    return result
```

### 6.5 MapReduce 与 Fan-out/Fan-in 的关系

| 特性 | Send Fan-out/Fan-in | map()/amap() |
|------|---------------------|--------------|
| **抽象层级** | 底层原语 | 高层封装 |
| **适用场景** | 不同节点处理不同任务 | 相同节点处理不同数据 |
| **灵活性** | 高 — 各分支可完全不同 | 中 — 所有分支执行同一逻辑 |
| **代码量** | 较多 | 较少 |
| **底层关系** | `map()` 内部就是用 Send 实现的 | Send 的语法糖 |

简单来说：**`map()` 是专门针对"对列表每个元素做相同操作"这一常见模式的便捷 API**，其底层仍然依赖 Send 机制。

---

## 本章小结

本章系统介绍了 LangGraph 中的 **MapReduce 模式**，这是一种高效的批量数据处理范式：

1. **Map 阶段**通过 `map()` / `amap()` 方法，将一个节点或子图映射到输入列表的每个元素上，实现**自动化的并行分发执行**。列表字段被拆解广播，标量字段自动复制到每个分支。

2. **Reduce 阶段**通过 reducer 函数控制如何将各 Map 分支的输出合并。支持**拼接、投票、加权平均、LLM 摘要、去重融合**等多种策略，应根据具体业务场景选择。

3. **实战案例**展示了完整的批量文档问答系统，包括 15 个文档的并行 RAG 分析和 LLM 综合。性能对比显示，相比串行处理可获得 **5~10 倍**的加速比。

4. **生产环境注意事项**：大量元素时分批处理、错误隔离不传播、进度追踪、资源消耗控制。`map()` 本质上是 Send Fan-out 的一种高层封装，适合"同构并行"场景。

MapReduce 模式是构建**高吞吐量 Agent 系统**的核心设计模式之一，特别适用于批量文档处理、多源数据采集、大规模内容分析等场景。

---

> **参考来源**：黑马程序员 LangChain 课程 - BV178w1z7EHQ（第3章 Agent 进阶）
