# MonetAgent-Hierarchical 实现方案

## Context

MonetGPT 当前是单轮4阶段 pipeline（white-balance-tone-contrast → color-temperature → hsl → local-editing），在复杂多步指令下容易漂移、过编辑、丢失身份。目标：升级为分层 Agent 系统，支持多轮交互（10-50轮稳定）、中途用户干预（replan/rollback）、结构化记忆（Context Folding），结合已有的局部蒙版编辑（Direction 1），形成 "Puzzle-aware Hierarchical Local Retoucher"。

参考工作：Agent Banana（Context Folding + Layer Decomposition）、CoSTA（子任务树 + A* 搜索）、LangGraph（状态图 + 检查点）。

**设计决策**：
- **VLM 后端**：Planner/Quality 节点复用现有 `InferenceEngine` 的 OpenAI 兼容 API（同一个 model endpoint），不引入额外 API 依赖
- **Quality 检查频率**：仅在每轮所有子目标完成后做一次质量检查（非每步检查），减少 VLM 调用开销
- **实现分期**：第一期 4 周出 Demo（核心 Agent 循环 + Gradio + 基础多轮），第二期 4 周补全（数据合成 + 评估 + 加固）

---

## 架构总览

```
User (Gradio / CLI)
    │
    ▼
┌─ LangGraph StateGraph ─────────────────────────────┐
│                                                      │
│  START → [Planner] → [Executor] → [Quality] ─┐      │
│              ▲           ▲                    │      │
│              │           │              pass & 有    │
│         replan/      modify          下一步? → loop  │
│         rollback                          │         │
│              │                       全部完成        │
│              └── [HumanReview] ◄──────────┘         │
│                       │                              │
│                  approve                             │
│                       ▼                              │
│              [ContextFold] → END                     │
└──────────────────────────────────────────────────────┘
```

**核心原则**：现有单轮 pipeline 完全不动。Agent 系统是上层包装，通过调用 `ImageEditingPipeline.execute_edit()` 和 `MaskedExecutor` 执行实际编辑。

---

## 新增文件

### 1. `agent/state.py` — 核心数据结构

定义 LangGraph 的 `AgentState` TypedDict 和辅助 dataclass：

```python
# 关键数据结构
@dataclass
class AssetNode:          # 图像版本 DAG 节点
    uri: str              # 文件路径
    parent_uri: str       # 父图像路径
    transform_summary: str
    adjustments: dict
    turn_id: int

@dataclass
class ToolContext:         # 瞬时工具记录（每轮清除）
    tool_name: str
    params: dict
    thought: str
    result_summary: str
    success: bool

@dataclass
class ActionRecord:        # 持久高层记录（累积保留）
    turn_id: int
    intent: str
    plan_summary: str
    outcome: str           # completed | partial | rolled_back
    validated_asset_uri: str

@dataclass
class SubGoal:             # Planner 输出的子目标
    id: int
    stage_type: str        # "global" | "local"
    operation_category: str # "white-balance-tone-contrast" | "color-temperature" | "hsl" | "local-editing"
    description: str
    adjustments: dict      # global ops 的 {op: value}
    local_specs: list      # local ops 的 JSON array
    status: str            # pending | completed | failed
    retry_count: int

class AgentState(TypedDict):
    session_id: str
    original_image_path: str
    current_image_path: str
    style: str
    turn_id: int
    user_message: str
    sub_goals: list                              # List[SubGoal]
    current_sub_goal_idx: int
    accrued_dehaze: float
    quality_score: float
    quality_pass: bool
    quality_assessment: str
    asset_graph: Annotated[list, append_reducer]  # List[AssetNode] 追加式
    tool_contexts: list                           # List[ToolContext] 每轮清除
    action_history: Annotated[list, append_reducer] # List[ActionRecord] 追加式
    human_decision: str                           # approve | modify | rollback | replan
    human_modifications: dict
    error_message: str
    is_complete: bool
```

### 2. `agent/graph.py` — StateGraph 构建

```python
def build_agent_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("quality", quality_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("context_fold", context_fold_node)
    builder.add_node("error_handler", error_handler_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner)  # → executor | context_fold | error_handler
    builder.add_edge("executor", "quality")
    builder.add_conditional_edges("quality", route_after_quality)  # → executor(下一步) | human_review | planner(replan)
    builder.add_conditional_edges("human_review", route_after_human) # → context_fold | planner | executor
    builder.add_edge("context_fold", END)
    builder.add_edge("error_handler", "planner")

    return builder.compile(checkpointer=checkpointer or MemorySaver())
```

**路由逻辑**（Quality 仅在每轮最后一步后触发）：
- `route_after_planner`: 有子目标 → executor；空计划或 is_complete → context_fold；error → error_handler
- executor 内部循环执行全部子目标，完成后进入 quality
- `route_after_quality`: score >= 0.6 → human_review；score < 0.6 且 retry < 2 → planner(replan)；retry >= 2 → human_review
- `route_after_human`: approve → context_fold；rollback/replan → planner；modify → executor

### 3. `agent/nodes.py` — 5 个节点实现

**planner_node**:
- 输入：user_message + current_image + `_build_planner_context(state)` 生成的压缩历史
- 调用 `InferenceEngine` 的 VLM API，使用专门的 planner prompt
- 输出：`sub_goals: List[SubGoal]`，每个含 stage_type + operation_category + adjustments/local_specs
- replan 时注入 quality_assessment 作为额外上下文

**executor_node**:
- 读取 `sub_goals[current_sub_goal_idx]`
- global → 写 JSON config → `ImageEditingPipeline.execute_edit(config_path, src, output)`（走现有 non-GIMP + GIMP 路径）
- local → 写 JSON array config → `execute_edit()` 自动 dispatch 到 `_execute_local_edit()` → `MaskedExecutor`
- 值缩放走现有 `config.py:get_processed_predictions()`
- 记录 AssetNode + ToolContext，推进 current_sub_goal_idx

**quality_node**:
- 发送 original + current 双图给 VLM
- 输出 0-1 分数 + 文字评估
- 阈值 0.6 控制 pass/fail

**human_review_node**:
- 使用 LangGraph `interrupt()` 暂停，将当前图像 + 质量评估 + 子目标摘要推送给 UI
- 用户选择 approve/modify/rollback/replan
- rollback：遍历 asset_graph 回溯到指定 turn 的图像

**context_fold_node**:
- 追加 ActionRecord 到 action_history
- 清空 tool_contexts（瞬时数据）
- 推进 turn_id
- 重置 human_decision 等临时状态

### 4. `agent/context_folding.py` — 3 层记忆压缩

```
Layer 1 - AssetContext: 图像版本 DAG（uri, parent, transform）
Layer 2 - ToolContext:  瞬时工具参数（每轮清除）
Layer 3 - ActionContext: 持久高层摘要（累积）
```

**折叠策略**：
- 最近 3 轮：保留完整 ActionRecord
- 更早轮次：压缩为单行摘要 "T{n}:{outcome}"
- AssetGraph：只保留 current_image_path 的祖先链 + original
- 总 token 预算：~2000 tokens（planner prompt 约 500 token，留足给用户消息+图像）

**关键函数**：`fold_context_for_prompt(asset_graph, action_history)` → 生成给 planner 的压缩历史字符串。

### 5. `agent/prompts.py` — Planner/Quality 系统 prompt

- **PLANNER_SYSTEM_PROMPT**: 指导 VLM 输出 1-6 个 SubGoal 的 JSON array，含 stage_type、operation_category、adjustments/local_specs
- **QUALITY_SYSTEM_PROMPT**: 指导 VLM 对比 original vs current，输出 0-1 分数 + 评估文字
- **_create_planner_prompt()**: 注入 folded context、user_message、style、replan feedback
- **_create_quality_prompt()**: 注入 user_intent、sub_goal_desc、adjustments

### 6. `agent/session.py` — 会话管理

- `create_session(image_path, style)` → 初始化 AgentState，创建 session 目录，复制原图
- 使用 UUID 生成 session_id，对应 LangGraph 的 thread_id

### 7. `agent/cli.py` — 交互式 CLI

```
> make it warmer and brighter
[Planner] 分解为 2 个子目标...
[Executor] Stage 1: Temperature+40, Exposure+20 → done
[Quality] Score: 0.82 ✓
Result: sessions/abc123/turn0_step0.tif
> the sky is too bright now
...
```

### 8. `app.py` — Gradio 交互 Demo

布局：
- 左列：原图 + 当前图 对比显示
- 中列：聊天历史 + 输入框
- 右列：JSON 调整历史 + 子目标状态
- 底部：中间图像画廊（asset_graph 可视化）
- 审查面板（interrupt 时显示）：质量分数 + 评估 + approve/modify/rollback/replan 按钮

使用 SqliteSaver 做持久化检查点，支持浏览器刷新后恢复。

### 9. `configs/agent_config.yaml` — Agent 配置

```yaml
planner:
  max_sub_goals: 6
  temperature: 0.2
  max_retries_per_sub_goal: 2
quality:
  min_pass_score: 0.6
  auto_approve_threshold: 0.85  # > 此值跳过 human review
context_folding:
  max_history_tokens: 2000
  keep_recent_turns: 3
session:
  base_dir: "./sessions"
  checkpoint_backend: "sqlite"
```

---

## 修改现有文件

### `inference/core.py` — 最小改动
在 `InferenceEngine` 类中新增 1 个方法：
```python
def query_structured(self, image_path, system_prompt, user_prompt, temperature=0.2):
    """通用结构化 VLM 查询，供 planner/quality 节点使用。"""
    # 复用现有 encode_image_to_base64 + client.chat.completions.create
```
**不修改** `StagedEditingPipeline` 和其他任何现有方法。

### `pipeline/core.py` — 最小改动
在 `ImageEditingPipeline` 类中新增 1 个便利方法：
```python
def execute_single_stage(self, adjustments, src_path, output_path, is_local=False):
    """执行单阶段编辑，agent executor 调用。自动处理临时 config 文件。"""
    # 写临时 JSON → 调用 self.execute_edit() → 清理
```
**不修改**现有 `execute_edit()`、`_execute_local_edit()` 等方法。

### `dataset/constants.py`
追加多轮类型常量：
```python
MULTI_TURN_TYPES = ["global_init", "global_refine", "local_edit", "rollback_correct", "style_shift"]
```

### `dataset/combine_jsons.py`
在 `json_paths` 列表中追加 `'data/sharegpt_puzzle_mt.json'`

### `configs/inference_config.yaml`
追加 `agent:` 部分（不影响现有配置）

---

## 多轮数据合成 Pipeline

### 新文件

**`dataset/generate_puzzle_mt.py`** — 多轮轨迹生成器
- 对每张源图，采样 3-6 轮轨迹：
  - Turn 1: global_init（暖调、提亮等）
  - Turn 2: global_refine（修正过曝高光等）
  - Turn 3: local_edit（局部修肤色等）
  - Turn 4: rollback_correct（撤销 + 替代方案）
  - Turn 5: style_shift（风格转换）
- 每轮生成 config → 调用现有 MaskedExecutor/Pipeline 执行 → 保存中间图
- 输出：完整轨迹 JSON

**`dataset/query_llm_multiturn.py`** — 多轮推理生成
- 发送多轮对话 + 所有中间图给 Gemini
- 生成每轮的分析 reasoning（带上下文感知）

**`dataset/create_datasets_multiturn.py`** — 多轮 ShareGPT 格式化
- 扩展 ShareGPT 为多轮结构：
```
system → user_turn1 → assistant_analysis1 → user_json_prompt1 → assistant_json1 →
user_turn2 → assistant_analysis2 → user_json_prompt2 → assistant_json2 → ...
```

**规模目标**：5000 基图 × 3-5 轮 = 15k+ 轨迹

---

## 新增依赖

```
langgraph>=0.4.0
langgraph-checkpoint>=2.0.0
gradio>=4.0
```

---

## 实现顺序（分两期）

### 第一期：4 周出 Demo

**Week 1: 核心 Agent 骨架**
- [ ] 创建 `agent/` 包：`__init__.py`, `state.py`, `graph.py`, `session.py`
- [ ] 实现 AgentState TypedDict + 全部 dataclass（AssetNode, ToolContext, ActionRecord, SubGoal）
- [ ] 构建 StateGraph（placeholder 节点），用 MemorySaver 验证 invoke/resume 循环
- [ ] 添加 `InferenceEngine.query_structured()` 到 `inference/core.py`
- [ ] 添加 `ImageEditingPipeline.execute_single_stage()` 到 `pipeline/core.py`
- [ ] 创建 `configs/agent_config.yaml`

**Week 2: Planner + Executor 节点**
- [ ] 编写 `agent/prompts.py`：PLANNER_SYSTEM_PROMPT + _create_planner_prompt()
- [ ] 实现 planner_node：VLM 调用 → SubGoal JSON 解析（复用 `extract_json_array_from_response()`）
- [ ] 实现 executor_node：全局子目标循环 + 调用 `execute_single_stage()`
- [ ] 实现 executor 对 local 子目标的分发（走 `_execute_local_edit` 路径）
- [ ] 端到端测试：单张图 → planner 分解 → executor 执行 → 输出编辑图

**Week 3: Quality + Context Folding + 基础多轮**
- [ ] 编写 QUALITY_SYSTEM_PROMPT + _create_quality_prompt()
- [ ] 实现 quality_node：双图对比 → 分数 + 评估（每轮末尾触发）
- [ ] 实现 `agent/context_folding.py`：fold_context_for_prompt(), prune_asset_graph()
- [ ] 实现 context_fold_node：追加 ActionRecord + 清除 ToolContext + 推进 turn_id
- [ ] 实现 error_handler_node
- [ ] 连通完整循环：planner → executor → quality → context_fold
- [ ] 测试 5 轮多轮 session 稳定执行

**Week 4: Gradio Demo + 基础 Human Review**
- [ ] 实现 human_review_node（简化版：approve / replan 两种选择）
- [ ] 实现 `app.py`：原图+当前图对比、聊天历史、输入框、审查面板
- [ ] 对接 LangGraph `interrupt()` + `Command(resume=...)` 在 Gradio 中的渲染
- [ ] 中间图像画廊（显示 asset_graph）
- [ ] 端到端 Demo 测试：3-5 轮交互式修图

### 第二期：4 周补全

**Week 5: 完整 Human-in-the-Loop + Rollback**
- [ ] 扩展 human_review_node：支持 rollback（asset_graph 回溯）+ modify（用户参数覆盖）
- [ ] 切换 MemorySaver → SqliteSaver 持久化
- [ ] 实现 `agent/cli.py` 交互式 CLI
- [ ] Rollback 正确性测试

**Week 6: 多轮数据合成**
- [ ] 实现 `dataset/generate_puzzle_mt.py`：轨迹模板 + 随机采样 + GT 执行
- [ ] 实现 `dataset/query_llm_multiturn.py`：多轮 reasoning 生成
- [ ] 实现 `dataset/create_datasets_multiturn.py`：多轮 ShareGPT 格式化
- [ ] 修改 `dataset/combine_jsons.py` + `dataset/constants.py`
- [ ] 生成 500 条验证轨迹

**Week 7: 评估**
- [ ] 单轮回归测试（PPR10K 原有流程不退化）
- [ ] 多轮评估 50 条轨迹：IC, SSIM-OM, 漂移率
- [ ] Context overflow 压测（50 轮）
- [ ] Rollback 正确性验证

**Week 8: 加固 + 优化**
- [ ] 修复评估发现的问题
- [ ] VLM 调用优化：quality score > auto_approve_threshold 时跳过 human review
- [ ] GPU 内存优化：MaskGenerator 在非 local 阶段 unload
- [ ] 全流程集成测试
- [ ] 大规模数据合成（扩到 5000+ 条轨迹）

---

## 验证方案

| 指标 | 目标 | 方法 |
|------|------|------|
| 单轮 IC | >= 0.85 | VLM-as-judge，100 张测试图 |
| 5 轮后 IC | >= 0.80 | VLM-as-judge，50 条多轮 session |
| SSIM-OM | >= 0.80 | 结构相似度（未编辑区域） |
| 50 轮 context overflow | 0% | 监控 planner context token 数 |
| Rollback 正确性 | 100% | 回滚图像 vs 检查点完全一致 |
| 单轮延迟（不含 VLM） | < 10s | 4K 图像，单 A100 |

---

## 最终文件树

```
agent/                          # 新包
├── __init__.py
├── state.py                    # AgentState + dataclasses
├── graph.py                    # StateGraph 构建
├── nodes.py                    # 5 个节点实现
├── prompts.py                  # Planner/Quality prompts
├── context_folding.py          # 3 层记忆压缩
├── session.py                  # 会话管理
└── cli.py                      # 交互式 CLI
app.py                          # Gradio Demo
configs/agent_config.yaml       # Agent 配置
dataset/generate_puzzle_mt.py   # 多轮轨迹生成
dataset/query_llm_multiturn.py  # 多轮 reasoning
dataset/create_datasets_multiturn.py  # 多轮 ShareGPT

# 修改文件（最小改动）
inference/core.py               # +query_structured()
pipeline/core.py                # +execute_single_stage()
dataset/constants.py            # +MULTI_TURN_TYPES
dataset/combine_jsons.py        # +puzzle_mt 路径
configs/inference_config.yaml   # +agent section
```
