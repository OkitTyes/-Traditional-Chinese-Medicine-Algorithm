# -Traditional-Chinese-Medicine-Algorithm

## 研究与工程路线：先闭环闭集检索（对齐/识别）baseline，再扩展开集与VQA

你们的设定是：
- **无框（no box）**：图像级别对齐/识别
- **自由文本答案（open-ended VQA）**：用于可用性验证

因此建议用“双轨”定义任务（学术上更标准，也更容易做对照实验）：

### 1）检索线（对齐/识别指标，优先闭环）
- **任务定义**：Image→Text Retrieval（把“识别”定义成：从固定候选文本库里检索最匹配条目）
- **候选文本库**：药材名称/别名/拉丁名/炮制名/鉴别要点/功效主治/方剂条目等
- **评测指标**：Recall@K（R@1/5/10）、MRR（可选 nDCG）
- **对称验证**：Text→Image Retrieval（审稿更喜欢）

### 2）VQA/对话线（可用性）
- **任务定义**：图文问答（自由文本答案）
- **评测建议**：
  - 规范化后 EM/F1（别名、繁简、炮制名同义）
  - 语义指标（BERTScore/Rouge-L）只做补充，不做唯一结论
  - 人评量表：事实性、专业性、完整性、是否引用图中证据（无框也可做“证据描述”）

---

## 一步闭环：检索 baseline（闭集，zero-shot）

我们提供了一个可直接运行的 **CLIP zero-shot 检索基线**：`tcmvl_retrieval.py`

### 数据格式（jsonl）
每行一个样本，必须包含：
- `image`: 图片路径
- `text`: 候选文本（名称/别名/描述均可）
- `concept_id`: 该条文本对应的概念ID（建议用“规范名ID/药材ID/方剂ID”等）

示例见：`data/example_retrieval.jsonl`

### 运行（输出 i2t/t2i 的 R@K/MRR）
安装依赖：

```bash
pip install -r requirements.txt
```

运行示例（你需要把示例里的 `data/images/...` 换成真实图片路径）：

```bash
python tcmvl_retrieval.py --data data/example_retrieval.jsonl --out retrieval_metrics.json
```

---

## 从闭集到开集：怎么扩展（建议按阶段做）

### 阶段A：闭集（先闭环）
- 候选库固定（例如测试集药材条目固定）
- 目标：稳定得到可复现的 R@K/MRR；对齐算法的提升可直接体现在这些指标上

### 阶段B：开集（再扩展）
开集的关键是**候选库可增长 + 分布外概念**：
- 候选文本库从“测试集固定列表”扩展为“知识库/词典/说明书条目集合”
- 测试时加入新药材/新炮制法/新产地（概念不在训练集）
- 评测拆成：
  - in-domain closed-set（保持与既有工作可比）
  - out-of-domain open-set（证明泛化）

### 阶段C：VQA LoRA 基线（6–7B）
- 选择一个 6–7B 现成 VLM 作为可用性强基线（中文优先可选 Qwen2-VL 7B；通用经典可选 LLaVA 7B）
- 先跑通训练与评测闭环，再用你们 VQA 数据域适配
- 最终报告：检索指标 + VQA 指标 + 人评
