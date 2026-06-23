# SHypX 论文复现 — 完整参考文档

## 论文信息

- **标题**: Explaining Hypergraph Neural Networks: From Local Explanations to Global Concepts
- **作者**: Shiye Su (Stanford), Iulia Duta, Lucie Charlotte Magister, Pietro Liò (Cambridge)
- **状态**: Under Review
- **论文PDF**: `g:\xdu\可解释性\超图可解释性\Explaining Hypergraph NNFrom Local Explanations to Global Concepts.pdf`
- **论文全文提取**: `g:\xdu\可解释性\超图可解释性\paper_full_text.txt`

## 代码仓库

- **位置**: `g:\project\Allset`
- **GitHub**: `https://github.com/sepidism/Allset` (HyperEX 论文的 fork，底层是 AllSet 模型)
- **Python 环境**: `G:/conda_envs/shpyx` (Python 3.8, PyTorch 1.12.1, PyG 2.5.2)
- **运行命令**: `G:/conda_envs/shpyx/python.exe src/run_shypx.py --dataset H-RANDHOUSE`

---

## 一、SHypX 算法核心

### 1.1 局部解释器 (Local Explainer) — 论文 Section 4.1

**目标**: 对目标节点 v，找到解释子超图 G_expl，使其既保真(faithful)又简洁(concise)。

**损失函数** (论文 Eq.2):

```
L(f, G_sub, G_comp, X, v) = λ_pred · D_KL(f(G_sub, X, v) || f(G_comp, X, v)) + λ_size · |G_sub|₁
```

- `G_comp`: v 的 d-hop 计算子超图（搜索空间限制在此范围内）
- `|G_sub|₁`: 子超图中 node-hyperedge link 的数量（L1 范数）
- `λ_pred`, `λ_size`: 控制保真度与简洁性的权衡

**优化方法**:
1. **平均场近似**: 将联合概率分布分解为独立边际概率的乘积
   ```
   Pr(G_sub) ≈ ∏ π_{v,e}  (π_{v,e} = Pr(v∈e 在子图中))
   ```
2. **Gumbel-Softmax 离散采样**: 从 π_{v,e} 可微地采样二值 y_{v,e} ∈ {0,1}
3. **梯度反传**: 将采样子超图通过 frozen hyperGNN，计算 loss，梯度反传更新 π_{v,e}
4. **后处理**: 取优化过程中 loss 最低的子超图，只保留包含目标节点 v 的连通分量

**论文关键超参**:
- 初始化 π ≈ 0.95（即所有 link 初始大概率保留）
- Gumbel-Softmax temperature = 1.0
- Adam 优化器, lr=0.01, 400 epochs
- λ_pred = 1, λ_size = 0.05 (合成数据) / 0.005 (真实数据)

### 1.2 全局解释器 (Global Explainer) — 论文 Section 4.2

**流程**:
1. **K-Means 聚类**: 在 hyperGNN 的 latent space {z_v} 上聚类，每个簇是一个"概念"
2. **代表节点**: 每个概念 c 选离聚类中心最近的节点 v*_c
3. **概念解释**: 对 v*_c 运行局部解释器 → G_expl(v*_c)
4. **类别映射**: 多数投票 MajorityVote(c) → ClassExplanation(y)

**论文关键超参**:
- k = 10 (大部分数据集) / k = 15 (H-COMMHOUSE)

### 1.3 评估指标 — 论文 Section 5.2

**广义保真度** (Eq.8):
```
Fid^s_- = (1/N) Σ s(p(G_expl), p(G_comp))   // 越低越好，说明 G_expl 足以复现预测
```

相似度函数 s 有四种:
| 名称 | 含义 | 公式 |
|------|------|------|
| `acc` | 准确率匹配 | 1(argmax 相同) |
| `kl` | KL 散度 | D_KL(p || q) |
| `tv` | 全变差距离 | 0.5 · ||p - q||₁ |
| `xent` | 交叉熵 | -Σ p(c) log q(c) |

**简洁性指标**:
- Size = |G_expl|₁ (解释子超图的 node-hyperedge link 数量)
- Density = |G_expl|₁ / |G_comp|₁ (越小越好)

**概念完备性** (Concept Completeness):
- 对 concept label 做多数投票分类的准确率
- 越接近任务精度说明概念提取越成功

### 1.4 合成数据集 — 论文 Section 5.1 + Appendix B

**构造方式**: Base + Motif + Perturbations

**Base 图**:
- **Random base**: 随机二分图 → 最大连通分量 → 逆星展开 → 随机超图
- **Tree base**: 确定性 3-uniform 超图，每个超边包含父节点 + 2 个子节点

**Motif**:
- **House**: 5 节点，3 类（top/Class1, middle/Class2, bottom/Class3）
- **Cycle**: 6 节点，全 Class1
- **Grid**: 3×3=9 节点，全 Class1

**论文 Table 3 数据规模**:

| 数据集 | Base | Motif | #Base Nodes | #Motifs | #Perturbations | #Classes | Features |
|--------|------|-------|:-----------:|:-------:|:--------------:|:--------:|----------|
| H-RANDHOUSE | random | house | 312 | 100 | 80 | 4 | 随机高斯 |
| H-COMMHOUSE | random×2 | house | 648 | 200 | 80 | 8 | 双峰正态 |
| H-TREECYCLE | tree | cycle | 255 | 80 | 80 | 2 | 随机高斯 |
| H-TREEGRID | tree | grid | 255 | 80 | 80 | 2 | 随机高斯 |

> **注意**: 论文说 H-RANDHOUSE/H-TREECYCLE/H-TREEGRID 用 all-ones 特征，但实验发现 1 维全 1 特征在 PMA attention 下模型无法学习（所有节点得到相同 key/value）。当前实现改为 16 维随机高斯特征，标签仍完全由结构决定。

### 1.5 论文主要结果

**Table 1 — 合成数据集**:

| 数据集 | 方法 | Fid^acc_- | Fid^kl_- | Size | Density |
|--------|------|:---------:|:--------:|:----:|:-------:|
| H-RANDHOUSE | SHypX | **0.01** | **0.04** | 9.2 | 0.19 |
| | HyperEX | 0.86 | 1.09 | 0.0 | 0.01 |
| H-COMMHOUSE | SHypX | **2e-3** | **0.02** | 9.2 | 0.20 |
| | HyperEX | 0.79 | 3.63 | 0.1 | 0.02 |
| H-TREECYCLE | SHypX | **3e-3** | **0.01** | 5.6 | 0.22 |
| | HyperEX | 0.35 | 0.64 | 0.0 | 0.00 |
| H-TREEGRID | SHypX | **0.01** | **0.02** | 15.1 | 0.45 |
| | HyperEX | 0.66 | 1.63 | 13.4 | 0.46 |

**Table 2 — 真实数据集** (代表性结果):

| 数据集 | Fid^kl_- (SHypX) | Fid^kl_- (HyperEX) | SHypX Size | SHypX Density |
|--------|:---:|:---:|:---:|:---:|
| CORA | **5e-4** | 0.03 | 1.4 | 0.61 |
| COAUTHORCORA | **3e-4** | 0.05 | 2.3 | 0.15 |
| COAUTHORDBLP | **3e-4** | 0.05 | 2.3 | 0.15 |
| ZOO | **0.01** | 0.09 | 6.7 | 0.01 |

**Table 4 — AllSetTransformer 在合成数据集上的精度**:

| 数据集 | Accuracy |
|--------|:--------:|
| H-RANDHOUSE | 95.09% |
| H-COMMHOUSE | 97.15% |
| H-TREECYCLE | 83.95% |
| H-TREEGRID | 90.05% |

---

## 二、代码模块映射

### 已完成 ✅

| 论文章节 | 论文内容 | 代码文件 | 核心类/函数 |
|----------|---------|---------|------------|
| §4.1 优化方法 | Gumbel-Softmax 采样 | `src/explainer/sampling.py` | `GumbelSoftmaxSampler`, `gumbel_softmax_sample()` |
| §4.1 局部解释器 | 损失函数+梯度优化+后处理 | `src/explainer/local_explainer.py` | `LocalExplainer.explain()`, `get_computation_subhypergraph()`, `find_connected_component()` |
| §4.2 全局解释器 | K-Means+代表节点+类别映射 | `src/explainer/global_explainer.py` | `GlobalExplainer.explain()`, `extract_concepts()`, `concept_to_class_mapping()` |
| §5.2 评估指标 | 广义保真度+Size+Density | `src/explainer/metrics.py` | `fidelity_minus()`, `generalized_fidelity()`, `explanation_size()`, `explanation_density()` |
| §5.1+App.B | 合成数据集生成 | `src/synthetic/base.py` | `generate_random_base()`, `generate_tree_base()` |
| §5.1+App.B | Motif 生成 | `src/synthetic/motifs.py` | `generate_house_motif()`, `generate_cycle_motif()`, `generate_grid_motif()` |
| §5.1+App.B | 数据集组装 | `src/synthetic/datasets.py` | `H_RANDHOUSE()`, `H_COMMHOUSE()`, `H_TREECYCLE()`, `H_TREEGRID()` |
| §5.3 实验 | 训练+解释+评估流程 | `src/run_shypx.py` | `main()`, `run_local_explanations()`, `run_global_explanations()` |

### 底层模型（AllSet / HyperEX 原始代码，已存在）

| 文件 | 内容 | SHypX 中的角色 |
|------|------|---------------|
| `src/models.py` | SetGNN (AllSetTransformer), CEGCN, HGNN, HCHA 等 | 被解释的 hyperGNN 模型 |
| `src/layers.py` | PMA, HalfNLHconv, MLP, HypergraphConv 等 | 模型的基础层 |
| `src/preprocessing.py` | ExtractV2E, Add_Self_Loops, norm_contruction | 数据预处理 |
| `src/train.py` | 原始训练脚本（含参数解析） | 参考，不一定直接使用 |

---

## 三、现有实现与论文的关键差异

### 3.1 特征维度问题

| 项目 | 论文 | 当前实现 | 影响 |
|------|------|---------|------|
| H-RANDHOUSE 特征 | "none (ones)" — 1 维全 1 | 16 维随机高斯 `N(0,1)` | 论文声称 95% 精度，但 1 维全 1 在 PMA attention 下物理上不可学习 |
| H-TREECYCLE 特征 | 同上 | 16 维随机高斯 | 同上 |
| H-TREEGRID 特征 | 同上 | 16 维随机高斯 | 同上 |

**技术原因**: PMA attention 的 key/value 投影是 `Linear(1, hidden)`，所有节点输入相同 → 所有节点 key/value 相同 → 注意力均匀 → V→E 聚合后所有同度超边表示相同 → 无法区分 motif 内不同位置的节点。多轮消息传递能缓解但不能完全解决。16 维随机特征让每个节点初始表示不同，模型才能真正学习结构模式。

### 3.2 模型精度差距

| 数据集 | 论文精度 | 当前精度 | 可能原因 |
|--------|:---:|:---:|------|
| H-RANDHOUSE | 95% | ~74% (500 ep) | 数据规模不同、训练超参差异 |
| H-TREECYCLE | 84% | 未完整测试 | 同上 |

### 3.3 PyG 兼容性修改

以下文件被修改以适配 PyG 2.5（原本依赖 PyG 1.6 + torch_scatter）:

| 文件 | 修改内容 |
|------|---------|
| `src/layers.py` | `torch_scatter` → PyG 内置 `scatter` 兜底；aggregate 签名适配 PyG 2.5 |
| `src/preprocessing.py` | 同上；`rand_train_test_idx` 索引 dtype 修复 |
| `src/models.py` | 无修改 |
| `src/train.py` | `torch_sparse` 改为可选导入 |

---

## 四、待完成任务

### 任务 A：数据集规模验证
**文件**: `src/synthetic/datasets.py`
**当前参数**: H_RANDHOUSE 已调整默认值为 `num_base_nodes=400, num_motifs=100, num_perturbations=80`；H_TREECYCLE/H_TREEGRID 已调整 `tree_depth=7`
**需验证**: 生成后各数据集的 base 节点数是否接近论文 Table 3 的数值

### 任务 B：AllSet 模型精度调优
**目标**: 在合成数据集上达到论文 Table 4 的精度
**关键超参**: `All_num_layers` (1 or 3), `heads` (1/4/8), `MLP_hidden` (16/64/256), `lr`, `epochs`
**参考**: 论文说 "three layers deep, sum aggregation, no dropout, dim-16 message passing"

### 任务 C：可视化
- **Figure 3**: Faithfulness vs Concision 曲线 — 扫描不同 `λ_pred/λ_size` 比值，画 Fid^kl vs Size 图
- **Figure 4**: 概念子超图可视化 — 用 networkx/matplotlib 绘制每个概念的代表性子超图结构

### 任务 D：环境补完
- 安装 `pandas`（加载真实数据集需要）
- 修复 `torch-scatter`/`torch-sparse` 的 DLL 加载警告
- 配置 pip 国内镜像源解决 SSL 问题

---

## 五、常用命令速查

```bash
# 激活环境（需要通过 conda run 或直接指定 python 路径）
G:/conda_envs/shpyx/python.exe src/run_shypx.py --dataset H-RANDHOUSE

# 快速测试合成数据集生成
G:/conda_envs/shpyx/python.exe -c "
import sys; sys.path.insert(0,'src')
from synthetic.datasets import H_RANDHOUSE
d = H_RANDHOUSE(seed=42)
print(f'nodes={d.n_x.item()}, hyperedges={d.num_hyperedges.item()}')
"

# 单独测试局部解释器
G:/conda_envs/shpyx/python.exe -c "
import sys; sys.path.insert(0,'src')
from explainer.local_explainer import LocalExplainer, get_computation_subhypergraph
# ... (需要先加载已训练的 model 和 data)
"
```

## 六、关键论文引用

| 论文 | 与本项目关系 |
|------|-------------|
| Chien et al. 2021 "You are AllSet" (ICLR 2022) | AllSet 超图神经网络模型 — 本项目的基座模型 |
| Maleki et al. 2023 "Learning to Explain Hypergraph Neural Networks" | HyperEX — 唯一的已有超图解释器（代码未公开），SHypX 的 baseline |
| Su et al. "Explaining Hypergraph NN" (Under Review) | SHypX — **我们正在复现的论文** |
| Ying et al. 2019 "GNNExplainer" | 图解释器鼻祖，SHypX 的合成数据集设计受其启发 |
| Magister et al. 2021 "GCExplainer" | 图全局解释器，SHypX 的概念提取受其启发 |
| Jang et al. 2016 "Gumbel-Softmax" | SHypX 使用的离散采样技术 |
