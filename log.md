# Track 1 Experiment Log

Last updated: 2026-05-16

Track: Portrait Composition Understanding  
Dataset size: 2000 test images  
Submission format: one JSON list with 2000 items

## 1. 评判标准说明

### 1.1 SRCC

SRCC 是 Spearman Rank Correlation Coefficient，中文通常叫 Spearman 秩相关系数。

它衡量的是预测 `total_score` 和真实 `total_score` 的排序一致性。换句话说，它更关心“哪些图片应该排在前面，哪些图片应该排在后面”，不直接关心预测分数的绝对数值是否完全一致。

如果没有并列排名，常见形式是：

```text
SRCC = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
```

其中 `d_i` 是第 `i` 张图片在预测排序和真实排序中的排名差。实际实现遇到并列值时，一般等价于对 rank 后的预测值和真实值计算 Pearson correlation。

对本任务的意义：

- 只作用于 `total_score`。
- 如果 2000 张图几乎都预测成同一个分数，例如 1999 张都是 62，SRCC 会非常低。
- 如果预测分数的绝对值不准，但排序大体正确，SRCC 仍然可以较高。
- 当前优化 `total_score` 的第一目标应是提升排序质量。

### 1.2 PLCC

PLCC 是 Pearson Linear Correlation Coefficient，中文通常叫 Pearson 线性相关系数。

它衡量的是预测 `total_score` 和真实 `total_score` 的线性一致性。它不仅关心排序，也关心预测分数是否能用一个近似线性关系对应真实分数。

常见形式是：

```text
PLCC = cov(pred, gt) / (std(pred) * std(gt))
```

对本任务的意义：

- 只作用于 `total_score`。
- 如果预测分数几乎是常数，`std(pred)` 接近 0，PLCC 会非常低。
- 如果预测排序不错，但分布形状和真实分布差很多，PLCC 仍可能低于 SRCC。
- Rank/quantile mapping 可以改善分布，但如果排序代理本身不准，也会把错误排序固定下来。

### 1.3 Criteria Acc

Criteria Acc 是 13 个构图/美学维度的分类准确率。

每张图片有 13 个 criteria，每个 criterion 需要输出一个等级：

```text
A = Poor
B = Medium
C = Good
```

总计评估数量是：

```text
2000 images * 13 criteria = 26000 labels
```

对本任务的意义：

- 它是 exact match accuracy，只要某个 criterion 的 A/B/C 不等于真实标签，就算错。
- `NOT_RES` 或非法输出会被视为错误。
- 如果模型过度输出 `C`，可能在高质量图片上表现不错，但对中低分样本区分不足。
- Criteria Acc 会间接影响 `total_score` 后处理，因为当前 RankIQA-style 分数依赖 13 个 criteria 的排序代理。

### 1.4 Answer Acc

Answer Acc 是视觉问答四选一准确率。

每张图片有一个问题和四个选项：

```text
A / B / C / D
```

评估方式是 2000 张图片逐题 exact match：

```text
Answer Acc = correct_answers / 2000
```

对本任务的意义：

- 它只评估 `answer` 字段。
- 与 `total_score` 的 SRCC/PLCC 没有直接数学关系。
- 当前 Answer Acc 已经在 `0.79-0.80` 附近，不是主要瓶颈。
- 后续不建议优先改 answer prompt，除非有明确证据说明 answer 下滑。

## 2. 当前结论

1. 当前主要瓶颈仍是 `total_score`。
2. RankIQA-style 后处理已经把 SRCC/PLCC 从约 `0.01` 拉回到 `0.65`，说明“恢复排序分布”方向有效。
3. 当前结果低于 baseline `SRCC ~= 0.70`，主要原因是排序代理太粗：criteria 只有 A/B/C 三档，且当前预测明显偏向 `C`。
4. Criteria Acc 为 `0.40`，说明 criteria 本身还有较多错误；基于错误 criteria 做总分排序会天然受限。
5. Answer Acc 为 `0.79`，接近历史 `0.80`，不是优先优化目标。

## 3. Daily Log

## 2026-05-07

### 3.1 修改内容

本次修改集中在 `qwen-vl-finetune/evaluation/evaluation_multi.py`，目标是不改模型权重，只优化 Track 1 推理和 `total_score` 后处理。

1. 推理数据接口改为服务器真实路径：
   - `DATASET_ROOT = "/root/autodl-tmp/PortraitCraft_dataset"`
   - `INPUT_JSON = os.path.join(DATASET_ROOT, "track_1_test.json")`
   - `TRAIN_JSON = os.path.join(DATASET_ROOT, "track_1_train.json")`
   - `IMAGES_PATH = DATASET_ROOT`
   - `OUTPUT_JSON = "/root/autodl-tmp/CVPR/qwen-vl-finetune/track_1_test_res.json"`

2. 增加图片路径自动解析：
   - 新增 `IMAGE_SUBDIRS = [""] + [f"images_{i:02d}" for i in range(11)]`
   - 新增 `resolve_image_path(image_name)`
   - 作用：当 `track_1_test.json` 里的 `image_path` 只有文件名时，自动在 `images_00` 到 `images_10` 中寻找图片。

3. 调整 prompt 中的 criteria 输入：
   - 原逻辑会把输入 JSON 里的 `level=A/B/C/x` 拼进 prompt。
   - 新逻辑只列出 13 个 criteria 名称，让模型基于图像重新判断。
   - 目标是降低复制模板值或旧伪标签的风险，尤其避免 `total_score` 继续贴近模板分。

4. 迁移 RankIQA 思路做 `total_score` 后处理：
   - 新增 `LEVEL_TO_SCORE`、`CRITERION_WEIGHTS`、`compute_rank_proxy()`。
   - 根据模型预测的 13 个 criteria 生成排序代理分。
   - 在 `merge_results()` 中调用 `apply_rankiqa_scores()`。
   - 若服务器存在 `track_1_train.json`，将测试集排序映射到训练集 `total_score` 分布。
   - 若训练集不可用，则退化为 `criteria proxy * 10` 的整数分。

5. 官方提交格式保持不变：
   - 推理阶段仍输出 `image_path / total_score / criteria / question / options / answer`。
   - 最终仍通过 `convert_json_test.py` 转成官方要求的 `{ "criteria": { name: { "level": "A/B/C" } } }` 格式。

### 3.2 论文依据

1. RankIQA: Learning from Rankings for No-Reference Image Quality Assessment, ICCV 2017  
   Link: https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_RankIQA_Learning_From_ICCV_2017_paper.pdf  
   迁移点：该工作强调从相对排序中学习图像质量判断。Track 1 的 SRCC 直接评估 `total_score` 排序相关性，因此本次不再完全信任单张图生成的绝对分，而是用 criteria 生成排序代理分，再映射到训练集分布。

2. NIMA: Neural Image Assessment, arXiv:1709.05424  
   Link: https://arxiv.org/abs/1709.05424  
   参考点：NIMA 将图像美学视为主观质量分布问题，而不是只预测单个均值。当前实现没有完整迁移分布预测，只借鉴“不要只依赖一个直接回归分”的思路，将 `total_score` 交给后处理校准。

3. MUSIQ: Multi-scale Image Quality Transformer, arXiv:2108.05997  
   Link: https://arxiv.org/abs/2108.05997  
   参考点：图像质量/美学评估应尽量保留原始比例和全局构图。当前代码使用 `resize_keep_aspect(image_path, 2048)`，没有中心裁剪，符合全局构图判断需求，因此没有额外改动裁剪策略。

4. PortraitCraft: A Benchmark for Portrait Composition Understanding and Generation, arXiv:2604.03611  
   Link: https://arxiv.org/abs/2604.03611  
   参考点：Track 1 同时评估总分预测、13 个构图属性理解和视觉问答。当前优化重点放在 `total_score`，因为历史提交中 SRCC/PLCC 极低的直接原因是总分几乎为常数。

### 3.3 理想效果

1. `total_score` 不再集中在单一模板值，例如 1999 张都是 62。
2. 测试集总分形成接近训练集的合理分布，图片之间有稳定排序。
3. SRCC 和 PLCC 明显高于旧提交的 `0.01`。
4. Criteria Acc 和 Answer Acc 不应因总分后处理明显下降。
5. 最终提交 JSON 仍完全符合官方格式，包含 2000 条结果。

### 3.4 实际效果

2026-05-07 23:59 提交结果：

```text
Participant: borui_yao
ID: 715797
SRCC: 0.65
PLCC: 0.65
Criteria Acc: 0.40
Answer Acc: 0.79
```

对比：

```text
旧错误版本: SRCC ~= 0.01, PLCC ~= 0.01
baseline: SRCC ~= 0.70
当前版本: SRCC = 0.65, PLCC = 0.65
```

本地当前 `track_1_test.json` 统计：

```text
items: 2000
total_score unique values: 61
score range: 19-90
score mean: 62.591
criteria levels: C=20089, B=5352, A=526, NOT_RES=33
answer distribution: A=528, B=516, D=479, C=477
C-count distribution top: 11C=365, 12C=359, 10C=326, 13C=268
unique criteria patterns: 648
```

### 3.5 问题原因

1. `total_score` 已从常数分恢复成有分布的排序分，因此 SRCC/PLCC 从 `0.01` 回到 `0.65`。这说明 RankIQA-style 方向有效。
2. 当前 SRCC 低于 baseline `0.70`，主要原因是排序代理分不够精确：
   - 当前 rank proxy 只使用 `A/B/C` 离散等级，信息粒度较粗。
   - `LEVEL_TO_SCORE = {A/Poor: 3.0, B/Medium: 6.0, C/Good: 8.5}` 是手工设定，没有基于训练集拟合。
   - `CRITERION_WEIGHTS` 也是经验权重，可能与官方真实 `total_score` 权重不一致。
   - 当前预测明显偏向 `C`，例如 `13C` 有 268 张，`11C/12C/13C` 合计 992 张，导致 high-score 区间区分度不足。
3. 直接把测试排序映射到训练集分布会改善 PLCC 分布，但也可能拉低局部排序：
   - 若测试集真实分布与训练集分布不完全一致，quantile mapping 会引入偏差。
   - 若 criteria 预测本身有错，排序映射会放大这些错误。
4. Criteria Acc 为 `0.40`，略低且存在 `NOT_RES=33`，说明当前 prompt 去掉旧 criteria level 后，模型重新判断的 criteria 没有明显优于 baseline。这个改动减少模板污染，但可能损失了旧伪标签对 criteria 的帮助。
5. Answer Acc 为 `0.79`，与之前 `0.80` 接近，说明 answer 链路不是当前主要瓶颈。

### 3.6 下一步改进方案

1. 优先用 `track_1_train.json` 拟合 `LEVEL_TO_SCORE` 和 `CRITERION_WEIGHTS`，替代手工权重。
2. 不优先改 answer，不重跑 answer prompt，避免引入不必要波动。
3. 对 criteria 采用更保守策略：
   - 可以保留旧 level 作为 noisy prior，但要求模型重新判断。
   - 或在后处理阶段将 `NOT_RES` 映射到最接近的合法等级，避免直接错。
4. 对 `total_score` 做小规模 ablation，不先重跑模型，比较以下方案：
   - raw model score
   - criteria weighted score
   - train quantile mapped score
   - baseline score + calibrated delta
5. 下一次提交前，必须先统计最终 JSON：
   - `items == 2000`
   - `total_score` 不应接近常数
   - `NOT_RES == 0`
   - A/B/C 分布不能过度偏向单一等级

### 3.7 运行方式

在服务器上清理旧 part 文件后重新跑，避免脚本误认为已经完成：

```bash
cd /root/autodl-tmp/CVPR/qwen-vl-finetune
rm -f track_1_test_res.json track_1_test_res.json.part*.json
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_multi.py
python convert_json_test.py --input track_1_test_res.json --output track_1_test_res_final.json
```

## 2026-05-09

### 修改内容

本次只修改 `qwen-vl-finetune/evaluation/evaluation_multi.py` 的 prompt，不改模型权重、不改 answer 逻辑、不改 RankIQA-style 后处理。

新增严格分类阈值：

```text
Poor: score < 5
Medium: 5 <= score < 7
Good: score >= 7
```

要求模型对每个 criterion 先估计隐藏的 0-10 分数，但不输出该分数，只输出 `Good / Medium / Poor`。

同时明确：

- `Good` 不等于“还可以”，必须有明确正向证据。
- 普通、混合、有轻微缺陷的情况应判为 `Medium`。
- 明显影响构图的问题应判为 `Poor`，即使图片整体观感还不错。
- 不能因为整体印象好，就把大多数 criteria 都判成 `Good`。

### 论文依据

本次改动主要基于任务官方定义，而不是引入新的模型论文：

- Track 1 明确给出 criteria 等级和分数区间：`A/Poor = [0,5)`, `B/Medium = [5,7)`, `C/Good = [7,10]`。
- 之前 prompt 只写 `Good / Medium / Poor`，没有把这些硬阈值告诉模型，导致分类边界偏软。
- RankIQA-style 后处理依赖 criteria 排序代理，因此 criteria 过度偏 `C` 会直接限制 SRCC 上限。

### 理想效果

1. 降低 `C` 的过度使用，尤其减少 `11C / 12C / 13C` 的高分段堆积。
2. 增加 `B` 和合理的 `A`，让 criteria 分布更接近真实难度。
3. 减少 total_score 排序代理中的并列和近似并列，提高 SRCC。
4. 保持 Answer Acc 基本不变，因为 answer prompt 和输出逻辑没有改。
5. 最终官方 JSON 格式不变。

### 现实状态

尚未重新全量推理和提交，因此没有新的线上指标。当前只能确认代码层面的 prompt 已改为硬阈值版本。

### 风险与原因

1. 该改动可能提升 criteria 严格性，但也可能让模型过度保守，导致 `Good` 变少过头。
2. 如果真实测试集本身高质量图片较多，过度压低 `C` 可能损害 Criteria Acc。
3. 由于仍然只输出三档 A/B/C，total_score 排序粒度仍然有限；该改动只能缓解高分段堆积，不能完全替代训练集拟合权重。

### 下一步

1. 全量跑完后先统计最终 JSON：
   - `items == 2000`
   - `NOT_RES == 0`
   - `C` 占比是否明显低于之前的约 `77%`
   - `11C / 12C / 13C` 是否明显减少
   - `total_score` 是否仍有足够分布
2. 如果 `C` 占比下降但 SRCC 没升，优先做训练集拟合 `LEVEL_TO_SCORE` 和 `CRITERION_WEIGHTS`。
3. 如果 Criteria Acc 明显下降，回退到较软 prompt，只保留 `Good does not mean acceptable` 这一类轻约束。

## 2026-05-12

### 实际提交结果

```text
Participant: borui_yao
ID: 724954
SRCC: 0.65
PLCC: 0.66
Criteria Acc: 0.39
Answer Acc: 0.78
```

对比：

```text
上次提交 (2026-05-07): SRCC=0.65, PLCC=0.65, Criteria Acc=0.40, Answer Acc=0.79
本次提交 (2026-05-12): SRCC=0.65, PLCC=0.66, Criteria Acc=0.39, Answer Acc=0.78
baseline:              SRCC≈0.70
```

### 分析

1. SRCC 与上次持平（0.65），说明严格 prompt 对排序质量没有实质提升。
2. PLCC 微升 0.01，属于噪声范围，不具有参考意义。
3. Criteria Acc 从 0.40 小幅下降到 0.39，说明严格硬阈值 prompt 对模型的 criteria 判断轻微有害，或至少没有帮助。
4. Answer Acc 从 0.79 降到 0.78，属于正常波动，answer 逻辑未改动。
5. 结论：严格 prompt 策略（硬阈值版本）效果不显著，当前瓶颈不在 prompt 措辞，而在排序代理的权重拟合。

### 下一步

1. 优先用 `track_1_train.json` 拟合 `LEVEL_TO_SCORE` 和 `CRITERION_WEIGHTS`，替换当前手工设定值。
2. 不再调整 prompt 措辞，保持当前硬阈值版本作为基线。
3. 对 total_score 后处理做 ablation，比较：
   - 纯 criteria weighted proxy
   - train quantile mapped score
   - 两者结合

### 后续改动（同日，待重跑）

本次只修改 `qwen-vl-finetune/evaluation/evaluation_multi.py`。不改模型权重，不动 answer 逻辑。

#### 改动 1：隐藏浮点分细化排序粒度

- 旧 prompt 只让模型输出 `level: Good/Medium/Poor`，排序代理被 3 档锁死，2000 张图大量近似并列。
- 新 prompt 要求每个 criterion 同时输出 `{"score": <float 0-10>, "level": "<Good|Medium|Poor>"}`，并强调使用细粒度（如 4.2、5.6、7.3、8.7）。
- 排序代理直接消费 `score` 连续值；最终提交仍只用 `level`，格式不变。
- `MAX_NEW_TOKENS` 由 512 提到 768 容纳更长输出。
- `get_score_value()` 已经能优先取 `score`，只要 prompt 引导模型输出该字段即可。

#### 改动 2：训练集拟合权重 + 自适应 LEVEL_TO_SCORE

- 旧 `CRITERION_WEIGHTS` 与 `LEVEL_TO_SCORE` 是手工拍的，未基于训练集分布。
- 新增 `fit_params_from_train()`：
  - 加载 `track_1_train.json`，构建 `X[N,13] = criterion scores`、`y[N] = total_score`。
  - 用 `numpy.linalg.lstsq` 拟合 `total_score = X @ w + b`，得到 13 个回归权重 + 截距。
  - 同时按 [0,5) / [5,7) / [7,10] 分桶，对每个 level 求训练集 score 均值，得到拟合版 `LEVEL_TO_SCORE`。
  - 启动时执行一次，结果缓存到模块全局变量；并行 worker 在 `apply_rankiqa_scores` 中显式触发一次拟合避免竞态。
- `compute_rank_proxy()` 改用拟合权重 + 截距，返回 1-100 量纲的预测分。
- 如果 `track_1_train.json` 缺失或可用样本 < 50，自动回退到手工 `CRITERION_WEIGHTS / LEVEL_TO_SCORE`。
- 顺手修了 `apply_rankiqa_scores` 兜底 bug：旧逻辑在 proxy 为 None 时 fallback 用 `raw_score / 10`，scale 与其他 proxy 不一致；现统一改为取所有有效 proxy 的中位数填充。

#### 论文依据

- RankIQA (ICCV 2017): 强调从相对排序中学习。拟合权重是其在低成本场景下的工程近似——用训练集 ground truth 直接学一组线性映射，而不是端到端再训。
- NIMA (arXiv:1709.05424): 借鉴"评估应做校准而非直接回归"，故继续保留 quantile mapping。

#### 理想效果

1. SRCC 提升：连续 score + 拟合权重让排序粒度从离散 39 档（13×3）变成近似连续，且权重不再偏差。
2. PLCC 提升：拟合权重的 intercept + 量纲与训练集一致，再叠加 quantile mapping。
3. Criteria Acc 不应下降：prompt 仍要求输出硬阈值映射后的 level，与上次一致。
4. Answer Acc 不变：未改 answer 逻辑。

#### 风险

1. 模型可能不严格按 `{"score": x.x, "level": ...}` 输出，导致 `score` 字段缺失；后处理已兼容回退到 level→score 映射。
2. 拟合权重对训练集分布过拟合的风险存在，但 2000 测试集量级下 13 个参数应当远不到过拟合。
3. 新 prompt 输出更长，若 token 超出 768 上限会截断 JSON；目前评估应足够，必要时再提到 1024。

#### 后续 idea 5（VQA → criteria 反向信号）

利用每道 VQA 问题明确针对某个 criterion 的事实，把模型选择的选项作为该 criterion 的额外信号。难度评估：

- 步骤 1：正则解析问题中提到的 criterion 名（简单，命中率高）。
- 步骤 2：选项文本 → level 映射（最难）。三种方案：
  - A. 二次调用模型推理选项含义：安全但 GPU 时间翻倍。
  - B. 用训练集统计 answer ↔ 真实 level 的条件分布：稳但需先在 train 集跑一遍模型。
  - C. 关键词启发式（clear/strong→Good, lack/harsh→Poor）：零成本但脆弱。
- 步骤 3：与直接预测做加权融合。

当前 GPU 时间紧张，先观察 1+3 效果后再决定是否上 idea 5（推荐方案 B）。

#### 运行方式

服务器上清掉旧 part 文件后重跑：

```bash
cd /root/autodl-tmp/CVPR/qwen-vl-finetune
rm -f track_1_test_res.json track_1_test_res.json.part*.json
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_multi.py
python convert_json_test.py --input track_1_test_res.json --output track_1_test_res_final.json
```

启动时观察日志确认拟合成功：
- `✅ fitted on N train items`
- 13 个权重和 intercept 打印值合理（权重应大致在 0.5-2.0，intercept 在 -10 到 +10 量级）。

### 方向调整：改走 GEPA 多候选 + 自修正（同日，待重跑）

#### 背景

放弃前面"隐藏浮点分 + 训练集拟合权重 + RankIQA quantile mapping"这条线，重写 `evaluation/evaluation_multi.py`。原因是该路线对排序代理仍依赖 criteria 离散等级和后处理校准，且训练集拟合权重对真实测试分布的迁移效果不确定；改为在推理阶段直接通过多候选采样和自洽性筛选，让模型自身输出更稳定的 `total_score` 和 criteria。

新方案核心思路：**对每张图采 K 个候选，按"输出自洽性"打分选最优，低分候选触发二次修正**。代码主入口仍是 `evaluation/evaluation_multi.py`，CLI 参数化（argparse）控制模式开关。

#### 改动 1：GEPA 多候选推理 + 聚合选优

- 新增 `INFERENCE_MODE = "gepa" | "baseline"` 开关。
- GEPA 模式下，对每张图调用 `infer_one_gepa()` 生成 `K=5` 个候选：温度序列 `[0.2, 0.3, 0.4, 0.5, 0.7]` 递增覆盖。
- 用 `aggregate_candidates_gepa()` 计算每个候选的质量分（见改动 2），按质量分降序选最优候选作为最终输出。
- 成本：单图推理次数 1 → 5（最坏含 self-refine 6 次）。GPU 时间预算需要重新评估。

#### 改动 2：三维质量评分体系

新增 `score_total_quality()` 综合 3 个维度（0-100）：

1. `score_internal_consistency` (权重 50%)
   - 由 criteria 分布推导期望分：`Good=80, Medium=50, Poor=20` 加权平均。
   - 与模型实际 `total_score` 偏差越小，分数越高（阈值 5/10/15/20/30 阶梯）。
   - 目的：识别"total_score 与 criteria 矛盾"的候选并淘汰。

2. `score_criteria_quality` (权重 30%)
   - 完整性 40 分：13 维度齐全。
   - 格式正确 40 分：value 在 `{Good, Medium, Poor}`。
   - 懒政惩罚 -20 分：13 维度全输出同一个 level。

3. `score_answer_validity` (权重 20%)
   - answer ∈ {A,B,C,D} → 100，否则 0。

#### 改动 3：Self-Refinement 二次修正

- 当最优候选的 `consistency < CONSISTENCY_THRESHOLD (=30)` 时，触发 `self_refine()`。
- 把第一次的原始输出和问题诊断喂回模型（纯文本，不带图），让其重新输出。
- 修正温度 0.4，比初次推理略高，鼓励调整。
- 仅当修正后质量分高于原候选才替换。

#### 改动 4：鲁棒 JSON 解析（`extract_json_robust`）

四级兜底：
1. 标准 `extract_json` 解析。
2. 清除 markdown 代码块后再解析。
3. 正则匹配所有 `{...}` 块逐一尝试。
4. 修复单引号、尾部逗号等常见错误后再解析。

配合 `validate_and_fix_parsed()`：缺失维度补 `Medium`，越界 `total_score` 截断到 1-100，非法 answer 默认 `A`。彻底消除上游 `NOT_RES` 来源。

#### 改动 5：CoT prompt（可选）

- 新增 `ENABLE_COT = True` 开关。
- CoT 版 prompt 让模型先逐条分析 13 个维度（"先思考再下结论"），再输出 FINAL OUTPUT JSON。
- 同时保留 `--no-cot` 走原来精简 prompt 的路径。
- 注意：CoT 输出更长，`MAX_NEW_TOKENS` 保持 768。
- 两套 prompt **都已移除** `criteria_text`（即不再向模型注入测试 JSON 里的 criteria prior），与 2026-05-07 改动保持一致。

#### 改动 6：LoRA 自动加载

- `DemoServer.__init__` 检测 `MODEL_NAME` 目录下有无 `adapter_config.json`：
  - 有：先加载 base model（PROCESSOR_NAME），再 `PeftModel.from_pretrained()` 加 LoRA adapter，最后 `merge_and_unload()` 合并。
  - 无：按普通完整 checkpoint 加载。
- 当前 `MODEL_NAME = "./models/checkpoint"`，对应训练产出的 checkpoint。

#### 改动 7：工程化细节

- `argparse` 接管所有运行参数（`--mode/--cot/--no-cot/--self_refine/--num_candidates/--input_json/...`），不必改源码切换实验。
- 文件路径预建索引：扫描 `IMAGES_PATH` 下子目录构建 `filename → full_path` 字典，避免每张图都 `os.path.exists` 探测多个子目录。
- Worker 日志加详细统计：`skipped_missing / skipped_parse / errors`，前 3 个错误打印完整 traceback 和 raw_preview，便于定位。
- 增加 `PeftModel` import；新增 `re` import 给 robust parser 用。

#### 已抛弃 / 不再保留

- ❌ `fit_params_from_train()` 训练集线性回归拟合权重
- ❌ `CRITERION_WEIGHTS / LEVEL_TO_SCORE` 手工 + 自适应版本
- ❌ `compute_rank_proxy() / apply_rankiqa_scores()` RankIQA 后处理
- ❌ 隐藏浮点分 prompt（要求 `{"score": float, "level": str}` 双字段）
- 即不再做"基于 criteria 重新估算 total_score"的后处理，模型直接输出的 `total_score` 经多候选筛选后直接使用。

#### 论文依据

1. **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (arXiv:2507.19457)
   核心：通过多候选生成 + 反思评估 + 选优，能在不更新模型参数的情况下显著提升任务表现。本次实现是其推理时变体——多温度采样 + 自洽性评分 + 低分修正。

2. **Self-Refine: Iterative Refinement with Self-Feedback** (Madaan et al., NeurIPS 2023, arXiv:2303.17651)
   核心：让模型用同一个 LLM 对自己输出做诊断 + 修正。本次 `self_refine()` 是其简化版（单轮修正），并且 feedback 由结构化质量评分给出而非自由文本。

3. **Self-Consistency Improves Chain of Thought Reasoning** (Wang et al., ICLR 2023, arXiv:2203.11171)
   核心：CoT + 多采样 + 多数投票优于单次 CoT。本次没做真正的"投票"，但用 `score_internal_consistency` 充当类似的一致性度量来在 K 个候选中筛选。

#### 理想效果

1. **SRCC / PLCC 提升**：多候选 + 一致性筛选淘汰了"total_score 和 criteria 矛盾"的输出，total_score 信号更稳定；同时 criteria 自洽时排序代理隐含更可靠。
2. **Criteria Acc 略升**：CoT 让模型先分析再下结论，应减少"全 Good 偷懒"的情况；懒政惩罚（-20）显式压制 13 维度同值的输出。
3. **NOT_RES = 0**：`validate_and_fix_parsed` 强制补全和合法化。
4. **Answer Acc 持平**：answer 逻辑未改，但 CoT 可能轻微帮助 answer 与图像对齐。

#### 风险

1. **GPU 时间 5-6 倍**：K=5 候选 + 可能 self-refine。GPU 紧张时需要把 `--num_candidates` 调低（如 3）或关 self-refine 测试。
2. **质量评分体系是启发式的**：`score_internal_consistency` 假设 "Good=80/Medium=50/Poor=20 加权平均 = total_score"，与官方真实 total_score 计算方式可能不一致。若官方计算更接近其他权重组合，该评分会把"看似自洽实则错的"候选排到前面。
3. **CoT 在 4B 模型上可能反噬**：小模型的链式推理常常被自己绕进去，输出更长但质量未必更高。可对照跑 `--no-cot` 看差异。
4. **Self-Refinement 阈值 30 偏低**：当前只在严重不一致时触发，覆盖面小；后续可调高（如 50）看是否更激进地修正能涨分。
5. **没有 ground truth 校准**：所有筛选信号都是模型自检，可能"自洽地错"。需配合训练集 sanity check（跑一遍 train 集，看选优后的 SRCC 是否真高于单次采样）。

#### 待验证

1. K=3 vs K=5 vs K=7 的成本/收益曲线。
2. CoT vs no-CoT 的实际增益。
3. Self-Refinement 触发率（worker 日志里需要打印 refine 命中数，当前缺这条统计）。
4. 训练集上的"GEPA 选优后 SRCC" vs "单次采样 SRCC"，作为线下指标证伪。

#### 运行方式

```bash
cd /root/autodl-tmp/CVPR/qwen-vl-finetune
rm -f track_1_test_res.json track_1_test_res.json.part*.json
# 默认 GEPA + CoT + Self-Refine
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_multi.py
# 想关 CoT 或 self-refine：
# python evaluation/evaluation_multi.py --no-cot --no-self_refine
# 想跑基线对照：
# python evaluation/evaluation_multi.py --mode baseline --no-cot
python convert_json_test.py --input track_1_test_res.json --output track_1_test_res_final.json
```

提交前 sanity check：
- `items == 2000`
- `NOT_RES == 0`（应天然为 0）
- A/B/C 分布、`total_score` 直方图、unique 数与上次对比
- worker 日志：`skipped_parse / errors` 应远低于上次（robust parser + validate 兜底）

## 2026-05-13

### 实际提交结果

```text
Participant: borui_yao
ID: 726880
SRCC: 0.83
PLCC: 0.83
Criteria Acc: 0.51
Answer Acc: 0.89
```

对比：

```text
2026-05-07: 0.65 / 0.65 / 0.40 / 0.79
2026-05-12: 0.65 / 0.66 / 0.39 / 0.78
2026-05-13: 0.83 / 0.83 / 0.51 / 0.89
```

### 当前判断

1. 这次提升不是单一 prompt tweak，主要是多因素叠加：
   - `./models/checkpoint` 已确认是 LoRA adapter，不是完整模型目录。
   - 当前推理链路采用 `GEPA + CoT + Self-Refinement + robust parser`。
   - `total_score` 不再依赖 RankIQA-style 后处理，而是依赖多候选筛选后的模型直接输出。
2. LoRA checkpoint 很可能是本次大幅提分的最大变量。
3. 推理侧增强也明显有效，否则四项指标不会同时上涨。

### 本地结果分布

当前 `track_1_test_res.json` 统计：

```text
items: 2000
score_unique: 19
score_range: 32-82
score_mean: 59.956
criteria: Good=17841, Medium=4549, Poor=3610
answers: B=526, A=506, C=491, D=477
13 Good: 395
12 Good: 348
11 Good: 234
```

结论：

1. 输出格式已经稳定，非法 answer / 缺失 criteria 基本被清除。
2. `Poor` 已经明显回来了，criteria 不再像早期版本那样一边倒偏 `Good`。
3. 高分段仍然偏拥挤，`11/12/13 Good` 合计仍然较高，后续还可以继续优化总分区分度。

### 为什么会提分

1. **LoRA adapter**
   - `adapter_config.json` 已确认存在，说明当前跑的是 base model + LoRA merge。
   - 这通常直接提升 criteria 理解和 answer 选择能力。

2. **GEPA 多候选推理**
   - 每张图生成 5 个候选，温度递增覆盖。
   - 候选中更差的输出会被后续质量评分淘汰。

3. **CoT prompt**
   - 先逐维分析 13 个 criteria，再输出 JSON。
   - 对 `Criteria Acc` 和 `Answer Acc` 的帮助最直接。

4. **一致性筛选**
   - `score_internal_consistency` 会检查 `criteria` 分布和 `total_score` 是否匹配。
   - 这一步对 `SRCC/PLCC` 提升贡献很大。

5. **Self-Refinement + robust parser**
   - 低一致性候选会二次修正。
   - 坏 JSON、缺字段、非法值会被兜底修复。

### 今日脚本升级

本次在 `evaluation_multi.py` 上新增两个推理增强点：

1. **多候选共识分**
   - 新增 `score_candidate_consensus()`。
   - 对每个候选计算：
     - criteria 是否与其他候选多数一致
     - answer 是否与其他候选多数一致
     - total_score 是否接近候选集中心
   - `aggregate_candidates_gepa()` 不再只按 `quality_score` 选优，而是按：

```text
rank_score = quality_score * 0.75 + consensus_score * 0.25
```

2. **GEPA 运行统计**
   - 在 `worker_run()` 中新增：
     - `avg_valid_candidates`
     - `refine_triggered`
     - `refine_applied`
     - `avg_best_quality`
     - `avg_best_consensus`
     - `avg_best_rank`
   - 用于判断当前 0.83 的来源究竟更偏向模型本身，还是多候选筛选/修正机制。

3. **Self-Refine 替换条件修正**
   - 旧逻辑只按 `quality_score` 判断修正后是否替换。
   - 现逻辑改成按 `rank_score` 判断，和主聚合逻辑保持一致，避免修正后自洽但与候选共识更差的结果被错误替换。

### 这次升级的理想效果

1. 在不改模型权重的前提下，让 GEPA 真正利用“跨候选共识”，而不是只看单候选自评。
2. 进一步降低偶然高分、但和其余候选明显冲突的 bad candidate 被选中的概率。
3. 为下一步优化提供可观测统计，而不是继续盲调 prompt。

### 下一步

1. 先用升级后的脚本再跑一轮，对比：
   - `SRCC/PLCC` 是否继续涨
   - `refine_triggered / refine_applied` 是否足够高
   - `avg_best_consensus` 是否和高分提交正相关
2. 如果分数继续涨，说明“共识分”是有效方向。
3. 如果分数不涨但统计显示候选有效率已经很高，下一步优先优化高分段拥挤问题，而不是继续加候选数。

## 2026-05-14

### 当前推理修改状态

今天的 `evaluation_multi.py` 仍处在待全量重跑阶段，`res/v3` 只是 2026-05-13 的旧结果，不用于评估今天的新推理。

当前新推理主线：

```text
GEPA 多温度候选 -> quality_score 排序 -> 温度稳定性判断 -> Top-3 融合 / fallback
```

具体逻辑：

1. 每张图仍生成 5 个候选，温度为 `[0.2, 0.3, 0.4, 0.5, 0.7]`。
2. `score_temperature_stability()` 评估同一张图在不同温度下的输出稳定性：
   - `total_score` 方差
   - 13 个 criteria 的多数派一致性
   - answer 多数派一致性
3. `worker_run()` 根据 `stability` 和 `consistency` 走 2x2 决策：
   - `stability < 45` 且 `consistency < 30`：回退到 `temperature=0.2` 的保守候选。
   - `stability < 45` 且 `consistency >= 30`：只使用 best candidate，不做融合。
   - 其他情况：使用 `fuse_candidates()` 做 Top-3 加权融合。
4. `fuse_candidates()` 的输出方式：
   - `total_score`：Top-3 按 `quality_score` 加权平均。
   - criteria：Top-3 按维度加权投票。
   - answer：所有有效候选按 `quality_score` 加权投票。

### 本次新增补丁：GEPA_BRANCH_STATS

本次只新增运行统计，不改变最终提交 JSON 的字段格式，也不改变候选融合决策。

新增统计项：

```text
GEPA_BRANCH_STATS:
items
avg_valid_candidates
fusion
fallback_t02
best_only
refine_triggered
refine_applied
avg_stability
avg_consistency
```

### 为什么加这个补丁

当前新推理的关键不只是模型输出，而是 2x2 决策矩阵实际如何分流。没有统计时，全量跑完只能看到榜单结果，无法判断涨分或掉分来自哪里。

这些统计用于判断：

1. `fusion` 占比高且榜单提升：说明 Top-3 融合路线有效。
2. `fallback_t02` 占比高且榜单下降：说明稳定性阈值可能过严，或多温度候选波动过大。
3. `best_only` 占比高：说明融合被频繁禁用，当前策略偏保守。
4. `avg_stability` 低：说明温度序列过散，候选间分歧过大。
5. `avg_consistency` 低：说明 `total_score` 和 criteria 分布仍不够自洽。
6. `refine_triggered / refine_applied` 可判断 Self-Refinement 是否真正参与修正。

### 理想效果

这次补丁本身不应直接提高 `SRCC/PLCC/Criteria Acc/Answer Acc`，因为它只增加日志观测。

理想效果是：

1. 全量跑完后能解释新推理的行为分布。
2. 为下一步调参提供依据：
   - 是否调整 `stability < 45` 阈值。
   - 是否保留 Top-3 fusion。
   - 是否恢复跨候选共识分。
   - 是否把 P1 跨维度相关性评分接入 `quality_score`。

### 当前风险

1. `score_cross_dimension_consistency()` 已定义但尚未接入候选打分，因此 P1 当前不影响结果。
2. 2026-05-13 版本里的 `score_candidate_consensus()` / `rank_score` 已被当前新推理替换。如果新推理掉分，需要优先比较“共识选优”与“Top-3 融合”两条路线。
3. `res/v3` 是旧结果，不能用于证明 2026-05-14 新推理有效。

### 下一步

全量重跑后先记录每个 worker 的 `GEPA_BRANCH_STATS`，再对比榜单四项指标：

```text
SRCC
PLCC
Criteria Acc
Answer Acc
```

如果 `fusion` 占多数且分数提升，继续优化融合权重；如果 `fallback_t02` 或 `best_only` 占比异常高，先调稳定性阈值和温度序列，不急着改 prompt。

## 2026-05-16

### 实际提交结果

```text
Participant: borui_yao
ID: 733603
SRCC: 0.84
PLCC: 0.84
Criteria Acc: 0.51
Answer Acc: 0.90
```

对比：

```text
2026-05-13: 0.83 / 0.83 / 0.51 / 0.89
2026-05-16: 0.84 / 0.84 / 0.51 / 0.90
2026-05-16 23:07: 0.86 / 0.85 / 0.50 / 0.89
```

### 当前判断

1. 当前推理链路已经进入高分段，`SRCC/PLCC` 从 0.83 到 0.84，说明 Top-3 融合与稳定性分流方向有效。
2. `Answer Acc` 已达到 0.90，是当前最稳定的指标，不应优先改 answer prompt 或 answer 决策方式。
3. `Criteria Acc` 仍停在 0.51，说明 13 个 criteria 的 A/B/C 分类仍是主要瓶颈。
4. 当前继续大改 prompt 或温度序列的风险较高，容易破坏已经稳定的 `SRCC/PLCC` 和 `Answer Acc`。

### 23:07 新提交结果与判断

```text
Participant: borui_yao
ID: 734758
SRCC: 0.86
PLCC: 0.85
Criteria Acc: 0.50
Answer Acc: 0.89
Equal Avg: 0.7750
```

对比 10:01 的 `733603`：

```text
733603 Equal Avg: 0.7725
734758 Equal Avg: 0.7750
delta: +0.0025
```

结论：

1. 新融合明显提升了 `SRCC/PLCC`，说明 `total_score` 稳健融合有效。
2. `Criteria Acc` 和 `Answer Acc` 各下降 0.01，说明 criteria/answer 的多候选融合略激进。
3. 下一步不应回退 `total_score`，而应收紧 criteria/answer 融合，减少候选噪声对后两项的干扰。

### 稳定涨分策略：Metric-Decoupled Conservative Fusion

目标：在不改模型权重、不大改 prompt、不增加推理成本的前提下，对候选融合做细粒度调整。不同指标使用不同融合策略，而不是把 `total_score`、criteria、answer 完全绑定在同一个候选上。

#### 1. Answer 保持当前策略

保持当前所有有效候选按 `quality_score` 加权投票的方式。

原因：

```text
Answer Acc = 0.90
```

answer 已经是强项，继续改 prompt 或引入 P1 规则可能带来无收益扰动。

#### 2. total_score 保持 Top-3 加权平均，但增加轻量 outlier clamp

当前 `total_score` 已经比较稳定，不能回到强后处理或训练集分布映射。

建议只增加轻量约束：

```text
expected_score = mean(criteria_score)
criteria_score: Good=80, Medium=50, Poor=20

if abs(fused_total_score - expected_score) > 12:
    final_total_score = round(0.75 * fused_total_score + 0.25 * expected_score)
else:
    final_total_score = fused_total_score
```

作用：

1. 保留模型直接输出的排序能力。
2. 只修正明显偏离 criteria 的 outlier。
3. 理论上主要稳定 `SRCC/PLCC`，避免少量异常样本拖分。

#### 3. Criteria 改成逐维置信度融合

当前 `Criteria Acc = 0.51`，还有提升空间。建议不再简单用 Top-3 每维投票，而是对每个 criterion 单独判断投票置信度：

```text
对每个 criterion：
    使用所有有效候选做 quality_score 加权投票
    如果第一名和第二名差距明显：
        用加权投票结果
    如果差距很小：
        回退到 best candidate 在该维度的判断
```

原因：

1. 多候选多数投票适合高置信维度。
2. 低置信维度上，简单投票容易被随机候选拉偏。
3. best candidate 通常更自洽，作为低置信 fallback 更稳。

这项最有希望提升 `Criteria Acc`，同时不会明显影响 answer。

#### 4. P1 跨维度一致性只做低置信 tie-break，不做强惩罚

当前 `score_cross_dimension_consistency()` 已定义但未接入。建议不要直接用它大幅改变 `quality_score`，因为手写相关性规则可能误伤复杂图像。

更稳的接入方式：

```text
只在某个 criterion 投票差距很小的时候使用 P1。
如果某个标签会造成 Good/Poor 强冲突，优先选择 Medium 或次高票标签。
```

这样 P1 只处理模型自己也不确定的 case，不会大面积改动稳定样本。

### 理想效果

预期影响：

```text
SRCC / PLCC: 0.84 -> 0.845~0.86
Criteria Acc: 0.51 -> 0.52~0.54
Answer Acc: 0.90 -> 基本保持
```

该策略的核心不是追求大幅跃迁，而是在当前高分段继续压低候选融合噪声，争取稳定小涨。

### 为什么这个策略比继续改 prompt 稳

1. Prompt 改动会同时影响 criteria、answer、total_score，风险不可控。
2. 当前 answer 已经 0.90，不适合用 prompt 大改去冒险。
3. 融合层改动只影响最终选择逻辑，不改变模型生成分布。
4. 逐维 criteria 融合能针对当前瓶颈 `Criteria Acc=0.51`，比全局改推理更精确。

### 下一步执行建议

优先实现：

```text
1. criteria per-dimension confidence fusion
2. total_score lightweight outlier clamp
3. P1 only as low-confidence tie-break
```

暂时不要做：

```text
1. 大改 prompt
2. 提高候选数量
3. 改 answer 决策
4. 强行恢复训练集 quantile mapping
```

### 本次已实现：四指标平衡融合

用户明确目标不是只冲前两个指标，因此本次实现调整为四指标平衡融合：在保留 answer 稳定性的前提下，同时优化 `total_score` 和 criteria。

实现位置：

```text
evaluation_multi.py
- get_candidate_level()
- get_weighted_level_votes()
- fuse_criteria_by_confidence()
- estimate_score_from_criteria()
- weighted_median_score()
- fuse_total_score()
- fuse_candidates() 中接入 fuse_total_score()
```

#### A. Criteria 逐维置信度融合

不再简单使用 Top-3 每维投票，而是对每个 criterion 使用所有有效候选做 `quality_score` 加权投票。

具体策略：

```text
if top1_vote - top2_vote 足够大:
    使用加权投票 top1
else:
    进入低置信分支
```

低置信分支：

```text
候选标签集合 = 有票标签 + best candidate 标签
使用 P1 跨维度一致性作为 tie-break
再看 vote_share
最后看是否为 best candidate 标签
```

作用：

1. 高置信维度利用多候选共识，提高 criteria 稳定性。
2. 低置信维度避免被随机候选拉偏。
3. P1 只在低置信场景参与，不做全局强惩罚，降低误伤风险。

预期主要影响：

```text
Criteria Acc: 0.51 -> 0.52~0.54
```

#### B. total_score 稳健融合

具体策略：

1. 保留 Top-3 按 `quality_score` 加权平均，避免破坏当前已经有效的 PLCC 线性信号。
2. 当 Top-3 的 `total_score` 分歧较大时，引入加权中位数：

```text
if score_spread >= 18:
    fused_score = 0.65 * weighted_mean + 0.35 * weighted_median
else:
    fused_score = weighted_mean
```

3. 使用 fused criteria 估计弱约束分：

```text
Good = 80
Medium = 50
Poor = 20
expected_score = mean(criteria_score)
```

4. 只在总分明显偏离 criteria 隐含质量时做轻量拉回：

```text
if abs(fused_score - expected_score) > 12:
    fused_score = 0.75 * fused_score + 0.25 * expected_score
```

理论作用：

1. `SRCC`：减少少量高温候选导致的离群分，降低排序反转。
2. `PLCC`：保留加权平均的连续分数信号，同时削弱异常点对线性相关的破坏。
3. `Criteria Acc`：通过 fused criteria 与 total_score 弱约束形成更一致的输出。
4. `Answer Acc`：不影响 answer 决策。

风险：

1. 如果模型原始 `total_score` 已经非常准，过强的 criteria 拉回会损失排序细节。
2. 因此本次只在 `abs(diff) > 12` 时触发，且只用 25% 权重拉回。
3. Criteria 低置信 tie-break 使用的是手工 P1 规则，因此只允许在低置信场景生效，避免大范围影响。

验证方式：

全量跑完后重点比较：

```text
SRCC / PLCC 是否高于 0.84
Criteria Acc 是否高于 0.51
Answer Acc 是否保持 0.90 左右
score_unique 是否保持充足
score range 是否未明显收缩
GEPA_BRANCH_STATS 中 fusion 占比是否仍为主路径
```

### 23:07 后续保守修正

针对 `734758` 的表现，已做保守修正：

```text
保留 total_score 稳健融合
criteria 改为 Top-3 置信投票 + 低置信回退 best candidate
answer 改为加权投票 + 低置信回退 best candidate
P1 不再参与最终 criteria 决策
```

原因：

1. `SRCC/PLCC` 已经涨到 `0.86/0.85`，说明总分融合有效，不能回退。
2. `Criteria/Answer` 下降更像是多候选投票把 best candidate 的正确判断拉偏。
3. 因此后续只收紧后两项融合逻辑，不改 prompt、不改模型、不改候选数。

新增阈值：

```text
CRITERIA_CONFIDENCE_MARGIN = 0.20
ANSWER_CONFIDENCE_MARGIN = 0.15
```

预期：

```text
SRCC/PLCC: 尽量保持 0.86/0.85
Criteria Acc: 0.50 -> 尝试回到 0.51
Answer Acc: 0.89 -> 尝试回到 0.90
```
