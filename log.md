# Track 1 Experiment Log

Last updated: 2026-05-09

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
