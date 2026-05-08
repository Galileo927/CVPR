# Track 1 Experiment Log

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
