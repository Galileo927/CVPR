# Track 1 Inference Change Log

Date: 2026-05-07

## 修改内容

本次修改集中在 `qwen-vl-finetune/evaluation/evaluation_multi.py`，目标是不改模型权重，只优化 Track 1 推理和 `total_score` 后处理。

1. 推理数据接口改为服务器数据路径：
   - `DATASET_ROOT = "/root/autodl-tmp/PortraitCraft_dataset"`
   - `INPUT_JSON = os.path.join(DATASET_ROOT, "track_1_test.json")`
   - `TRAIN_JSON = os.path.join(DATASET_ROOT, "track_1_train.json")`
   - `IMAGES_PATH = DATASET_ROOT`

2. 增加图片路径自动解析：
   - 新增 `IMAGE_SUBDIRS = [""] + [f"images_{i:02d}" for i in range(11)]`
   - 新增 `resolve_image_path(image_name)`
   - 作用：当 `track_1_test.json` 里的 `image_path` 只有文件名时，自动在 `images_00` 到 `images_10` 中寻找图片。

3. 调整 prompt 中的 criteria 输入：
   - 原逻辑会把输入 JSON 里的 `level=A/B/C/x` 拼进 prompt。
   - 现逻辑只列出 13 个 criteria 名称，让模型基于图像重新判断。
   - 目的：降低复制模板值或旧伪标签的风险，尤其避免 `total_score` 继续贴近模板分。

4. 迁移 RankIQA 思路做 `total_score` 后处理：
   - 新增 `LEVEL_TO_SCORE`、`CRITERION_WEIGHTS`、`compute_rank_proxy()`。
   - 根据模型预测的 13 个 criteria 生成排序代理分。
   - 在 `merge_results()` 中调用 `apply_rankiqa_scores()`。
   - 若服务器存在 `track_1_train.json`，将测试集排序映射到训练集 `total_score` 分布。
   - 若训练集不可用，则退化为 `criteria proxy * 10` 的整数分。

5. 官方提交格式保持不变：
   - 推理阶段仍输出 `image_path / total_score / criteria / question / options / answer`。
   - 最终仍通过 `convert_json_test.py` 转成官方要求的 `{ "criteria": { name: { "level": "A/B/C" } } }` 格式。

## 来源论文依据

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

## 改后理想状态

1. `total_score` 不再集中在单一模板值，例如 1999 张都是 62。
2. 测试集总分会形成接近训练集的合理分布，图片之间有稳定排序。
3. SRCC 和 PLCC 应明显高于旧提交的 0.01。
4. Criteria Acc 和 Answer Acc 不应因总分后处理而明显下降。
5. 最终提交 JSON 仍完全符合官方格式，包含 2000 条结果。

## 当前实际状态

1. 本地已完成代码修改，`evaluation_multi.py` 语法检查通过。
2. 尚未在服务器 GPU 上重新跑完整 2000 张推理，因此没有新的排行榜实测分数。
3. 旧的新提交问题已定位：`track_1_test.json` 中 `total_score` 几乎全为 62，导致 SRCC/PLCC 约为 0.01。
4. 旧副本得分较高的主要原因是 `total_score` 有 45 到 87 的非恒定分布，因此排序指标更好。
5. 当前已确认服务器项目目录为 `/root/autodl-tmp/CVPR/qwen-vl-finetune`，脚本中的 `OUTPUT_JSON` 与该路径一致。

## 建议运行方式

在服务器上清理旧 part 文件后重新跑，避免脚本误认为已经完成：

```bash
cd /root/autodl-tmp/CVPR/qwen-vl-finetune
rm -f track_1_test_res.json track_1_test_res.json.part*.json
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_multi.py
python convert_json_test.py --input track_1_test_res.json --output track_1_test_res_final.json
```
