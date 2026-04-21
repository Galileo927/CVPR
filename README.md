# PortraitCraft：用于人像构图理解与生成的基准

**论文：** 本仓库是论文 [PortraitCraft: A Benchmark for Portrait Composition Understanding and Generation](https://arxiv.org/abs/2604.03611)（arXiv:2604.03611）的**官方实现**。

---

本仓库支持面向竞赛场景的人像构图理解与生成功能，包含两个赛道；每个赛道的数据格式、训练脚本与评测脚本都在对应子目录中。

| 赛道 | 目录 |
|------|------|
| **赛道一：人像构图理解（Portrait Composition Understanding）** | `qwen-vl-finetune/` |
| **赛道二：人像构图生成（Portrait Composition Generation）** | `qwen-image-finetune/` |

请使用与你目标赛道对应的子目录进行数据准备、训练和评测。

---

## 数据下载

**PortraitCraft** 数据集发布在 Hugging Face。下载后，请在各赛道对应的配置和数据转换脚本中设置路径。

| 项目 | 链接 |
|------|------|
| PortraitCraft 数据集（两个赛道的 train/test） | [https://huggingface.co/datasets/zijielou/PortraitCraft](https://huggingface.co/datasets/zijielou/PortraitCraft) |

---

## 预训练模型

发布的 **PortraitCraft** 基础模型权重在 Hugging Face。下载后，将 `MODEL_PATH`、`model_name_or_path` 或 YAML 中对应字段指向本地模型目录（请按赛道选择对应变体）。

| 用途 | 链接 |
|------|------|
| PortraitCraft 预训练权重（赛道一与赛道二） | [https://huggingface.co/yytang225/PortraitCraft](https://huggingface.co/yytang225/PortraitCraft) |

基础训练时长参考：

- 赛道一：单张 L20x GPU，训练 0.5 epoch 约 2 小时。
- 赛道二：4 张 L20x GPU，训练 12000 step 约 14 小时。

### 环境依赖

可参考如下安装方式：

```bash
cd qwen-vl-ft/
pip install -r requirements.txt
```

## 赛道一：`Portrait Composition Understanding`（qwen-vl-finetune）

赛道一基于 Qwen-VL 进行构图理解微调。以下内容介绍 `qwenvl` 目录结构与 `qwen-vl-finetune` 的启动脚本。

### 工作流

1. 准备并转换数据集（见下方“自定义数据配置”）。
2. 在训练脚本中修改模型路径、数据路径与超参数。
3. 训练完成后进行推理，并将输出转换成官方提交 JSON 格式。

### 代码结构（`qwenvl`）

`qwenvl` 目录包含以下模块：

#### `train/`
- `trainer.py`：在 Hugging Face Trainer 基础上改造的训练器。
- `train_qwen.py`：训练主入口。
- `argument.py`：模型、数据、训练参数 dataclass 定义。

#### `data/`
- `__init__.py`：数据集配置注册。
- `data_processor.py`：QwenVL 数据处理逻辑。
- `rope2d.py`：RoPE 实现。

#### `tools`
- `process_bbox.ipynb`：将 bbox 转换到 QwenVL 训练格式；如有 grounding 数据可参考。
- `pack_data.py`：将样本按长度进行打包（packing）。

### 自定义数据配置

自定义数据需符合以下格式。

#### JSON 数据结构

**媒体字段规范：**
- `image`：媒体文件路径（必填）
- prompt 中媒体标记：
  - `<image>`：图像理解任务

#### 示例样本

1. **单图示例：**

```json
[
  {
    "image_path": "unsplash_people_00010_67dcd75a09e4.jpg",
    "criteria": {
      "Color Harmony": {
        "score": 7.0,
        "reason": "The vibrant colors of the mural and the subjects' clothing create a lively and energetic mood, though the extreme colorfulness makes the image slightly busy."
      },
      "Visual Style Consistency": {
        "score": 7.5,
        "reason": "The overall aesthetic—bright, casual, vibrant, and sunny—is consistently maintained throughout the image, fitting the intended lifestyle mood."
      },
      "Sharpness": {
        "score": 7.6,
        "reason": "The overall image is clear, with key details like facial features, hair, and clothing of the main subjects well-defined and sharp."
      },
      "Light and Shadow Modeling": {
        "score": 5.4,
        "reason": "The hard directional sunlight creates harsh shadows under chins and noses, which is not particularly flattering for portraiture and lacks refined tonal transitions."
      },
      "Creativity and Originality": {
        "score": 6.2,
        "reason": "The image feels like a very conventional lifestyle stock photograph, lacking a unique perspective or original visual language."
      },
      "Exposure Control": {
        "score": 6.3,
        "reason": "Exposure is generally reasonable, but the bright, direct sunlight causes some harsh highlights on the skin and clothing, with deep shadows that slightly reduce tonal layering."
      },
      "Application of Classical Composition Principles": {
        "score": 5.0,
        "reason": "The composition is a standard group lineup, but the framing feels somewhat arbitrary, particularly on the right side where elements are poorly cropped."
      },
      "Depth of Field and Layering": {
        "score": 5.2,
        "reason": "There is minimal depth of field; the brightly colored, busy background is almost entirely in focus, which fails to provide good separation for the subjects."
      },
      "Visual Center Stability": {
        "score": 5.1,
        "reason": "While the interacting women in the center draw the eye, the highly distracting background and the confusing elements on the right edge weaken the stability of the visual center."
      },
      "Visual Flow Guidance": {
        "score": 5.2,
        "reason": "The viewer's eye naturally follows the smiles and the glasses, but the flow is interrupted by the cluttered background and the awkward framing on the right."
      },
      "Structural Support Stability": {
        "score": 5.1,
        "reason": "The group forms a loose horizontal arrangement that grounds the image moderately well, but the unresolved right edge makes the structure feel unbalanced."
      },
      "Appropriateness of Negative Space": {
        "score": 4.2,
        "reason": "The frame feels very crowded with multiple subjects and a highly complex background, lacking sufficient negative space to let the image breathe."
      },
      "Subject Integrity": {
        "score": 4.0,
        "reason": "The woman on the far right is heavily cropped and turned away, and there is a disembodied arm holding a glass entering the frame awkwardly, harming the integrity of the subjects."
      }
    },
    "total_score": 53
  }
]
```

可直接参考 `demo/single_images.json` 等示例进行训练。

#### 训练数据集配置

如果要新增或修改训练数据，请按以下步骤：

##### 1. 数据转换

按 `qwen-vl-finetune/convert_json_train.py` 的格式转换（仅为示例，可按需定制训练内容）：

```python
cd qwen-vl-finetune/
python convert_json_train.py
```

##### 2. 在 `data/__init__.py` 中定义数据集字典

```python
DATASET_NAME = {
  "annotation_path": "/path/to/annotations.json",
  "data_path": "/path/to/image/data",  # 如果标注内是绝对路径可留空
}
```

##### 3. 注册到 `data_dict`

```python
data_dict = {
  "your_dataset_name": DATASET_NAME,
  # ... other datasets
}
```

##### 采样率控制

可在数据集名后追加 `%X` 指定采样比例：
- `"dataset_name%50"` 表示采样 50%
- `"dataset_name%20"` 表示采样 20%

##### 使用示例

1. 定义数据集：

```python
SINGLEIMAGES = {
  "annotation_path": "./demo/single_images_train_convert.json",
  "data_path": "./demo/images",
}

data_dict = {
  "my_dataset": MY_DATASET,
  "singleimages": SINGLEIMAGES,
}
```

2. 在训练中使用：

```python
dataset_names = ["my_dataset%50"]
configs = data_list(dataset_names)
```

##### 注意事项

- `annotation_path` 应指向 JSON 或 JSONL 标注文件。
- 如果标注中的媒体路径是绝对路径，`data_path` 可留空。
- 多数据集混训时，采样率按数据集分别生效。
- 可直接使用的一些开源数据：`nyu-visionx/Cambrian-10M`、`lmms-lab/LLaVA-NeXT-Data`、`FreedomIntelligence/ALLaVA-4V`、`TIGER-Lab/VisualWebInstruct`。
- 训练数据必须满足：
  - 每个问题中的一个 `<image>` 必须对应一个图像文件。
  - `<video>` 同理必须对应视频文件。
  - 这些特殊标记不能出现在答案文本中。
- 如开源数据存在丢图等问题，可用 `tools/check_image.py` 做完整性检查。

### 用法

训练模型示例：

```bash
#!/bin/bash
# 完整 QwenVL 训练启动脚本（带参数说明）

# ======================
# 分布式配置
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# 路径配置
# ======================
MODEL_PATH="/path/to/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="./checkpoints"
CACHE_DIR="./cache"

# ======================
# 模型配置
# ======================
DATASETS="your_dataset%100"

# ======================
# 训练参数
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 576*28*28 \
         --min_pixels 16*28*28 \
         --video_fps 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_pixels 1664*28*28 \
         --video_min_pixels 256*28*28 \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 3 \
         --lora_enable True \
         --lora_r 8 \
         --lora_alpha 16 \
         --lora_dropout 0.0 \
         --deepspeed zero3.json
```

脚本参数可以归纳为：

- 模块训练开关：`tune_mm_vision`、`tune_mm_mlp`、`tune_mm_llm`。
  - 若图像与视频混训，建议 `tune_mm_vision=False`。
- `data_flatten`：将 batch 内样本拼接为单序列。
- `data_packing`：需先用 `tools/pack_data.py` 预处理。
- 学习率建议范围：`1e-6` 到 `2e-7`。
- 分辨率对性能影响很大，需认真设置 `--max_pixels` 与 `--min_pixels`。
- 若训练 Qwen2.5-VL-32B，建议参考 `scripts/sft_32b.sh`，通常需要 8 张 80G GPU。
- 可在模型 `config.json` 增加 `"_attn_implementation": "flash_attention_2"` 启用 FlashAttention。
- Qwen3VL MoE 不支持 DeepSpeed ZeRO-3，且 HF 官方实现当前不含 load balancing loss。

**训练示例：**

```bash
cd qwen-vl-finetune
bash scripts/sft_qwen3_4b.sh
```

**评测示例：**

```bash
cd qwen-vl-finetune
python evaluation/evaluation_multi.py
```

**转换为最终提交格式：**

```bash
cd qwen-vl-finetune
python convert_json_test.py
```

---

## 赛道二：`Portrait Composition Generation`（qwen-image-finetune）

赛道二使用 Qwen-Image 进行构图生成（例如 LoRA）。与赛道一独立，工作目录在 `qwen-image-finetune/`。

### 工作流

1. 准备数据并转换到竞赛格式。
2. 在 `train_configs` 与启动命令中更新路径和超参数。
3. 训练后评测并导出结果。

### 示例

1. **数据转换（仅示例）**

```bash
cd qwen-image-finetune/
python convert_json_train.py
```

2. **训练**

```bash
cd qwen-image-finetune/
CUDA_VISIBLE_DEVICES=7 accelerate launch train.py --config ./train_configs/train_lora.yaml
```

3. **评测**

```bash
cd qwen-image-finetune/
python evaluation.py
```

---

## 参考代码

- **赛道一：** [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- **赛道二：** [FlyMyAI/flymyai-lora-trainer](https://github.com/FlyMyAI/flymyai-lora-trainer)

---

## 赛道一内容总结

赛道一的目标是让模型基于图像进行细粒度构图理解，输出结构化结果（各维度等级/理由、总分、问答结果）。从工程角度可概括为：

1. **任务本质**
- 不是生成图像，而是图像理解与结构化评估。
- 强调可解释性（多维度 criteria）和标准化输出（JSON）。

2. **模型训练核心**
- 基座是 Qwen-VL 系列多模态模型。
- 训练策略通过 `tune_mm_vision`、`tune_mm_mlp`、`tune_mm_llm` 控制。
- 融合相关层（projector/merger）可通过 `mm_projector_lr` 单独设学习率，是重要优化抓手。

3. **最小闭环**
- `convert_json_train.py` 转训练数据。
- 在 `data/__init__.py` 注册数据集。
- 用 `scripts/sft_*.sh` 启动训练。
- `evaluation/evaluation_multi.py` 推理。
- `convert_json_test.py` 转提交格式。

4. **高风险点**
- `<image>` 与样本媒体数量不匹配。
- 图像路径错误或坏图导致训练/推理中断。
- 推理 JSON 不稳定导致评测解析失败。
- 分辨率和 batch 设置不当导致 OOM。

5. **实操建议**
- 先跑通 baseline，再逐步调融合层和学习率。
- 先保证输出格式稳定，再追求更高指标。
- 每组实验固定配置、数据版本与随机种子，保证可复现。
