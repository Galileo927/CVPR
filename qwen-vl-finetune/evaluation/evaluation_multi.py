import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from multiprocessing import Process
import math
import glob
import multiprocessing as mp
from PIL import Image
import re
Image.MAX_IMAGE_PIXELS = None

mp.set_start_method("spawn", force=True)
WORKERS_PER_GPU = 1


PROCESSOR_NAME = "./models/Qwen3-VL-4B-Instruct"

MODEL_NAME = "./models/checkpoint"

INPUT_JSON = "/root/autodl-tmp/PortraitCraft_dataset/track_1_test.json"

OUTPUT_JSON = "/root/autodl-tmp/CVPR/qwen-vl-finetune/track_1_test_res.json"

IMAGES_PATH = "/root/autodl-tmp/PortraitCraft_dataset"

# ================== 推理增强配置 ==================
# 模式开关: "baseline" | "gepa"
INFERENCE_MODE = "gepa"

# GEPA 候选采样数
GEPA_NUM_CANDIDATES = 5

# GEPA 多候选温度序列 (递增，多样性)
GEPA_TEMPERATURES = [0.2, 0.3, 0.4, 0.5, 0.7]

# Phase 1 CoT 思考模式: True=先分析再输出 JSON, False=直接输出 JSON
ENABLE_COT = True

# Self-Refinement 开关: 对低一致性候选做二次修正
ENABLE_SELF_REFINE = True

# Phase 1 内部一致性阈值: 低于此分的候选触发 Self-Refinement
CONSISTENCY_THRESHOLD = 30

MAX_NEW_TOKENS = 768




def resize_keep_aspect(image_path, max_size=2048):
    img = Image.open(image_path)

    w, h = img.size

    # 如果已经符合要求，直接返回
    if max(w, h) <= max_size:
        return img

    # 计算缩放比例
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    return img



# ================== Model ==================
class DemoServer:
    def __init__(self, gpu_id, config=None):
        if config is None:
            config = {}
        self.config = config
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")

        model_name = config.get("model_name", MODEL_NAME)
        processor_name = config.get("processor_name", PROCESSOR_NAME)

        lora_config_file = os.path.join(model_name, "adapter_config.json")
        is_lora = os.path.isfile(lora_config_file)

        if is_lora:
            print(f"[worker-{gpu_id}] Loading base model: {processor_name}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                processor_name,
                torch_dtype=torch.float16
            )
            print(f"[worker-{gpu_id}] Loading LoRA adapter: {model_name}")
            self.model = PeftModel.from_pretrained(self.model, model_name)
            self.model = self.model.merge_and_unload()
            print(f"[worker-{gpu_id}] LoRA merged and unloaded")
        else:
            print(f"[worker-{gpu_id}] Loading model: {model_name}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )

        self.model = self.model.to(self.device)

        print(f"[worker-{gpu_id}] Loading processor: {processor_name}")
        self.processor = AutoProcessor.from_pretrained(processor_name)

        self.max_new_tokens = config.get("max_new_tokens", MAX_NEW_TOKENS)
        self.enable_cot = config.get("enable_cot", ENABLE_COT)
        self.enable_self_refine = config.get("enable_self_refine", ENABLE_SELF_REFINE)
        self.gepa_num_candidates = config.get("num_candidates", GEPA_NUM_CANDIDATES)
        self.gepa_temperatures = config.get("gepa_temperatures", GEPA_TEMPERATURES)
        self.consistency_threshold = config.get("consistency_threshold", CONSISTENCY_THRESHOLD)

        print(f"[worker-{gpu_id}] Model loaded on GPU {gpu_id}")


    def infer_one(self, image_path, prompt, temperature=None, top_p=None):
        """
        单次推理。temperature 和 top_p 可通过参数覆盖。

        用途：
        - baseline 模式: 使用默认参数 (temp=0.2, top_p=0.9)
        - GEPA 模式: 循环调用，每次使用不同的 temperature
        """
        img = resize_keep_aspect(image_path, 2048)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        temp = temperature if temperature is not None else 0.2
        tp = top_p if top_p is not None else 0.9

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=tp
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "assistant" in output:
            output = output.split("assistant")[-1].strip()

        return output

    def infer_one_gepa(self, image_path, prompt, num_candidates=None, use_cot=None):
        """
        Phase 2 核心: GEPA 多候选推理 + 聚合选优。

        流程:
        1. 生成 K 个候选（不同 temperature）
        2. 每个候选解析 JSON 并打分
        3. 聚合: 跨候选共识选最优
        4. 如果最优候选一致性低 → 触发 Self-Refinement

        返回: (best_raw_text, best_parsed_json, all_candidates_info)
        """
        if num_candidates is None:
            num_candidates = self.gepa_num_candidates
        if use_cot is None:
            use_cot = self.enable_cot

        # 候选温度序列（递增覆盖）
        temps = self.gepa_temperatures[:num_candidates]

        candidates = []
        for temp in temps:
            raw = self.infer_one(image_path, prompt, temperature=temp)
            parsed = extract_json_robust(raw)
            candidates.append({
                "raw": raw,
                "parsed": parsed,
                "temperature": temp
            })

        # 聚合: 质量评分 + 跨候选共识
        best, all_scored = aggregate_candidates_gepa(candidates, self.consistency_threshold)

        # Phase 3: Self-Refinement — 低一致性候选用二次推理修正
        if self.enable_self_refine and best.get("needs_refine", False):
            raw_refined = self.self_refine(best["raw"], best["score_detail"])
            parsed_refined = extract_json_robust(raw_refined)
            if parsed_refined is not None:
                refined_score, refined_detail = score_total_quality(parsed_refined)
                # 仅当修正后质量提升才替换
                if refined_score > best["quality_score"]:
                    best["raw"] = raw_refined
                    best["parsed"] = parsed_refined
                    best["quality_score"] = refined_score
                    best["score_detail"] = refined_detail
                    best["refined"] = True

        return best, all_scored

    def self_refine(self, raw_text, score_detail):
        """
        Phase 3: 自我修正（纯文本，无图片输入）。

        输入模型第一次的输出和质量问题，生成修正后的版本。
        使用更高的 temperature (0.4) 让模型有创造性地修正。

        注意：这里是纯文本推理，不需要图片，所以直接构造 messages。
        """
        refine_prompt = build_refine_prompt(raw_text, score_detail, self.consistency_threshold)

        messages = [{"role": "user", "content": [{"type": "text", "text": refine_prompt}]}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "assistant" in output:
            output = output.split("assistant")[-1].strip()

        return output


# ================== Prompt ==================
def build_prompt(item, use_cot=False):
    """
    构建推理 prompt。

    use_cot=True 时启用 Chain-of-Thought：
    - 先让模型逐维度分析
    - 再输出 FINAL OUTPUT JSON
    - 强制模型"先思考再下结论"，减少直觉性瞎猜
    """
    question = item["question"]
    options = item["options"]

    options_text = "\n".join([
        f"{k}. {v}"
        for k, v in options.items()
    ])

    if use_cot:
        prompt = f"""You are an expert visual aesthetics evaluator specializing in portrait photography.

Analyze the given portrait image systematically across 13 aesthetic dimensions, then provide your final structured assessment.

---
QUESTION:
{question}

---
OPTIONS:
{options_text}

---
ANALYSIS PHASE:
Please analyze the image for each dimension, noting your reasoning:

1. Color Harmony: Is the color palette balanced and visually pleasing?
2. Visual Style Consistency: Does the overall aesthetic feel coherent?
3. Sharpness: Are key elements (face, subject) in clear focus?
4. Light and Shadow Modeling: Is the lighting flattering and professionally executed?
5. Creativity and Originality: Does the composition show unique vision?
6. Exposure Control: Is the image properly exposed (not too bright/dark)?
7. Application of Classical Composition Principles: Does it follow rule of thirds, golden ratio, etc.?
8. Depth of Field and Layering: Is background appropriately blurred/separated?
9. Visual Center Stability: Is the main subject properly centered/weighted?
10. Visual Flow Guidance: Does the image guide the viewer's eye naturally?
11. Structural Support Stability: Does the overall composition feel balanced and grounded?
12. Appropriateness of Negative Space: Is there enough breathing room without excess emptiness?
13. Subject Integrity: Are all subjects fully captured without awkward cropping?

For each dimension, briefly state your observation then label it: Good / Medium / Poor.

---
COMPOSITE ASSESSMENT:
Based on your analysis above:
- Calculate an overall aesthetic score (1-100) that reflects the weighted balance of all dimensions
- Identify which composition principle the portrait best demonstrates (from the question options above)

---
FINAL OUTPUT (STRICT JSON ONLY):
Based on your complete analysis above, output the following JSON with NO additional text:

{{
  "total_score": "<integer 1-100 consistent with your analysis>",
  "criteria": {{
    "Color Harmony": "<Good|Medium|Poor based on your analysis>",
    "Visual Style Consistency": "<Good|Medium|Poor>",
    "Sharpness": "<Good|Medium|Poor>",
    "Light and Shadow Modeling": "<Good|Medium|Poor>",
    "Creativity and Originality": "<Good|Medium|Poor>",
    "Exposure Control": "<Good|Medium|Poor>",
    "Application of Classical Composition Principles": "<Good|Medium|Poor>",
    "Depth of Field and Layering": "<Good|Medium|Poor>",
    "Visual Center Stability": "<Good|Medium|Poor>",
    "Visual Flow Guidance": "<Good|Medium|Poor>",
    "Structural Support Stability": "<Good|Medium|Poor>",
    "Appropriateness of Negative Space": "<Good|Medium|Poor>",
    "Subject Integrity": "<Good|Medium|Poor>"
  }},
  "answer": "<A|B|C|D consistent with your analysis>"
}}

CRITICAL RULES:
- Each criterion MUST be judged independently based on your actual observations
- total_score is a NUMBER (e.g. 65), criteria values are "Good" or "Medium" or "Poor". They are different fields.
- total_score MUST be consistent with the criteria distribution (if most are Good, score should be ≥65)
- DO NOT output identical labels for all criteria
- You MUST output diverse ratings: use at least 2 different levels across the 13 criteria. A good image should have many "Good" ratings; a poor image should have many "Poor" ratings.
- answer MUST align with the overall aesthetic direction your analysis suggests
- Return ONLY valid JSON — no explanations, no observations, no markdown"""
    else:
        prompt = f"""You are an expert visual aesthetics evaluator.

You MUST analyze the image and output STRICT JSON ONLY.

---

TASKS:
1. Predict overall aesthetic score (1–100) based on visual evidence
2. Predict each criterion level: Good / Medium / Poor
3. Answer the multiple-choice question (A/B/C/D)

---

QUESTION:
{question}

---

OPTIONS:
{options_text}

---

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "total_score": "<integer 1-100 inferred from image>",
  "criteria": {{
    "Color Harmony": "<Good|Medium|Poor>",
    "Visual Style Consistency": "<Good|Medium|Poor>",
    "Sharpness": "<Good|Medium|Poor>",
    "Light and Shadow Modeling": "<Good|Medium|Poor>",
    "Creativity and Originality": "<Good|Medium|Poor>",
    "Exposure Control": "<Good|Medium|Poor>",
    "Application of Classical Composition Principles": "<Good|Medium|Poor>",
    "Depth of Field and Layering": "<Good|Medium|Poor>",
    "Visual Center Stability": "<Good|Medium|Poor>",
    "Visual Flow Guidance": "<Good|Medium|Poor>",
    "Structural Support Stability": "<Good|Medium|Poor>",
    "Appropriateness of Negative Space": "<Good|Medium|Poor>",
    "Subject Integrity": "<Good|Medium|Poor>"
  }},
  "answer": "<A|B|C|D>"
}}

---

IMPORTANT RULES:
- DO NOT copy any fixed value pattern.
- DO NOT output identical labels for all criteria.
- Each criterion MUST be judged independently from image evidence.
- total_score MUST NOT be a template value; it must reflect real visual quality.
- total_score is a NUMBER (e.g. 65), criteria values are "Good" or "Medium" or "Poor". They are different fields.
- answer MUST be grounded in visible evidence.
- If unsure, choose Medium instead of guessing Good/Poor blindly.
- You MUST output diverse ratings: use at least 2 different levels across the 13 criteria. A good image should have many "Good" ratings; a poor image should have many "Poor" ratings.

---

Return ONLY valid JSON. No explanation. No markdown."""

    return prompt


# ================== JSON parser ==================
VALID_LEVELS = {"Good", "Medium", "Poor"}
EXPECTED_DIMS = [
    "Color Harmony", "Visual Style Consistency", "Sharpness",
    "Light and Shadow Modeling", "Creativity and Originality",
    "Exposure Control", "Application of Classical Composition Principles",
    "Depth of Field and Layering", "Visual Center Stability",
    "Visual Flow Guidance", "Structural Support Stability",
    "Appropriateness of Negative Space", "Subject Integrity"
]


def validate_and_fix_parsed(parsed):
    if not parsed or not isinstance(parsed, dict):
        return parsed

    criteria = parsed.get("criteria")
    if isinstance(criteria, dict):
        fixed_criteria = {}
        for dim in EXPECTED_DIMS:
            val = criteria.get(dim)
            if isinstance(val, str) and val.strip() in VALID_LEVELS:
                fixed_criteria[dim] = val.strip()
            else:
                fixed_criteria[dim] = "Medium"
        parsed["criteria"] = fixed_criteria

    ts = parsed.get("total_score")
    try:
        ts = int(float(ts))
        parsed["total_score"] = max(1, min(100, ts))
    except (ValueError, TypeError):
        parsed["total_score"] = 50

    ans = str(parsed.get("answer", "")).strip().upper()
    if ans not in ("A", "B", "C", "D"):
        parsed["answer"] = "A"

    return parsed


def extract_json(text):
    try:
        text = text.replace("```json", "").replace("```", "")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        result = json.loads(text[start:end + 1])
        return validate_and_fix_parsed(result)
    except:
        return None


# ================== Phase 1: 推理质量评分函数 ==================
def score_internal_consistency(parsed):
    """
    Phase 1 核心：判断模型输出是否自洽（内在一致性评分）。

    逻辑：
    - 从 criteria 分布推导"期望总分"
    - 期望分与实际 total_score 的差距越小 → 一致性越高 → 分数越高

    分数范围: 0-100
    - ≥80: 高度自洽（模型认真分析过）
    - 50-79: 基本自洽（小偏差，可接受）
    - <50: 严重矛盾（可能是乱猜或模板输出）

    评分维度：
    1. total_score 与 criteria 分布的自洽性 (0-100)
    """
    if not parsed or not isinstance(parsed, dict):
        return 0.0

    criteria = parsed.get("criteria", {})
    if not isinstance(criteria, dict) or len(criteria) == 0:
        return 0.0

    # 1) 从 criteria 分布计算期望分
    level_scores = {"Good": 80, "Medium": 50, "Poor": 20}
    level_count = {"Good": 0, "Medium": 0, "Poor": 0}

    for v in criteria.values():
        v_str = str(v).strip()
        if v_str in level_scores:
            level_count[v_str] += 1

    total_levels = sum(level_count.values())
    if total_levels == 0:
        return 0.0

    # 加权期望分: Good×80 + Medium×50 + Poor×20 / 总维度数
    expected_score = sum(
        level_scores[k] * level_count[k]
        for k in level_scores
    ) / total_levels

    # 2) 获取实际 total_score
    try:
        actual = int(float(parsed.get("total_score", 0)))
    except (ValueError, TypeError):
        return 0.0

    # 3) 一致性偏差计算
    # 偏差越大，扣分越多
    deviation = abs(actual - expected_score)

    if deviation <= 5:
        consistency_score = 100.0
    elif deviation <= 10:
        consistency_score = 85.0
    elif deviation <= 15:
        consistency_score = 70.0
    elif deviation <= 20:
        consistency_score = 55.0
    elif deviation <= 30:
        consistency_score = 35.0
    else:
        consistency_score = max(0.0, 20.0 - (deviation - 30) * 0.5)

    return consistency_score


def score_criteria_quality(parsed):
    if not parsed or not isinstance(parsed, dict):
        return 0.0

    criteria = parsed.get("criteria", {})
    if not isinstance(criteria, dict):
        return 0.0

    score = 0.0

    expected_dims = set(EXPECTED_DIMS)

    # 1) 完整性 (0-40): 13 维度全有则满分
    present_dims = set(criteria.keys())
    missing = expected_dims - present_dims
    completeness = 40.0 * (1.0 - len(missing) / 13.0)
    score += completeness

    # 2) 格式正确性 (0-40): 每个有效值 +分
    valid_count = 0
    for v in criteria.values():
        v_str = str(v).strip()
        if v_str in VALID_LEVELS:
            valid_count += 1

    if len(criteria) > 0:
        format_score = 40.0 * valid_count / len(criteria)
        score += format_score

    # 3) 懒政检测 (0-20 惩罚): 如果所有维度值相同，说明模型在偷懒
    if len(criteria) >= 5:
        values = list(criteria.values())
        normalized = [str(v).strip().lower() for v in values]
        all_same = all(v == normalized[0] for v in normalized)
        if all_same:
            score -= 20.0

    return max(0.0, min(100.0, score))


def score_answer_validity(parsed):
    """
    评估 answer 字段的合法性。

    分数范围: 0-100
    - answer 在 {A,B,C,D} → 100
    - 其他 → 0
    """
    if not parsed:
        return 0.0
    ans = str(parsed.get("answer", "")).strip().upper()
    return 100.0 if ans in ('A', 'B', 'C', 'D') else 0.0


def score_total_quality(parsed):
    """
    综合评分：整合所有信号，给出一个 0-100 的质量分数。

    各维度权重：
    - 内部一致性: 50% (最重要)
    - criteria 质量: 30%
    - answer 合法性: 20%

    返回: (total_score, {
        "consistency": ...,
        "criteria_quality": ...,
        "answer_validity": ...,
        "final": ...
    })
    """
    if not parsed or not isinstance(parsed, dict):
        return 0.0, {
            "consistency": 0.0,
            "criteria_quality": 0.0,
            "answer_validity": 0.0,
            "final": 0.0
        }

    consistency = score_internal_consistency(parsed)
    criteria_quality = score_criteria_quality(parsed)
    answer_val = score_answer_validity(parsed)

    final = consistency * 0.5 + criteria_quality * 0.3 + answer_val * 0.2

    detail = {
        "consistency": consistency,
        "criteria_quality": criteria_quality,
        "answer_validity": answer_val,
        "final": final
    }

    return final, detail


# ================== P0: 温度稳定性评分 ==================
def score_temperature_stability(all_scored):
    """
    P0: 温度稳定性评分。
    原理：同一张图、不同温度采样下，模型判断越一致 → 越可信。
    
    返回 0~100。
    - ≥70: 高度稳定（模型很确定自己的判断）
    - 45~70: 中等波动（模型略有摇摆）
    - <45: 剧烈波动（模型自己也不确定）
    """
    valid = [c for c in all_scored if c.get("parsed") is not None]
    if len(valid) < 2:
        return 50.0

    # 1) total_score 稳定性：各候选总分标准差映射
    scores = []
    for c in valid:
        try:
            scores.append(int(float(c["parsed"].get("total_score", 0))))
        except:
            scores.append(50)
    if len(scores) >= 2:
        mean_s = sum(scores) / len(scores)
        score_std = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
        # std≤5→100, std≈25→50, std≥50→0
        score_stability = max(0, min(100, 100 - score_std * 2))
    else:
        score_stability = 50.0

    # 2) criteria 分布稳定性：13 维中与多数判断偏离的比例
    level_map = {"Good": 2, "Medium": 1, "Poor": 0}
    dim_conflict_count = 0
    total_dims = 0
    for dim in EXPECTED_DIMS:
        dim_levels = []
        for c in valid:
            val = str(c["parsed"].get("criteria", {}).get(dim, "")).strip()
            if val in level_map:
                dim_levels.append(level_map[val])
        if len(dim_levels) >= 4:  # 至少4个候选有值才评估
            total_dims += 1
            # 计算多数派占比：最高的 level 值出现次数
            max_count = max(dim_levels.count(l) for l in set(dim_levels))
            ratio = max_count / len(dim_levels)
            if ratio < 0.5:  # 多数派不到一半 → 严重分歧
                dim_conflict_count += 1
            elif ratio < 0.7:  # 70~100% 一致 → 轻微分歧
                dim_conflict_count += 0.3

    if total_dims > 0:
        criteria_stability = max(0, 100 - (dim_conflict_count / total_dims) * 100)
    else:
        criteria_stability = 50.0

    # 3) answer 稳定性：多数选项占比
    answers = []
    for c in valid:
        ans = str(c["parsed"].get("answer", "")).strip().upper()
        if ans in "ABCD":
            answers.append(ans)
    if answers:
        from collections import Counter
        top_count = Counter(answers).most_common(1)[0][1]
        answer_stability = (top_count / len(answers)) * 100
    else:
        answer_stability = 50.0

    return min(100, max(0,
        score_stability * 0.4 + criteria_stability * 0.4 + answer_stability * 0.2
    ))


# ================== P1: 跨维度相关性评分 ==================
AESTHETIC_CORRELATION_RULES = [
    # (维度A, 维度B, 矛盾惩罚值)
    # 这些维度在人像美学评估中存在自然正相关
    ("Sharpness", "Depth of Field and Layering", 8),
    ("Exposure Control", "Light and Shadow Modeling", 6),
    ("Visual Center Stability", "Visual Flow Guidance", 7),
    ("Structural Support Stability", "Application of Classical Composition Principles", 5),
    ("Appropriateness of Negative Space", "Visual Flow Guidance", 4),
]

CRITERIA_CONFIDENCE_MARGIN = 0.20
ANSWER_CONFIDENCE_MARGIN = 0.15


def score_cross_dimension_consistency(parsed):
    """
    P1: 跨维度相关性评分。
    原理：某些美学维度存在自然正相关。
    模型给出反常的 Good↔Poor 对立 → 扣分。
    
    返回 0~100。
    满分 = 无矛盾。只在模型自身判断彼此矛盾时扣分，
    图片本身质量好坏不影响此分。
    """
    if not parsed or not isinstance(parsed, dict):
        return 100.0
    criteria = parsed.get("criteria", {})
    if not isinstance(criteria, dict):
        return 100.0

    level_order = {"Good": 2, "Medium": 1, "Poor": 0}
    penalty = 0.0

    for dim_a, dim_b, max_p in AESTHETIC_CORRELATION_RULES:
        va = str(criteria.get(dim_a, "")).strip()
        vb = str(criteria.get(dim_b, "")).strip()
        if va not in level_order or vb not in level_order:
            continue
        diff = abs(level_order[va] - level_order[vb])
        if diff == 2:      # Good vs Poor → 严重反常
            penalty += max_p
        elif diff == 1:    # Good vs Medium 或 Medium vs Poor → 轻微警告
            penalty += max_p * 0.3

    return max(0, min(100, 100 - penalty))


def get_candidate_level(candidate, dim):
    if not candidate or not candidate.get("parsed"):
        return None
    val = str(candidate["parsed"].get("criteria", {}).get(dim, "")).strip()
    return val if val in VALID_LEVELS else None


def get_weighted_level_votes(candidates, dim):
    vote = {"Good": 0.0, "Medium": 0.0, "Poor": 0.0}
    for c in candidates:
        val = get_candidate_level(c, dim)
        if val in vote:
            vote[val] += max(0.0, float(c.get("quality_score", 0.0)))
    return vote


def fuse_criteria_by_confidence(valid_candidates):
    """
    Criteria-specific fusion.

    高置信维度使用 Top-3 候选的 quality_score 加权投票；
    低置信维度回退到 best candidate，降低多候选投票误伤。
    """
    if not valid_candidates:
        return {dim: "Medium" for dim in EXPECTED_DIMS}

    valid = sorted(valid_candidates, key=lambda x: x["quality_score"], reverse=True)
    best = valid[0]
    top_candidates = valid[:3]
    fused = {}

    for dim in EXPECTED_DIMS:
        vote = get_weighted_level_votes(top_candidates, dim)
        total = sum(vote.values())
        if total <= 0:
            best_val = get_candidate_level(best, dim)
            fused[dim] = best_val or "Medium"
            continue

        ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        top_label, top_weight = ranked[0]
        second_weight = ranked[1][1]
        margin = (top_weight - second_weight) / total

        if margin >= CRITERIA_CONFIDENCE_MARGIN:
            fused[dim] = top_label
        else:
            best_label = get_candidate_level(best, dim)
            fused[dim] = best_label or top_label

    return fused


def get_candidate_answer(candidate):
    if not candidate or not candidate.get("parsed"):
        return None
    ans = str(candidate["parsed"].get("answer", "")).strip().upper()
    return ans if ans in ("A", "B", "C", "D") else None


def fuse_answer_by_confidence(valid_candidates):
    """
    Answer fusion with low-confidence fallback.

    投票足够明确时使用所有候选加权投票；
    投票接近时回退到 best candidate，避免 0.90 强项被候选噪声拉低。
    """
    if not valid_candidates:
        return "A"

    valid = sorted(valid_candidates, key=lambda x: x["quality_score"], reverse=True)
    best_answer = get_candidate_answer(valid[0])
    answer_vote = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    for c in valid:
        ans = get_candidate_answer(c)
        if ans in answer_vote:
            answer_vote[ans] += max(0.0, float(c.get("quality_score", 0.0)))

    total = sum(answer_vote.values())
    if total <= 0:
        return best_answer or "A"

    ranked = sorted(answer_vote.items(), key=lambda x: x[1], reverse=True)
    top_answer, top_weight = ranked[0]
    second_weight = ranked[1][1]
    margin = (top_weight - second_weight) / total

    if margin >= ANSWER_CONFIDENCE_MARGIN:
        return top_answer

    return best_answer or top_answer


def estimate_score_from_criteria(criteria):
    """
    从 criteria 分布估计一个弱约束总分。
    只用于 total_score 离群修正，不作为主评分来源。
    """
    if not isinstance(criteria, dict):
        return 50.0

    level_scores = {"Good": 80.0, "Medium": 50.0, "Poor": 20.0}
    scores = []
    for dim in EXPECTED_DIMS:
        val = str(criteria.get(dim, "")).strip()
        if val in level_scores:
            scores.append(level_scores[val])

    if not scores:
        return 50.0

    return sum(scores) / len(scores)


def weighted_median_score(score_weight_pairs):
    """
    计算 total_score 的加权中位数。
    相比均值，能压低少量高温候选导致的极端分数。
    """
    valid_pairs = []
    for score, weight in score_weight_pairs:
        try:
            valid_pairs.append((int(float(score)), float(weight)))
        except (TypeError, ValueError):
            continue

    if not valid_pairs:
        return 50

    valid_pairs.sort(key=lambda x: x[0])
    total_weight = sum(max(0.0, w) for _, w in valid_pairs)
    if total_weight <= 0:
        return valid_pairs[len(valid_pairs) // 2][0]

    running = 0.0
    midpoint = total_weight / 2.0
    for score, weight in valid_pairs:
        running += max(0.0, weight)
        if running >= midpoint:
            return score

    return valid_pairs[-1][0]


def fuse_total_score(top_candidates, weights, fused_criteria):
    """
    Metric-decoupled total_score fusion.

    目标：
    - 保留 Top-3 加权平均的线性信号，利于 PLCC。
    - 用加权中位数压制离群候选，利于 SRCC。
    - 只在明显偏离 criteria 隐含质量时做轻量拉回。
    """
    score_weight_pairs = []
    weighted_sum = 0.0
    total_weight = 0.0

    for c, w in zip(top_candidates, weights):
        try:
            score = int(float(c["parsed"].get("total_score", 0)))
        except (TypeError, ValueError):
            score = 50
        weight = max(0.0, float(w))
        score_weight_pairs.append((score, weight))
        weighted_sum += score * weight
        total_weight += weight

    if total_weight <= 0:
        return 50

    mean_score = weighted_sum / total_weight
    median_score = weighted_median_score(score_weight_pairs)
    score_values = [score for score, _ in score_weight_pairs]
    spread = max(score_values) - min(score_values) if score_values else 0

    # 高分歧样本更信中位数，低分歧样本保留均值。
    if spread >= 18:
        fused_score = 0.65 * mean_score + 0.35 * median_score
    else:
        fused_score = mean_score

    expected_score = estimate_score_from_criteria(fused_criteria)
    if abs(fused_score - expected_score) > 12:
        fused_score = 0.75 * fused_score + 0.25 * expected_score

    return max(1, min(100, round(fused_score)))


# ================== Phase 2: 多候选聚合 ==================
def aggregate_candidates_gepa(candidates, consistency_threshold=None):
    if consistency_threshold is None:
        consistency_threshold = CONSISTENCY_THRESHOLD
    """
    GEPA 多候选聚合：对 K 个候选进行跨候选共识选择。

    策略：
    1. 计算每个候选的质量分数 (Phase 1 的 score_total_quality)
    2. 按质量分数降序排列
    3. 如果最高分候选的一致性分 < 阈值，触发 Self-Refinement

    输入: [{"raw": ..., "parsed": ..., "score_detail": {...}}, ...]
    输出: (best_candidate, all_with_scores)
    """
    if not candidates:
        return None, []

    scored = []
    for c in candidates:
        parsed = c.get("parsed")
        if parsed:
            score, detail = score_total_quality(parsed)
            scored.append({
                "raw": c.get("raw", ""),
                "parsed": parsed,
                "quality_score": score,
                "score_detail": detail,
                "temperature": c.get("temperature")
            })
        else:
            scored.append({
                "raw": c.get("raw", ""),
                "parsed": None,
                "quality_score": 0.0,
                "score_detail": {"consistency": 0.0, "criteria_quality": 0.0, "answer_validity": 0.0, "final": 0.0},
                "temperature": c.get("temperature")
            })

    # 按质量分降序
    scored.sort(key=lambda x: x["quality_score"], reverse=True)

    best = scored[0]

    # 低一致性 → Self-Refinement 候选标记
    if best["score_detail"]["consistency"] < consistency_threshold:
        best["needs_refine"] = True

    return best, scored


def fuse_candidates(all_scored):
    """
    P0+P1 融合输出：
    - answer: 所有候选 quality_score 加权投票
    - criteria: Top-3 quality_score 加权 per-dimension 投票
    - total_score: Top-3 quality_score 加权平均后四舍五入

    all_scored: [{"parsed": {...}, "quality_score": 91.0}, ...]
    Returns: {"total_score": 72, "criteria": {...}, "answer": "B"}
    """
    valid = [c for c in all_scored if c.get("parsed") is not None]
    if not valid:
        return None

    valid.sort(key=lambda x: x["quality_score"], reverse=True)
    top3 = valid[:3]
    weights = [c["quality_score"] for c in top3]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    # ── criteria: 所有有效候选逐维置信度融合 ──
    fused_criteria = fuse_criteria_by_confidence(valid)

    # ── total_score: Top-3 稳健融合 + criteria 轻量约束 ──
    fused_score = fuse_total_score(top3, weights, fused_criteria)

    # ── answer: 加权投票 + 低置信回退 best candidate ──
    fused_answer = fuse_answer_by_confidence(valid)

    return {
        "total_score": fused_score,
        "criteria": fused_criteria,
        "answer": fused_answer
    }


# ================== Phase 2+3: 鲁棒 JSON 解析 ==================
def extract_json_robust(text):
    """
    多策略 JSON 提取，比 extract_json 更强的兜底解析。

    策略 1: 标准提取 (复用 extract_json)
    策略 2: 清除 markdown 后提取
    策略 3: 正则匹配所有 {...} 块，逐个尝试解析
    策略 4: 修复常见 JSON 错误（单引号、尾部逗号）再解析
    """
    # 策略 1: 标准提取 (已内含 validate_and_fix_parsed)
    result = extract_json(text)
    if result is not None:
        return result

    # 策略 2: 清除 markdown 代码块
    cleaned = text.replace("```json", "").replace("```", "").strip()

    # 策略 3: 正则找所有 {...} 块，逐一尝试
    # 处理嵌套 {...}，找最外层的匹配
    matches = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', cleaned)
    for match in matches:
        try:
            result = json.loads(match)
            return validate_and_fix_parsed(result)
        except json.JSONDecodeError:
            continue

    # 策略 4: 修复常见错误后重试
    try:
        fixed = cleaned
        fixed = fixed.replace("'", '"')
        fixed = re.sub(r',\s*\}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        json_start = fixed.find('{')
        json_end = fixed.rfind('}')
        if json_start != -1 and json_end > json_start:
            result = json.loads(fixed[json_start:json_end + 1])
            return validate_and_fix_parsed(result)
    except (json.JSONDecodeError, Exception):
        pass

    return None


# ================== Phase 3: Self-Refinement ==================
def build_refine_prompt(raw_text, score_detail, consistency_threshold=None):
    if consistency_threshold is None:
        consistency_threshold = CONSISTENCY_THRESHOLD
    """
    为 Self-Refinement 阶段构建修正 prompt。

    告诉模型它的输出存在哪些问题，要求重新输出。
    """
    consistency = score_detail.get("consistency", 0)
    criteria_q = score_detail.get("criteria_quality", 0)
    answer_v = score_detail.get("answer_validity", 0)

    issues = []
    if consistency < consistency_threshold:
        issues.append(f"- total_score 与 criteria 分布不一致（一致性得分 {consistency:.0f}/100）")
    if criteria_q < 60:
        issues.append(f"- criteria 格式有问题（质量得分 {criteria_q:.0f}/100）")
    if answer_v == 0:
        issues.append("- answer 字段无效（不是 A/B/C/D）")

    issues_text = "\n".join(issues) if issues else "- 输出格式不稳定"

    prompt = f"""You previously generated the following aesthetic assessment:

{raw_text[:800]}

Quality check identified these issues:
{issues_text}

Please regenerate a corrected version following these rules:
1. If total_score is high, ensure most criteria are Good (not Poor/Medium)
2. If total_score is low, ensure most criteria are Poor/Medium (not Good)
3. All 13 criteria MUST be present with exactly one of: Good, Medium, or Poor
4. Answer must be exactly one letter: A, B, C, or D
5. Output ONLY valid JSON — no explanations

Corrected JSON:"""

    return prompt


# ================== 读取历史结果 ==================
def build_done_set_from_parts():

    files = glob.glob(OUTPUT_JSON + ".part*.json")

    done = set()

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for x in data:
                    done.add(x["image_path"])
        except:
            continue

    print(f"♻️ Found done samples: {len(done)}")

    return done


# ================== Worker ==================
def worker_run(worker_id, gpu_id, data_chunk, config):
    """
    推理 worker 入口。

    config 包含所有运行时参数：
    - mode, num_candidates, enable_cot, enable_self_refine
    - output_json, images_path, model_name, processor_name, max_new_tokens
    - gepa_temperatures, consistency_threshold
    """
    mode = config.get("mode", INFERENCE_MODE)
    num_candidates = config.get("num_candidates", GEPA_NUM_CANDIDATES)
    enable_cot = config.get("enable_cot", ENABLE_COT)
    images_path = config.get("images_path", IMAGES_PATH)
    output_json = config.get("output_json", OUTPUT_JSON)

    print(f"[worker-{worker_id}] PID={os.getpid()} GPU={gpu_id} mode={mode} "
          f"chunk_size={len(data_chunk)} cot={enable_cot} "
          f"images_path={images_path}")

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)

    try:
        server = DemoServer(gpu_id, config)
    except Exception as e:
        print(f"[worker-{worker_id}] FATAL: failed to load model on GPU {gpu_id}: {e}")
        return

    out_path = output_json + f".part{worker_id}.json"

    results = []
    total_skipped_missing = 0
    total_skipped_parse = 0
    total_errors = 0
    total_gepa_items = 0
    total_valid_candidates = 0
    total_fallback_t02 = 0
    total_best_only = 0
    total_fusion = 0
    total_refine_triggered = 0
    total_refine_applied = 0
    total_stability = 0.0
    total_consistency = 0.0

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[worker-{worker_id}] load existing: {len(results)} from {out_path}")
        except:
            results = []
            print(f"[worker-{worker_id}] failed to load existing, starting fresh")

    for idx, item in enumerate(tqdm(data_chunk, desc=f"worker-{worker_id}")):
        image_path = item.get("_full_path", os.path.join(images_path, item["image_path"]))

        if not os.path.exists(image_path):
            total_skipped_missing += 1
            if idx < 3:
                print(f"[worker-{worker_id}] MISSING: {image_path}")
            continue

        prompt = build_prompt(item, use_cot=enable_cot)

        try:
            if mode == "gepa":
                best, all_scored = server.infer_one_gepa(
                    image_path, prompt,
                    num_candidates=num_candidates,
                    use_cot=enable_cot
                )
                raw = best["raw"]

                # P0: 温度稳定性
                stability = score_temperature_stability(all_scored)
                # 现有自洽性（best 候选的 consistency，由 aggregate_candidates_gepa 计算）
                consistency = best.get("score_detail", {}).get("consistency", 0)

                total_gepa_items += 1
                total_valid_candidates += sum(1 for c in all_scored if c.get("parsed") is not None)
                total_refine_triggered += int(bool(best.get("needs_refine", False)))
                total_refine_applied += int(bool(best.get("refined", False)))
                total_stability += float(stability)
                total_consistency += float(consistency)

                # ── 2×2 决策矩阵 ──
                parsed = None
                if stability < 45 and consistency < 30:
                    # ❌ 低稳 + 低洽 → 所有候选不可靠 → 降级取 temp=0.2 候选（最保守）
                    total_fallback_t02 += 1
                    for c in all_scored:
                        if c.get("temperature") == 0.2:
                            parsed = c.get("parsed")
                            if parsed is None:
                                parsed = extract_json_robust(c.get("raw", ""))
                            break
                    if parsed is None:
                        parsed = best["parsed"]  # 兜底
                    if idx < 3:
                        print(f"[worker-{worker_id}] FALLBACK t=0.2: stability={stability:.0f} consistency={consistency:.0f}")

                elif stability < 45 and consistency >= 30:
                    # ⚠️ 低稳 + 高洽 → 候选模板化嫌疑 → 弃用融合，单取最优候选
                    total_best_only += 1
                    if idx < 3:
                        print(f"[worker-{worker_id}] FALLBACK best-only: stability={stability:.0f} consistency={consistency:.0f}")
                    parsed = best["parsed"]
                    if parsed is None:
                        parsed = extract_json_robust(raw)

                else:
                    # ✅ 高稳高洽 或 高稳低洽 → 正常 Top-3 融合
                    #    （高稳低洽的 Self-Refine 已在 infer_one_gepa 内触发）
                    total_fusion += 1
                    fused = fuse_candidates(all_scored)
                    if fused is not None:
                        parsed = fused
                    else:
                        parsed = best["parsed"]
                        if parsed is None:
                            parsed = extract_json_robust(raw)
            else:
                raw = server.infer_one(image_path, prompt)
                parsed = extract_json(raw)

            if not parsed:
                total_skipped_parse += 1
                if idx < 3:
                    print(f"[worker-{worker_id}] PARSE FAIL on {item['image_path']}: "
                          f"raw_preview={raw[:200].replace(chr(10), ' ')}")
                continue

            record = {
                "image_path": item["image_path"],
                "total_score": parsed.get("total_score"),
                "criteria": parsed.get("criteria"),
                "question": item.get("question"),
                "options": item.get("options"),
                "answer": parsed.get("answer")
            }

            results.append(record)

            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, out_path)

        except Exception as e:
            total_errors += 1
            print(f"[worker-{worker_id}] ERROR item {idx}: {e}")
            if idx < 3:
                import traceback
                traceback.print_exc()
            continue

    print(f"[worker-{worker_id}] DONE: processed={len(results)} "
          f"skipped_missing={total_skipped_missing} "
          f"skipped_parse={total_skipped_parse} "
          f"errors={total_errors}")
    if total_gepa_items > 0:
        avg_valid_candidates = total_valid_candidates / total_gepa_items
        avg_stability = total_stability / total_gepa_items
        avg_consistency = total_consistency / total_gepa_items
        print(f"[worker-{worker_id}] GEPA_BRANCH_STATS: items={total_gepa_items} "
              f"avg_valid_candidates={avg_valid_candidates:.2f} "
              f"fusion={total_fusion} "
              f"fallback_t02={total_fallback_t02} "
              f"best_only={total_best_only} "
              f"refine_triggered={total_refine_triggered} "
              f"refine_applied={total_refine_applied} "
              f"avg_stability={avg_stability:.2f} "
              f"avg_consistency={avg_consistency:.2f}")


# ================== Merge ==================
def merge_results():

    files = glob.glob(OUTPUT_JSON + ".part*.json")

    unique = {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for x in json.load(fp):
                unique[x["image_path"]] = x

    all_results = list(unique.values())

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ merged done: {len(all_results)}")


# ================== Main ==================
def main():
    global INFERENCE_MODE, GEPA_NUM_CANDIDATES, ENABLE_COT, ENABLE_SELF_REFINE
    global INPUT_JSON, OUTPUT_JSON, IMAGES_PATH, MODEL_NAME, PROCESSOR_NAME, MAX_NEW_TOKENS

    import argparse

    parser = argparse.ArgumentParser(description="PortraitCraft Track1 推理")
    parser.add_argument("--mode", choices=["baseline", "gepa"], default=INFERENCE_MODE,
                        help="推理模式: baseline=单次采样, gepa=多候选+聚合选优")
    parser.add_argument("--num_candidates", type=int, default=GEPA_NUM_CANDIDATES,
                        help="GEPA 候选数量 (默认 5)")
    parser.add_argument("--cot", action="store_true", default=ENABLE_COT,
                        help="启用 Chain-of-Thought prompt 模式")
    parser.add_argument("--no-cot", dest="cot", action="store_false",
                        help="禁用 Chain-of-Thought prompt 模式")
    parser.add_argument("--self_refine", action="store_true", default=ENABLE_SELF_REFINE,
                        help="启用 Self-Refinement 低分候选修正")
    parser.add_argument("--no-self_refine", dest="self_refine", action="store_false",
                        help="禁用 Self-Refinement")
    parser.add_argument("--input_json", type=str, default=INPUT_JSON)
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--images_path", type=str, default=IMAGES_PATH)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--processor_name", type=str, default=PROCESSOR_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    args = parser.parse_args()

    INFERENCE_MODE = args.mode
    GEPA_NUM_CANDIDATES = args.num_candidates
    ENABLE_COT = args.cot
    ENABLE_SELF_REFINE = args.self_refine
    INPUT_JSON = args.input_json
    OUTPUT_JSON = args.output_json
    IMAGES_PATH = args.images_path
    MODEL_NAME = args.model_name
    PROCESSOR_NAME = args.processor_name
    MAX_NEW_TOKENS = args.max_new_tokens

    run_config = {
        "mode": INFERENCE_MODE,
        "num_candidates": GEPA_NUM_CANDIDATES,
        "enable_cot": ENABLE_COT,
        "enable_self_refine": ENABLE_SELF_REFINE,
        "output_json": OUTPUT_JSON,
        "images_path": IMAGES_PATH,
        "model_name": MODEL_NAME,
        "processor_name": PROCESSOR_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gepa_temperatures": GEPA_TEMPERATURES[:GEPA_NUM_CANDIDATES],
        "consistency_threshold": CONSISTENCY_THRESHOLD
    }

    print("=" * 60)
    print("PortraitCraft Track1 推理")
    print(f"  模式: {INFERENCE_MODE}")
    print(f"  CoT: {'启用' if ENABLE_COT else '禁用'}")
    print(f"  Self-Refinement: {'启用' if ENABLE_SELF_REFINE else '禁用'}")
    print(f"  GEPA 候选数: {GEPA_NUM_CANDIDATES}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"  Images path: {IMAGES_PATH}")
    print("=" * 60)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total items in input: {len(data)}")

    # Build filename -> full_path index by scanning subdirectories (image00, image01, ...)
    file_index = {}
    if os.path.isdir(IMAGES_PATH):
        for entry in os.listdir(IMAGES_PATH):
            sub_path = os.path.join(IMAGES_PATH, entry)
            if os.path.isdir(sub_path):
                for fname in os.listdir(sub_path):
                    file_index[fname] = os.path.join(sub_path, fname)
    print(f"Built file index: {len(file_index)} files")

    # Resolve full paths for all items
    missing_from_index = 0
    for item in data:
        fname = item["image_path"]
        if fname in file_index:
            item["_full_path"] = file_index[fname]
        else:
            missing_from_index += 1
            item["_full_path"] = os.path.join(IMAGES_PATH, fname)
    if missing_from_index > 0:
        print(f"WARNING: {missing_from_index} images not found in file index")

    done_set = build_done_set_from_parts()

    new_data = []
    for item in data:
        image_path = item["image_path"]
        if image_path not in done_set:
            new_data.append(item)

    print(f"remaining to process: {len(new_data)}")

    data = new_data

    # Sanity check: verify at least first 3 images exist
    for item in data[:3]:
        resolved = item.get("_full_path", "")
        exists = os.path.exists(resolved)
        print(f"  [check] {item['image_path']} -> {resolved} exists={exists}")

    NUM_GPUS = torch.cuda.device_count()
    TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

    print(f"GPUs: {NUM_GPUS}, total workers: {TOTAL_WORKERS}")

    if len(data) == 0:
        print("nothing to process, merging existing results")
        merge_results()
        return

    chunk_size = math.ceil(len(data) / TOTAL_WORKERS)

    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    print(f"Divided into {len(chunks)} chunks, sizes: {[len(c) for c in chunks]}")

    processes = []

    for i, chunk in enumerate(chunks):

        gpu_id = i // WORKERS_PER_GPU

        p = Process(
            target=worker_run,
            args=(i, gpu_id, chunk, run_config)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_results()


# ================== Entry ==================
if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=0,3,5,7 python evaluation/evaluation_multi.py

"""
