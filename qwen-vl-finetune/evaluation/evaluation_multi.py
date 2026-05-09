import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from multiprocessing import Process
import math
import glob
import multiprocessing as mp
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

mp.set_start_method("spawn", force=True)
WORKERS_PER_GPU = 1


PROCESSOR_NAME = "./models/Qwen3-VL-4B-Instruct"

MODEL_NAME = "./models/Qwen3-VL-4B-Instruct"
# MODEL_NAME = "./output_singleimages/checkpoint-1"

DATASET_ROOT = "/root/autodl-tmp/PortraitCraft_dataset"

INPUT_JSON = os.path.join(DATASET_ROOT, "track_1_test.json")

OUTPUT_JSON = "/root/autodl-tmp/CVPR/qwen-vl-finetune/track_1_test_res.json"

IMAGES_PATH = DATASET_ROOT

MAX_NEW_TOKENS = 512

TRAIN_JSON = os.path.join(DATASET_ROOT, "track_1_train.json")

IMAGE_SUBDIRS = [""] + [f"images_{i:02d}" for i in range(11)]

CRITERIA_NAMES = [
    "Color Harmony",
    "Visual Style Consistency",
    "Sharpness",
    "Light and Shadow Modeling",
    "Creativity and Originality",
    "Exposure Control",
    "Application of Classical Composition Principles",
    "Depth of Field and Layering",
    "Visual Center Stability",
    "Visual Flow Guidance",
    "Structural Support Stability",
    "Appropriateness of Negative Space",
    "Subject Integrity",
]

LEVEL_TO_SCORE = {
    "Poor": 3.0,
    "Medium": 6.0,
    "Good": 8.5,
    "A": 3.0,
    "B": 6.0,
    "C": 8.5,
}

CRITERION_WEIGHTS = {
    "Color Harmony": 1.0,
    "Visual Style Consistency": 1.0,
    "Sharpness": 1.1,
    "Light and Shadow Modeling": 1.0,
    "Creativity and Originality": 0.9,
    "Exposure Control": 1.0,
    "Application of Classical Composition Principles": 1.15,
    "Depth of Field and Layering": 1.05,
    "Visual Center Stability": 1.2,
    "Visual Flow Guidance": 1.15,
    "Structural Support Stability": 1.05,
    "Appropriateness of Negative Space": 1.1,
    "Subject Integrity": 1.2,
}




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


def resolve_image_path(image_name):
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    for subdir in IMAGE_SUBDIRS:
        image_path = os.path.join(IMAGES_PATH, subdir, image_name)
        if os.path.exists(image_path):
            return image_path

    return os.path.join(IMAGES_PATH, image_name)



# ================== Model ==================
class DemoServer:
    def __init__(self, gpu_id):
        print(f"🚀 Loading model on GPU {gpu_id}...")

        self.device = torch.device(f"cuda:{gpu_id}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

        print(f"✅ Model loaded on GPU {gpu_id}")


    def infer_one(self, image_path, prompt):
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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.2,
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
def build_prompt(item):

    question = item["question"]
    options = item["options"]

    criteria_text = "\n".join([f"{i + 1}. {k}" for i, k in enumerate(CRITERIA_NAMES)])

    options_text = "\n".join([
        f"{k}. {v}"
        for k, v in options.items()
    ])

    prompt = f"""
You are an expert visual aesthetics evaluator.

You MUST analyze the image and output STRICT JSON ONLY.

---

TASKS:
1. Predict overall aesthetic score (1–100) based on visual evidence
2. Predict each criterion level: Good / Medium / Poor
3. Answer the multiple-choice question (A/B/C/D)

---

CRITERIA TO EVALUATE:
{criteria_text}

---

STRICT LEVEL THRESHOLDS:
For each criterion, first estimate a hidden numeric score from 0 to 10.
Do NOT output these hidden scores.

- Poor: score < 5. The flaw is obvious and materially harms portrait composition.
- Medium: 5 <= score < 7. The criterion is acceptable or ordinary, but has visible limitations.
- Good: score >= 7. The criterion is clearly above average with strong visual evidence.

Good does NOT mean merely acceptable.
Use Good only when the image provides clear positive evidence for that specific criterion.
Use Medium for ordinary, mixed, or mildly flawed cases.
Use Poor when the defect is visually clear, even if the overall image still looks good.
Do not assign most criteria as Good based only on a positive overall impression.

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
- answer MUST be grounded in visible evidence.
- If evidence for Good is not clear, do not choose Good.

---

Return ONLY valid JSON. No explanation. No markdown.
"""

    return prompt


# ================== JSON parser ==================
def extract_json(text):
    try:
        text = text.replace("```json", "").replace("```", "")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except:
        return None


def get_score_value(value):
    if isinstance(value, dict):
        if isinstance(value.get("score"), (int, float)):
            return max(0.0, min(10.0, float(value["score"])))
        value = value.get("level")

    if isinstance(value, str):
        value = value.strip()

    return LEVEL_TO_SCORE.get(value)


def compute_rank_proxy(item):
    criteria = item.get("criteria")
    if not isinstance(criteria, dict):
        return None

    weighted_sum = 0.0
    weight_sum = 0.0

    for name in CRITERIA_NAMES:
        if name not in criteria:
            continue

        score = get_score_value(criteria[name])
        if score is None:
            continue

        weight = CRITERION_WEIGHTS.get(name, 1.0)
        weighted_sum += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return None

    return weighted_sum / weight_sum


def get_int_score(value):
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None

    return max(1, min(100, score))


def load_train_score_distribution():
    if not os.path.exists(TRAIN_JSON):
        print(f"⚠️ train json not found, fallback to criteria proxy score: {TRAIN_JSON}")
        return []

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []
    for item in data:
        score = get_int_score(item.get("total_score"))
        if score is not None:
            scores.append(score)

    scores.sort()
    print(f"✅ loaded train score distribution: {len(scores)}")

    return scores


def apply_rankiqa_scores(records):
    if not records:
        return records

    train_scores = load_train_score_distribution()
    ranked = []

    for idx, item in enumerate(records):
        proxy = compute_rank_proxy(item)
        raw_score = get_int_score(item.get("total_score"))

        if proxy is None and raw_score is not None:
            proxy = raw_score / 10.0

        if proxy is None:
            proxy = 6.0

        ranked.append((proxy, raw_score or 0, item.get("image_path", ""), idx))

    ranked.sort()

    if train_scores:
        max_train_idx = len(train_scores) - 1
        max_rank_idx = max(1, len(ranked) - 1)

        for rank, (_, _, _, idx) in enumerate(ranked):
            train_idx = round(rank * max_train_idx / max_rank_idx)
            records[idx]["total_score"] = train_scores[train_idx]
    else:
        for _, _, _, idx in ranked:
            proxy = compute_rank_proxy(records[idx])
            if proxy is not None:
                records[idx]["total_score"] = max(1, min(100, int(round(proxy * 10))))

    print(f"✅ applied RankIQA-style score calibration: {len(records)}")

    return records


# ================== NEW：读取历史结果 ==================
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
def worker_run(worker_id, gpu_id, data_chunk):

    server = DemoServer(gpu_id)

    out_path = OUTPUT_JSON + f".part{worker_id}.json"

    # ====== 只加载已有结果（不再做过滤）=====
    results = []

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"♻️ worker {worker_id} load existing: {len(results)}")
        except:
            results = []

    for idx, item in enumerate(tqdm(data_chunk, desc=f"worker-{worker_id}")):

        image_path = resolve_image_path(item["image_path"])

        if not os.path.exists(image_path):
            continue

        prompt = build_prompt(item)

        try:
            raw = server.infer_one(image_path, prompt)
            parsed = extract_json(raw)

            if not parsed:
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

            # ====== 每条都写 ======
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, out_path)

        except Exception as e:
            print(f"error worker {worker_id}:", e)
            continue

    print(f"✅ worker {worker_id} done, total: {len(results)}")


# ================== Merge ==================
def merge_results():

    files = glob.glob(OUTPUT_JSON + ".part*.json")

    unique = {}

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for x in json.load(fp):
                unique[x["image_path"]] = x

    all_results = apply_rankiqa_scores(list(unique.values()))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"✅ merged done: {len(all_results)}")


# ================== Main ==================
def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ====== 🔥 核心：读取历史结果 ======
    done_set = build_done_set_from_parts()

    # ====== 🔥 过滤未完成数据 ======
    new_data = []
    for item in data:
        image_path = item["image_path"]
        if image_path not in done_set:
            new_data.append(item)

    print(f"🚀 remaining to process: {len(new_data)}")

    data = new_data

    NUM_GPUS = torch.cuda.device_count()
    

    TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU

    print(f"🚀 GPUs: {NUM_GPUS}, total workers: {TOTAL_WORKERS}")

    if len(data) == 0:
        print("✅ nothing to process")
        merge_results()
        return

    chunk_size = math.ceil(len(data) / TOTAL_WORKERS)

    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    processes = []

    for i, chunk in enumerate(chunks):

        gpu_id = i // WORKERS_PER_GPU

        p = Process(
            target=worker_run,
            args=(i, gpu_id, chunk)
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
