import argparse
import json


LEVEL_TO_ABC = {
    "Poor": "A",
    "Medium": "B",
    "Good": "C",
}

VALID_LEVELS = {"A", "B", "C", "NOT_RES"}


def normalize_level(value):
    if isinstance(value, dict):
        level = value.get("level")
        if level in VALID_LEVELS:
            return level
        return LEVEL_TO_ABC.get(level, "NOT_RES")

    return LEVEL_TO_ABC.get(value, "NOT_RES")


def convert_item(item):
    criteria = item.get("criteria")
    if not isinstance(criteria, dict):
        return None

    new_criteria = {}
    for key, value in criteria.items():
        new_criteria[key] = {"level": normalize_level(value)}

    return {
        "image_path": item.get("image_path"),
        "criteria": new_criteria,
        "total_score": item.get("total_score"),
        "question": item.get("question"),
        "options": item.get("options"),
        "answer": item.get("answer"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert test result JSON criteria from Poor/Medium/Good to A/B/C level format."
    )
    parser.add_argument(
        "--input",
        default="./track_1_test_res.json",
        help="Input JSON path. Default: ./track_1_test_res.json",
    )
    parser.add_argument(
        "--output",
        default="./track_1_test_res_final.json",
        help="Output JSON path. Default: ./track_1_test_res_final.json",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = convert_item(item)
        if new_item is not None:
            new_data.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"converted: {len(new_data)} items -> {args.output}")


if __name__ == "__main__":
    main()
