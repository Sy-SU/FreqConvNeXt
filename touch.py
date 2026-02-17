import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("~/autodl-tmp/data/coco-20").expanduser()
ANN_DIR = ROOT / "annotations"

FILES = {
    "train": ANN_DIR / "instances_train20.json",
    "val": ANN_DIR / "instances_val20.json",
    "test": ANN_DIR / "instances_test20.json",
}


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def stat_split(name, data):
    print(f"\n========== {name.upper()} ==========")

    images = data["images"]
    anns = data["annotations"]
    cats = data["categories"]

    print(f"Images:      {len(images)}")
    print(f"Annotations: {len(anns)}")
    print(f"Categories:  {len(cats)}")

    # 类别 id -> 名字
    id2name = {c["id"]: c["name"] for c in cats}

    # 统计
    cat_to_imgs = defaultdict(set)
    cat_to_anns = defaultdict(int)

    for ann in anns:
        cid = ann["category_id"]
        cat_to_imgs[cid].add(ann["image_id"])
        cat_to_anns[cid] += 1

    print("\nPer-category statistics:")
    print(f"{'Class':20s} {'Images':>8s} {'Instances':>10s}")

    for c in cats:
        cid = c["id"]
        name = c["name"]
        img_cnt = len(cat_to_imgs[cid])
        ann_cnt = cat_to_anns[cid]
        print(f"{name:20s} {img_cnt:8d} {ann_cnt:10d}")


def main():
    print("COCO-20 Dataset Statistics")
    print(f"Root: {ROOT}")

    total_images = 0
    total_anns = 0

    for split, path in FILES.items():
        if not path.exists():
            print(f"\n[WARNING] Missing file: {path}")
            continue

        data = load_json(path)
        stat_split(split, data)

        total_images += len(data["images"])
        total_anns += len(data["annotations"])

    print("\n========== TOTAL ==========")
    print(f"Total Images:      {total_images}")
    print(f"Total Annotations: {total_anns}")


if __name__ == "__main__":
    main()
