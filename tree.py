import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# 配置区：按需修改
# =========================
SRC = Path("~/autodl-tmp/data/coco").expanduser()
DST = Path("~/autodl-tmp/data/coco-20").expanduser()

TRAIN_JSON = SRC / "annotations/instances_train2017.json"
VAL_JSON = SRC / "annotations/instances_val2017.json"

SRC_TRAIN_DIR = SRC / "train2017"
SRC_VAL_DIR = SRC / "val2017"

# 你的 20 类（固定）
CLASSES = [
    "person", "chair", "car", "dining table", "cup", "bottle", "bowl",
    "handbag", "truck", "bench", "backpack", "book", "cell phone",
    "sink", "clock", "tv", "potted plant", "couch", "dog", "knife"
]

TARGET_TRAIN_TOTAL_IMAGES = 15000  # 从 train2017 抽出来的总量（train+val）
VAL_SPLIT_RATIO = 0.10            # 从 15000 中切多少做 val（训练内验证集）
PERSON_CAP = 20000                # person 用于“比例权重”的上限（你要求的 20k）
SEED = 42                         # 可复现

# 复制策略
COPY_OVERWRITE = False            # True 覆盖已存在文件；False 跳过已存在文件
PRINT_EVERY = 500                 # 复制进度打印频率

random.seed(SEED)


# =========================
# 工具函数
# =========================
def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def save_json(obj, p: Path):
    with open(p, "w") as f:
        json.dump(obj, f)


def filter_to_coco20(data):
    """只保留指定 20 类；并丢弃不含这 20 类任何实例的图片。"""
    cats = [c for c in data["categories"] if c["name"] in CLASSES]
    cat_ids = {c["id"] for c in cats}

    anns = [a for a in data["annotations"] if a["category_id"] in cat_ids]
    keep_img_ids = {a["image_id"] for a in anns}
    imgs = [i for i in data["images"] if i["id"] in keep_img_ids]

    return {"images": imgs, "annotations": anns, "categories": cats}


def split_by_image_ids(data, keep_ids: set):
    """按 image_id 过滤 images 和 annotations。"""
    imgs = [i for i in data["images"] if i["id"] in keep_ids]
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    return {"images": imgs, "annotations": anns, "categories": data["categories"]}


def build_train_total_subset_greedy(train20):
    """
    从 train20（已过滤到 20 类）里贪心选出约 TARGET_TRAIN_TOTAL_IMAGES 张图。
    采用“按类别图像数做配额”，其中 person 的配额权重用 min(实际, PERSON_CAP)。
    多标签场景用“补齐收益最大”的图片优先。
    """
    cat_id_to_name = {c["id"]: c["name"] for c in train20["categories"]}

    # 类别 -> 包含该类别的 image_id 集合
    cat_to_imgs = defaultdict(set)
    for ann in train20["annotations"]:
        cat_to_imgs[ann["category_id"]].add(ann["image_id"])

    # image_id -> 该图包含哪些类别（cat_id）
    img_to_cats = defaultdict(set)
    for ann in train20["annotations"]:
        img_to_cats[ann["image_id"]].add(ann["category_id"])

    # 权重（person 截断 20k）
    weights = {}
    for cid, imgs in cat_to_imgs.items():
        name = cat_id_to_name[cid]
        w = len(imgs)
        if name == "person":
            w = min(w, PERSON_CAP)
        weights[cid] = w

    total_weight = sum(weights.values())
    if total_weight == 0:
        raise RuntimeError("No data left after filtering classes. Check CLASSES list.")

    # 配额：按权重分配到总图数，设一个下限防止某类被抽得太少
    quota = {
        cid: max(300, round(TARGET_TRAIN_TOTAL_IMAGES * w / total_weight))
        for cid, w in weights.items()
    }

    count = defaultdict(int)  # 当前已选图片对每类的覆盖数
    selected = set()

    # 贪心循环：每次找“最缺”的类别，然后挑一张能补最多缺口类别的图
    while len(selected) < TARGET_TRAIN_TOTAL_IMAGES:
        # 找 count/quota 最小的类别（最缺）
        cid_need = min(quota, key=lambda c: count[c] / quota[c])

        candidates = list(cat_to_imgs[cid_need] - selected)
        if not candidates:
            # 这个类别已经没有候选图可以补了，退出
            break

        # 选择能同时补最多“仍未达配额类别”的图片
        def gain(img_id):
            g = 0
            for c in img_to_cats[img_id]:
                if c in quota and count[c] < quota[c]:
                    g += 1
            return g

        best_img = max(candidates, key=gain)
        selected.add(best_img)

        for c in img_to_cats[best_img]:
            if c in quota:
                count[c] += 1

    # 生成子集
    subset = split_by_image_ids(train20, selected)

    # 如果因为某类候选耗尽导致不足 15k，可再随机补齐到 15k（从剩余中补）
    if len(subset["images"]) < TARGET_TRAIN_TOTAL_IMAGES:
        need = TARGET_TRAIN_TOTAL_IMAGES - len(subset["images"])
        all_img_ids = {i["id"] for i in train20["images"]}
        remain = list(all_img_ids - selected)
        random.shuffle(remain)
        extra = set(remain[:need])
        selected2 = selected | extra
        subset = split_by_image_ids(train20, selected2)

    return subset


def split_train_val(train_total):
    """
    将 train_total（约 15k）划分为 train/val。
    """
    img_ids = [i["id"] for i in train_total["images"]]
    random.shuffle(img_ids)

    val_size = int(round(len(img_ids) * VAL_SPLIT_RATIO))
    val_ids = set(img_ids[:val_size])
    train_ids = set(img_ids[val_size:])

    train_part = split_by_image_ids(train_total, train_ids)
    val_part = split_by_image_ids(train_total, val_ids)
    return train_part, val_part


def copy_images(images, src_dir: Path, dst_dir: Path, split_name: str):
    total = len(images)
    copied = 0
    skipped = 0
    missing = 0

    for idx, img in enumerate(images, 1):
        fname = img.get("file_name")
        if not fname:
            raise RuntimeError(f"[{split_name}] image missing file_name: {img}")

        src = src_dir / fname
        dst = dst_dir / fname

        if dst.exists() and not COPY_OVERWRITE:
            skipped += 1
        else:
            if not src.exists():
                missing += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

        if idx % PRINT_EVERY == 0 or idx == total:
            print(f"[{split_name}] {idx}/{total} processed | copied={copied} skipped={skipped} missing={missing}")

    if missing:
        print(f"[{split_name}] WARNING: {missing} source files were missing. Check dataset integrity.")
    return copied, skipped, missing


def main():
    # 创建目录结构（你要的语义：train/val/test）
    ann_dir = DST / "annotations"
    dst_train_dir = DST / "train"
    dst_val_dir = DST / "val"
    dst_test_dir = DST / "test"
    ann_dir.mkdir(parents=True, exist_ok=True)
    dst_train_dir.mkdir(parents=True, exist_ok=True)
    dst_val_dir.mkdir(parents=True, exist_ok=True)
    dst_test_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO JSON...")
    train = load_json(TRAIN_JSON)
    val = load_json(VAL_JSON)

    print("Filtering to COCO-20 classes...")
    train20 = filter_to_coco20(train)
    test20 = filter_to_coco20(val)   # 原 val2017 过滤后作为 test

    print("Building train_total subset (~15k) from train2017...")
    train_total = build_train_total_subset_greedy(train20)

    print(f"Splitting train_total into train/val with ratio {1-VAL_SPLIT_RATIO:.2f}/{VAL_SPLIT_RATIO:.2f} ...")
    train_part, val_part = split_train_val(train_total)

    # 保存 JSON（语义清晰：train/val/test）
    train_json_out = ann_dir / "instances_train20.json"
    val_json_out = ann_dir / "instances_val20.json"
    test_json_out = ann_dir / "instances_test20.json"

    print("Saving annotations JSON...")
    save_json(train_part, train_json_out)
    save_json(val_part, val_json_out)
    save_json(test20, test_json_out)

    print("\nCopying images...")
    print(f"Train images: {len(train_part['images'])}")
    print(f"Val images:   {len(val_part['images'])}")
    print(f"Test images:  {len(test20['images'])}")

    copy_images(train_part["images"], SRC_TRAIN_DIR, dst_train_dir, "train")
    copy_images(val_part["images"], SRC_TRAIN_DIR, dst_val_dir, "val")
    copy_images(test20["images"], SRC_VAL_DIR, dst_test_dir, "test")

    print("\nDone.")
    print(f"Dataset root: {DST}")
    print("Generated files:")
    print(f"  - {train_json_out}")
    print(f"  - {val_json_out}")
    print(f"  - {test_json_out}")
    print("Copied images into:")
    print(f"  - {dst_train_dir}")
    print(f"  - {dst_val_dir}")
    print(f"  - {dst_test_dir}")


if __name__ == "__main__":
    main()
