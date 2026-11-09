"""Generate language features from per-class GT PNG masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_mask(path, shape):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"{path}: expected grayscale mask {shape}, got {array.shape}")
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError(f"{path}: expected binary values 0/1/255")
    return array != 0


def build_segmentation(folder, shape, overlap):
    paths = sorted(p for p in folder.glob("*.png") if p.stem.lower() != "background")
    seg = np.full(shape, -1, dtype=np.int32)
    objects = []
    for path in paths:
        mask = read_mask(path, shape)
        if not mask.any():
            continue
        conflicts = mask & (seg >= 0)
        if conflicts.any() and overlap == "error":
            raise ValueError(f"{path}: {int(conflicts.sum())} overlapping foreground pixels; "
                             "choose --overlap first or last explicitly if intended")
        if overlap == "first":
            mask &= seg < 0
        if not mask.any():
            continue
        index = len(objects)
        seg[mask] = index
        objects.append({"label": path.stem.replace("_", " "), "caption": "",
                        "source_mask": path.name, "id": index})
    background = folder / "background.png"
    if background.exists():
        mask = read_mask(background, shape)
        if (mask & (seg >= 0)).any():
            raise ValueError(f"{background}: background overlaps foreground")
    # Explicit background and any unlabeled pixels share the final zero-vector row.
    seg[seg < 0] = len(objects)
    levels = np.full((4, *shape), -1, dtype=np.int32)
    levels[0] = seg
    return levels, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Scene directory or parent containing scene directories")
    parser.add_argument("--output_subdir", default="language_features")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overlap", choices=["error", "first", "last"], default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate all masks without loading a model or writing outputs")
    args = parser.parse_args()
    root = args.dataset_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = Path(args.output_subdir)
    if out.is_absolute() or ".." in out.parts or out == Path("."):
        parser.error("--output_subdir must be a relative subdirectory")
    scenes = [root] if (root / "segmentations").is_dir() else sorted(
        p.parent for p in root.rglob("segmentations") if p.is_dir())
    if not scenes:
        raise RuntimeError(f"No segmentations directories under {root}")
    jobs = []
    labels = set()
    for scene in scenes:
        images = {p.stem: p for p in (scene / "images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        folders = sorted(p for p in (scene / "segmentations").iterdir() if p.is_dir())
        for name in sorted(set(images) - {p.name for p in folders}):
            print(f"[WARNING] No GT, skipping {scene.name}/{name}")
        for folder in folders:
            if folder.name not in images:
                raise FileNotFoundError(f"No corresponding image for {folder}")
            if not list(folder.glob("*.png")):
                raise ValueError(f"No PNG masks in {folder}")
            with Image.open(images[folder.name]) as image:
                shape = (image.height, image.width)
            levels, objects = build_segmentation(folder, shape, args.overlap)
            target = scene / out
            names = [target / (folder.name + suffix)
                     for suffix in ("_s.npy", "_f.npy", "_filtered.json")]
            if not args.validate_only and not args.overwrite and any(p.exists() for p in names):
                raise FileExistsError(f"Outputs already exist for {folder}; use --overwrite")
            labels.update(obj["label"] for obj in objects)
            jobs.append((folder, shape, names))
            print(f"[VALID] {scene.name}/{folder.name}: {len(objects)} classes, shape={shape}")
    if not jobs:
        raise RuntimeError("No GT views found")
    if args.validate_only:
        print(f"Validated {len(jobs)} views; no files written.")
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_name, device=args.device)
    model.max_seq_length = args.max_seq_length
    ordered = sorted(labels)
    cache = {}
    if ordered:
        vectors = model.encode(ordered, convert_to_numpy=True, show_progress_bar=True)
        cache = dict(zip(ordered, vectors))
    dim = model.get_sentence_embedding_dimension()
    for folder, shape, names in jobs:
        levels, objects = build_segmentation(folder, shape, args.overlap)
        features = np.zeros((len(objects) + 1, dim), dtype=np.float32)
        for obj in objects:
            features[obj["id"]] = cache[obj["label"]]
        names[0].parent.mkdir(parents=True, exist_ok=True)
        np.save(names[0], levels)
        np.save(names[1], features)
        names[2].write_text(json.dumps({
            "objects": objects, "background_id": len(objects),
            "model_name": args.model_name, "max_seq_length": args.max_seq_length,
            "source": str(folder), "overlap_policy": args.overlap
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] {names[1]}: {features.shape}")


if __name__ == "__main__":
    main()
