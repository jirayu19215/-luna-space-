#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pool all LuSNAR scenes and re-split into train/val/test in the layout
ESANet expects.

The LuSNAR dataset ships as several scenes, each with a different number of
frames.  This tool walks every scene, matches each RGB frame with its depth
and semantic-label counterpart, pools everything, splits it into
train/val/test, and writes the result in the directory structure ESANet's
dataset loader reads:

    <output_dir>/
      train.txt  val.txt  test.txt        # one sample id per line (no ext)
      train/  val/  test/
        rgb/     <id>.png
        depth/   <id>.png
        labels/  <id>.png
      dataset_stats.json                   # depth mean/std, counts, class hist

Two split modes:
  * scene   (recommended) — whole scenes go to one split, so temporally
                            correlated frames never leak across splits.
  * random  — pool every frame then shuffle-split by ratio.

Because LuSNAR folder/label conventions vary, the RGB/depth/label sub-folder
names and file extensions are configurable.  Run with --dry_run first to see
what it detected without copying anything.

Example (Windows):
  python prepare_lusnar.py ^
    --input_root "D:\\Lunar\\LuSNAR_data" ^
    --output_dir "D:\\Lunar\\lusnar_esanet" ^
    --rgb_subdir image --depth_subdir depth --label_subdir semantic ^
    --split_mode scene --ratios 0.7 0.15 0.15 --seed 42
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from glob import glob

import cv2
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def find_scenes(root, scene_glob):
    """Return sorted scene directories under root."""
    if scene_glob:
        scenes = [p for p in glob(os.path.join(root, scene_glob)) if os.path.isdir(p)]
    else:
        scenes = [os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))]
    return sorted(scenes)


def index_dir(d):
    """Map file-stem -> full path for every image in directory d (recursive)."""
    out = {}
    if not os.path.isdir(d):
        return out
    for dirpath, _, files in os.walk(d):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                out[stem] = os.path.join(dirpath, f)
    return out


def collect_samples(scenes, rgb_sub, depth_sub, label_sub):
    """Build a flat list of matched (scene, id, rgb, depth, label) samples."""
    samples = []
    per_scene = defaultdict(int)
    skipped = 0
    for scene in scenes:
        name = os.path.basename(scene.rstrip('/\\'))
        rgb = index_dir(os.path.join(scene, rgb_sub))
        depth = index_dir(os.path.join(scene, depth_sub))
        label = index_dir(os.path.join(scene, label_sub))
        for stem, rgb_fp in sorted(rgb.items()):
            if stem in depth and stem in label:
                uid = f"{name}__{stem}"          # globally-unique id
                samples.append({
                    'scene': name, 'id': uid,
                    'rgb': rgb_fp, 'depth': depth[stem], 'label': label[stem],
                })
                per_scene[name] += 1
            else:
                skipped += 1
    return samples, per_scene, skipped


def split_random(samples, ratios, seed):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    parts = {'train': idx[:n_tr], 'val': idx[n_tr:n_tr + n_va], 'test': idx[n_tr + n_va:]}
    return {k: [samples[i] for i in v] for k, v in parts.items()}


def split_scene(samples, ratios, seed):
    by_scene = defaultdict(list)
    for s in samples:
        by_scene[s['scene']].append(s)
    total = len(samples)
    out = {'train': [], 'val': [], 'test': []}
    targets = {'train': ratios[0] * total, 'val': ratios[1] * total, 'test': ratios[2] * total}
    # assign the largest scenes first (ties broken by seed) to the split that is
    # furthest below its target *relative to that target* -> better balance and
    # far less likely to starve val/test than a plain shuffle.
    rng = random.Random(seed)
    scenes = sorted(by_scene, key=lambda s: (-len(by_scene[s]), rng.random()))
    for sc in scenes:
        deficit = {k: (targets[k] - len(out[k])) / max(targets[k], 1e-9) for k in out}
        pick = max(deficit, key=deficit.get)
        out[pick].extend(by_scene[sc])
    empty = [k for k, v in out.items() if not v]
    if empty:
        print(f"[!] scene-split left {empty} empty (too few scenes for these "
              f"ratios). Use --split_mode random, or move a scene manually.")
    return out


def compute_depth_stats(train_samples, max_files=500):
    """Mean/std of depth over (a sample of) the training split."""
    if not train_samples:
        return None, None
    rng = random.Random(0)
    pick = train_samples if len(train_samples) <= max_files else rng.sample(train_samples, max_files)
    s, ss, cnt = 0.0, 0.0, 0
    for smp in pick:
        d = cv2.imread(smp['depth'], cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = d.astype(np.float64)
        m = d > 0                       # ignore invalid/zero depth
        if m.any():
            v = d[m]
            s += v.sum(); ss += (v * v).sum(); cnt += v.size
    if cnt == 0:
        return None, None
    mean = s / cnt
    std = (ss / cnt - mean ** 2) ** 0.5
    return float(mean), float(std)


def label_histogram(samples, max_files=200):
    rng = random.Random(1)
    pick = samples if len(samples) <= max_files else rng.sample(samples, max_files)
    hist = defaultdict(int)
    for smp in pick:
        lab = cv2.imread(smp['label'], cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        if lab.ndim == 3:               # color-coded label -> flag it
            hist['__is_color__'] += 1
            continue
        u, c = np.unique(lab, return_counts=True)
        for cls, cnt in zip(u.tolist(), c.tolist()):
            hist[int(cls)] += int(cnt)
    return dict(sorted(hist.items(), key=lambda kv: str(kv[0])))


def write_split(split_name, items, out_dir, copy):
    sub = os.path.join(out_dir, split_name)
    for kind in ('rgb', 'depth', 'labels'):
        os.makedirs(os.path.join(sub, kind), exist_ok=True)
    ids = []
    op = shutil.copy2 if copy else os.symlink
    for it in items:
        ids.append(it['id'])
        for kind, key in (('rgb', 'rgb'), ('depth', 'depth'), ('labels', 'label')):
            dst = os.path.join(sub, kind, it['id'] + '.png')
            if not os.path.exists(dst):
                try:
                    op(it[key], dst)
                except FileExistsError:
                    pass
    with open(os.path.join(out_dir, f'{split_name}.txt'), 'w') as f:
        f.write('\n'.join(ids) + ('\n' if ids else ''))
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input_root', required=True, help='LuSNAR root (holds the scene folders)')
    ap.add_argument('--output_dir', required=True, help='where the re-split dataset is written')
    ap.add_argument('--scene_glob', default='', help='glob for scene dirs (default: every subdir)')
    ap.add_argument('--rgb_subdir', default='image')
    ap.add_argument('--depth_subdir', default='depth')
    ap.add_argument('--label_subdir', default='semantic')
    ap.add_argument('--split_mode', choices=['scene', 'random'], default='scene')
    ap.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=('TRAIN', 'VAL', 'TEST'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--symlink', action='store_true', help='symlink instead of copy (saves disk)')
    ap.add_argument('--dry_run', action='store_true', help='detect + report, write nothing')
    args = ap.parse_args()

    assert abs(sum(args.ratios) - 1.0) < 1e-6, 'ratios must sum to 1.0'

    scenes = find_scenes(args.input_root, args.scene_glob)
    print(f"Found {len(scenes)} scene(s): {[os.path.basename(s) for s in scenes]}")
    samples, per_scene, skipped = collect_samples(
        scenes, args.rgb_subdir, args.depth_subdir, args.label_subdir)
    print(f"Matched {len(samples)} RGB+depth+label triplets "
          f"(skipped {skipped} unmatched).")
    for sc, n in per_scene.items():
        print(f"    {sc}: {n}")
    if not samples:
        print("\n[!] No triplets matched. Check --rgb_subdir/--depth_subdir/"
              "--label_subdir and --scene_glob against your folder layout.")
        return

    splitter = split_scene if args.split_mode == 'scene' else split_random
    splits = splitter(samples, args.ratios, args.seed)
    print(f"\nSplit ({args.split_mode}): "
          + ', '.join(f"{k}={len(v)}" for k, v in splits.items()))

    stats = {
        'split_mode': args.split_mode, 'ratios': args.ratios, 'seed': args.seed,
        'counts': {k: len(v) for k, v in splits.items()},
        'per_scene': dict(per_scene),
    }
    mean, std = compute_depth_stats(splits['train'])
    stats['depth_mean'], stats['depth_std'] = mean, std
    stats['train_label_histogram'] = label_histogram(splits['train'])
    print(f"depth_mean={mean}  depth_std={std}")
    print(f"label values seen (train): {list(stats['train_label_histogram'].keys())}")

    if args.dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry_run to materialize.")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    for name in ('train', 'val', 'test'):
        write_split(name, splits[name], args.output_dir, copy=not args.symlink)
    with open(os.path.join(args.output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nDone -> {args.output_dir}")
    print("Next: point an ESANet LuSNAR dataset class at this folder "
          "(depth_mean/std above go into that class).")


if __name__ == '__main__':
    main()
