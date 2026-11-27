#!/usr/bin/env python3
"""Controlled upper-bound perception producer: PatchCore-style anomaly detection (GPU, run once).

A controlled detector producer for the replay: per category, a memory bank of mid-level
WideResNet-50 patch features from nominal images, and the nearest-neighbour distance of
each test patch to that bank as its anomaly score (Roth et al., CVPR 2022, pp. 14298-14308). The bank is
built only from nominal images of the source datasets that are *not* among the 2,000
benchmark frames (matched by SHA-256), so no benchmark image is seen during fitting.

For every benchmark frame the script stores the image score (maximum patch score), a
56x56 anomaly map, and the measured per-frame inference latency. Thresholds are *not*
chosen here: ``run_detector_analysis.py`` calibrates them on the training split only.

Needs torch and torchvision with CUDA (not part of the offline artifact):

    python experiments/detector_producer.py            # ~20 min on one L40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "datasets"
MANIFEST = DATA / "manufacturing-a1" / "manifest.jsonl"
FRAMES = DATA / "manufacturing-a1"
RAW = DATA / "raw"
OUT = ROOT / "experiments" / "results" / "detector"

SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def nominal_pool(source: str, category: str) -> list[Path]:
    """Nominal images of one category in the raw datasets."""
    if source == "mvtec_ad":
        base = RAW / "mvtec_ad" / category
        return sorted((base / "train" / "good").glob("*.png")) + sorted((base / "test" / "good").glob("*.png"))
    return sorted((RAW / category / "Data" / "Images" / "Normal").glob("*.JPG"))


def load(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return (x - MEAN) / STD


class Features(torch.nn.Module):
    """Locally aggregated layer-2 and layer-3 features of an ImageNet WideResNet-50-2."""

    def __init__(self) -> None:
        super().__init__()
        net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)                      # (B, 1536, 28, 28)
        f = F.adaptive_avg_pool1d(f.flatten(2).transpose(1, 2), 1024)  # (B, 784, 1024)
        return f


def greedy_coreset(feats: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    """k-center greedy subsampling on a random 128-d projection (PatchCore coreset)."""
    g = torch.Generator(device=feats.device).manual_seed(seed)
    proj = torch.randn(feats.shape[1], 128, device=feats.device, generator=g) / 128 ** 0.5
    z = feats @ proj
    n = max(1, int(ratio * len(z)))
    idx = [int(torch.randint(len(z), (1,), generator=g, device=feats.device))]
    dmin = torch.cdist(z, z[idx[-1]].unsqueeze(0)).squeeze(1)
    for _ in range(n - 1):
        nxt = int(torch.argmax(dmin))
        idx.append(nxt)
        dmin = torch.minimum(dmin, torch.cdist(z, z[nxt].unsqueeze(0)).squeeze(1))
    return feats[torch.tensor(idx, device=feats.device)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-category", type=int, default=200)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    frames = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    bench_hashes = {f["evidence"]["scene_hash"] for f in frames}
    cats = sorted({(f["source_dataset"], f["category"]) for f in frames})
    model = Features().to(dev)
    torch.backends.cudnn.benchmark = False

    rows, maps, ids = [], [], []
    fit_info = {}
    for src, cat in cats:
        pool = [p for p in nominal_pool(src, cat) if sha16(p) not in bench_hashes]
        rng = random.Random(f"{args.seed}:{src}:{cat}")
        train = rng.sample(pool, min(args.train_per_category, len(pool)))
        t0 = time.perf_counter()
        feats = torch.cat([model(load(p).to(dev)).squeeze(0) for p in train])
        bank = greedy_coreset(feats, args.coreset, args.seed)
        fit_s = time.perf_counter() - t0
        fit_info[f"{src}/{cat}"] = {"nominal_pool": len(pool), "train_images": len(train),
                                    "bank_patches": int(len(bank)), "fit_s": fit_s}
        print(f"[detector] {src}/{cat}: pool {len(pool)}, train {len(train)}, bank {len(bank)}, {fit_s:.1f}s", flush=True)
        for f in (f for f in frames if (f["source_dataset"], f["category"]) == (src, cat)):
            path = FRAMES / f["path"]
            torch.cuda.synchronize()
            t = time.perf_counter()
            x = load(path).to(dev)
            patch = model(x).squeeze(0)                    # (784, 1024)
            d = torch.cdist(patch, bank).min(dim=1).values.view(1, 1, 28, 28)
            amap = F.interpolate(d, size=(56, 56), mode="bilinear", align_corners=False).squeeze()
            score = float(d.max())
            torch.cuda.synchronize()
            latency = (time.perf_counter() - t) * 1000.0
            rows.append({"frame_id": f["frame_id"], "source_dataset": src, "category": cat,
                         "image_score": score, "latency_ms": latency})
            maps.append(amap.half().cpu().numpy())
            ids.append(f["frame_id"])
        del feats, bank
        torch.cuda.empty_cache()

    np.savez_compressed(OUT / "patchcore_maps.npz", frame_id=np.array(ids), maps=np.stack(maps))
    (OUT / "patchcore_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "patchcore_fit.json").write_text(json.dumps({
        "method": "PatchCore-style kNN on WideResNet-50-2 layer2+3 features (ImageNet), 224 px",
        "train_per_category": args.train_per_category, "coreset_ratio": args.coreset, "seed": args.seed,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0), "categories": fit_info,
    }, indent=2) + "\n")
    print(f"[detector] wrote {len(rows)} frames to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
