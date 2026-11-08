"""Retinal fundus image registration on the FIRE dataset, evaluated with the ground-truth control points.

For every image pair (reference = *_1.jpg, moving = *_2.jpg) the script compares:

1. No registration (identity): how far apart the corresponding points are to begin with.
2. Ground-truth fits, leave-one-out: a transform (similarity, affine or homography) fitted by least
   squares on 9 of the 10 control points and tested on the 10th, repeated for all 10. This shows how
   well each motion model can describe the pair, without using the test point for fitting.
3. Automatic registration: SIFT keypoints on the contrast-enhanced green channel, ratio-test
   matching and a RANSAC homography. The control points are NOT used here; they serve only to
   measure the error.

Error = target registration error (TRE): distance in pixels between a mapped moving point and its
reference point. Image similarity (normalised cross-correlation and mutual information inside the
overlapping field of view) is also reported before and after automatic registration.

Data: FIRE (Hernandez-Matas et al., 2017), 134 pairs of 2912 x 2912 px fundus images in three
categories: S (large overlap), P (small overlap) and A (large overlap with anatomical changes).

Usage:
    python retinal_registration.py --fire FIRE              # all pairs, writes results/ and figures/
    python retinal_registration.py --fire FIRE --category A
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ("similarity", "affine", "homography")


# ---------------------------------------------------------------- data
def find_pairs(fire_dir: Path) -> list[dict]:
    """Locate image pairs and their control-point files anywhere below fire_dir."""
    images = {p.stem: p for p in fire_dir.rglob("*.jpg")}
    pairs = []
    for gt in sorted(fire_dir.rglob("control_points_*_1_2.txt")):
        name = re.match(r"control_points_(\w+?)_1_2", gt.stem).group(1)
        if f"{name}_1" in images and f"{name}_2" in images:
            pairs.append({"pair": name, "category": name[0], "reference": images[f"{name}_1"],
                          "moving": images[f"{name}_2"], "points": gt})
    return pairs


def read_control_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """FIRE format: one point per line, 'x_ref y_ref x_moving y_moving'."""
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0:2].astype(float), data[:, 2:4].astype(float)


# ---------------------------------------------------------------- transforms (3 x 3, moving -> reference)
def fit_transform(src: np.ndarray, dst: np.ndarray, model: str) -> np.ndarray:
    """Least-squares fit of a transform mapping src points onto dst points."""
    n = len(src)
    if model == "similarity":  # x' = a x - b y + tx, y' = b x + a y + ty
        A = np.zeros((2 * n, 4))
        A[0::2] = np.c_[src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)]
        A[1::2] = np.c_[src[:, 1], src[:, 0], np.zeros(n), np.ones(n)]
        a, b, tx, ty = np.linalg.lstsq(A, dst.reshape(-1), rcond=None)[0]
        return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])
    if model == "affine":
        X = np.c_[src, np.ones(n)]
        M = np.linalg.lstsq(X, dst, rcond=None)[0].T  # 2 x 3
        return np.vstack([M, [0, 0, 1]])
    if model == "homography":
        H, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), 0)
        return H
    raise ValueError(model)


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.c_[pts, np.ones(len(pts))] @ T.T
    return h[:, :2] / h[:, 2:3]


def tre(T: np.ndarray, ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_transform(T, mov) - ref, axis=1)


def leave_one_out_tre(ref: np.ndarray, mov: np.ndarray, model: str) -> np.ndarray:
    errors = []
    for i in range(len(ref)):
        keep = np.arange(len(ref)) != i
        T = fit_transform(mov[keep], ref[keep], model)
        errors.append(tre(T, ref[i:i + 1], mov[i:i + 1])[0])
    return np.array(errors)


# ---------------------------------------------------------------- images
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Green channel (best vessel contrast in fundus images) with CLAHE."""
    green = img_bgr[:, :, 1]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)


def fov_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Circular field of view: pixels that are not part of the black border."""
    mask = (img_bgr.max(axis=2) > 15).astype(np.uint8)
    return cv2.erode(mask, np.ones((15, 15), np.uint8)).astype(bool)


def feature_registration(ref_bgr: np.ndarray, mov_bgr: np.ndarray, scale: float = 0.5,
                         ratio: float = 0.75, seed: int = 0) -> tuple[np.ndarray | None, int, int]:
    """SIFT + ratio test + RANSAC homography. Returns (T moving->reference at full size, matches, inliers)."""
    cv2.setRNGSeed(seed)
    small = [cv2.resize(preprocess(im), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             for im in (ref_bgr, mov_bgr)]
    masks = [cv2.resize(fov_mask(im).astype(np.uint8), None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST) for im in (ref_bgr, mov_bgr)]
    sift = cv2.SIFT_create(nfeatures=5000)
    (k1, d1), (k2, d2) = [sift.detectAndCompute(im, m) for im, m in zip(small, masks)]
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, 0, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 4:
        return None, len(good), 0
    src = np.float32([k2[m.queryIdx].pt for m in good]) / scale
    dst = np.float32([k1[m.trainIdx].pt for m in good]) / scale
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0 / scale,
                                    maxIters=5000, confidence=0.999)
    if H is None:
        return None, len(good), 0
    return H, len(good), int(inliers.sum())


def similarity(ref_bgr: np.ndarray, mov_bgr: np.ndarray, T: np.ndarray | None, scale: float = 0.25,
               bins: int = 32) -> tuple[float, float, float]:
    """NCC and mutual information (nats) of the green channels inside the overlapping field of view.

    Computed at reduced resolution for speed. Returns (ncc, mi, overlap fraction of the reference FOV).
    """
    S = np.diag([scale, scale, 1.0])
    T = np.eye(3) if T is None else T
    h, w = ref_bgr.shape[:2]
    size = (int(w * scale), int(h * scale))
    ref = cv2.resize(ref_bgr, size, interpolation=cv2.INTER_AREA)
    mov_small = cv2.resize(mov_bgr, (int(mov_bgr.shape[1] * scale), int(mov_bgr.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(mov_small, S @ T @ np.linalg.inv(S), size)
    mask = fov_mask(ref) & fov_mask(warped)
    if mask.sum() < 100:
        return float("nan"), float("nan"), float(mask.mean())
    a, b = ref[:, :, 1][mask].astype(float), warped[:, :, 1][mask].astype(float)
    ncc = float(np.corrcoef(a, b)[0, 1])
    joint, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 256], [0, 256]])
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    return ncc, mi, float(mask.sum() / max(fov_mask(ref).sum(), 1))


def overlay(ref_bgr, mov_bgr, T, scale=0.25):
    """Red = reference, green = moving (warped with T). Aligned vessels appear yellow."""
    S = np.diag([scale, scale, 1.0])
    T = np.eye(3) if T is None else T
    size = (int(ref_bgr.shape[1] * scale), int(ref_bgr.shape[0] * scale))
    r = cv2.resize(preprocess(ref_bgr), size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(preprocess(mov_bgr), (int(mov_bgr.shape[1] * scale), int(mov_bgr.shape[0] * scale)),
                   interpolation=cv2.INTER_AREA)
    g = cv2.warpPerspective(m, S @ T @ np.linalg.inv(S), size)
    return np.dstack([r, g, np.zeros_like(r)])


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fire", default="FIRE", help="folder with the extracted FIRE dataset")
    p.add_argument("--category", choices=["S", "P", "A"], default=None)
    p.add_argument("--out", default=str(ROOT))
    p.add_argument("--examples", nargs="*", default=["A01", "S01", "P01"])
    args = p.parse_args()

    pairs = find_pairs(Path(args.fire))
    if args.category:
        pairs = [q for q in pairs if q["category"] == args.category]
    if not pairs:
        raise SystemExit(f"No FIRE pairs found under {args.fire}")
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows, examples = [], {}
    for q in pairs:
        ref_pts, mov_pts = read_control_points(q["points"])
        ref_img, mov_img = cv2.imread(str(q["reference"])), cv2.imread(str(q["moving"]))
        row = {"pair": q["pair"], "category": q["category"], "n_points": len(ref_pts),
               "tre_identity": tre(np.eye(3), ref_pts, mov_pts).mean()}
        for model in MODELS:
            row[f"tre_loo_{model}"] = leave_one_out_tre(ref_pts, mov_pts, model).mean()
        H, n_matches, n_inliers = feature_registration(ref_img, mov_img)
        row.update(matches=n_matches, inliers=n_inliers,
                   tre_sift=tre(H, ref_pts, mov_pts).mean() if H is not None else np.nan)
        row["ncc_before"], row["mi_before"], row["overlap_before"] = similarity(ref_img, mov_img, None)
        row["ncc_after"], row["mi_after"], row["overlap_after"] = (
            similarity(ref_img, mov_img, H) if H is not None else (np.nan, np.nan, np.nan))
        rows.append(row)
        print(f"{q['pair']}: identity {row['tre_identity']:.1f} px | LOO affine {row['tre_loo_affine']:.1f} "
              f"homography {row['tre_loo_homography']:.1f} | SIFT {row['tre_sift']:.1f} px "
              f"({n_inliers}/{n_matches} inliers) | NCC {row['ncc_before']:.3f} -> {row['ncc_after']:.3f}",
              flush=True)
        if q["pair"] in args.examples:
            examples[q["pair"]] = (overlay(ref_img, mov_img, None), overlay(ref_img, mov_img, H), row)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "results" / "per_pair.csv", index=False)
    summarise(df, out)
    plot_examples(examples, out / "figures" / "overlays.png")
    plot_success(df, out / "figures" / "success_curve.png")


def summarise(df: pd.DataFrame, out: Path) -> None:
    cols = ["tre_identity", "tre_loo_similarity", "tre_loo_affine", "tre_loo_homography", "tre_sift"]
    groups = [("All", df)] + [(c, df[df.category == c]) for c in "SPA" if (df.category == c).any()]
    lines = ["| Category | Pairs | " + " | ".join(cols) + " | SIFT failed | SIFT TRE < 10 px |",
             "|---|---|" + "---|" * (len(cols) + 2)]
    for name, g in groups:
        med = [f"{g[c].median():.1f}" for c in cols]
        failed = int(g.tre_sift.isna().sum())
        ok = (g.tre_sift < 10).mean() * 100
        lines.append(f"| {name} | {len(g)} | " + " | ".join(med) + f" | {failed} | {ok:.0f}% |")
    table = "\n".join(lines)
    sim = df[["ncc_before", "ncc_after", "mi_before", "mi_after"]].median().round(3)
    text = ("Median over pairs of the mean TRE per pair (pixels, 2912 x 2912 images).\n\n" + table +
            "\n\nMedian similarity inside the overlapping field of view:\n\n" + sim.to_frame("median").to_markdown() + "\n")
    (out / "results" / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def plot_examples(examples: dict, path: Path) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(8, 4 * len(examples)), squeeze=False)
    for i, (name, (before, after, row)) in enumerate(examples.items()):
        axes[i, 0].imshow(before)
        axes[i, 0].set_title(f"{name}: before (mean TRE {row['tre_identity']:.0f} px)")
        axes[i, 1].imshow(after)
        axes[i, 1].set_title(f"{name}: after SIFT + RANSAC (TRE {row['tre_sift']:.1f} px)")
        for a in axes[i]:
            a.axis("off")
    fig.suptitle("Red = reference, green = registered moving image; aligned vessels look yellow")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_success(df: pd.DataFrame, path: Path) -> None:
    """Fraction of pairs whose mean TRE is below a threshold, as in the FIRE evaluation protocol."""
    thr = np.linspace(0, 25, 251)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in "SPA":
        g = df[df.category == c]
        if len(g):
            err = g.tre_sift.fillna(np.inf).to_numpy()
            ax.plot(thr, [(err < t).mean() for t in thr], label=f"{c} ({len(g)} pairs)")
    ax.set_xlabel("Threshold on mean TRE (px)")
    ax.set_ylabel("Fraction of pairs registered")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("SIFT + RANSAC homography on FIRE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
