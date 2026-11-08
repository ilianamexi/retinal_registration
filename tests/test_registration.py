import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retinal_registration as rr  # noqa: E402

TRUE_T = np.array([[0.98, -0.05, 80.0], [0.05, 0.98, -60.0], [1e-6, -2e-6, 1.0]])  # moving -> reference


def synthetic_fundus(size=1000, seed=0):
    """Bright disc with dark, randomly curved 'vessels' and a black border, roughly like a fundus image."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.47), 170, -1)
    for _ in range(60):
        pts = np.cumsum(rng.normal(0, 25, (12, 2)), axis=0) + rng.uniform(150, size - 150, 2)
        cv2.polylines(img, [pts.astype(np.int32)], False, int(rng.integers(40, 110)), int(rng.integers(2, 7)))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.merge([img // 3, img, img // 2])


def make_pair():
    ref = synthetic_fundus()
    mov = cv2.warpPerspective(ref, np.linalg.inv(TRUE_T), ref.shape[1::-1])  # so that TRUE_T maps mov -> ref
    rng = np.random.default_rng(1)
    mov_pts = rng.uniform(300, 700, (10, 2))
    ref_pts = rr.apply_transform(TRUE_T, mov_pts)
    return ref, mov, ref_pts, mov_pts


def test_fits_recover_exact_transform():
    _, _, ref_pts, mov_pts = make_pair()
    H = rr.fit_transform(mov_pts, ref_pts, "homography")
    assert rr.tre(H, ref_pts, mov_pts).max() < 1e-3
    assert rr.leave_one_out_tre(ref_pts, mov_pts, "homography").max() < 1e-2
    # an affine model cannot represent the small perspective part exactly, but comes close
    assert rr.leave_one_out_tre(ref_pts, mov_pts, "affine").mean() < 3


def test_similarity_fit_on_pure_rotation():
    rng = np.random.default_rng(2)
    src = rng.uniform(0, 100, (6, 2))
    a = np.radians(10)
    T = np.array([[1.1 * np.cos(a), -1.1 * np.sin(a), 5], [1.1 * np.sin(a), 1.1 * np.cos(a), -3], [0, 0, 1]])
    assert np.allclose(rr.fit_transform(src, rr.apply_transform(T, src), "similarity"), T)


def test_sift_registration_uses_only_images():
    ref, mov, ref_pts, mov_pts = make_pair()
    H, matches, inliers = rr.feature_registration(ref, mov)
    assert H is not None and inliers >= 20
    assert rr.tre(H, ref_pts, mov_pts).mean() < 3.0
    assert rr.tre(np.eye(3), ref_pts, mov_pts).mean() > 20
    ncc_before, _, _ = rr.similarity(ref, mov, None)
    ncc_after, _, _ = rr.similarity(ref, mov, H)
    assert ncc_after > ncc_before


def test_find_pairs_and_read_points(tmp_path):
    ref, mov, ref_pts, mov_pts = make_pair()
    (tmp_path / "Images").mkdir()
    (tmp_path / "Ground Truth").mkdir()
    cv2.imwrite(str(tmp_path / "Images" / "A01_1.jpg"), ref)
    cv2.imwrite(str(tmp_path / "Images" / "A01_2.jpg"), mov)
    np.savetxt(tmp_path / "Ground Truth" / "control_points_A01_1_2.txt", np.c_[ref_pts, mov_pts])
    pairs = rr.find_pairs(tmp_path)
    assert [(p["pair"], p["category"]) for p in pairs] == [("A01", "A")]
    r, m = rr.read_control_points(pairs[0]["points"])
    assert np.allclose(r, ref_pts) and np.allclose(m, mov_pts)
