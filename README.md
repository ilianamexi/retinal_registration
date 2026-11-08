# Retinal fundus image registration on FIRE

Registration of pairs of retinal fundus photographs: finding the transform that maps one image of an
eye onto another image of the same eye, so that vessels line up. The script evaluates it on the
**FIRE** dataset, which comes with 10 manually marked corresponding points per image pair.

## Method

For every pair (reference `*_1.jpg`, moving `*_2.jpg`):

| Step | What it does |
|---|---|
| **Automatic registration** | SIFT keypoints on the green channel after CLAHE contrast enhancement (vessels are most visible there), ratio-test matching, and a RANSAC homography. The ground-truth points are **not** used. |
| **Error** | Target registration error (TRE): the distance in pixels between each mapped moving point and its reference point, averaged over the 10 control points. |
| **Baseline** | TRE with no registration (identity transform). |
| **Model check** | Similarity, affine and homography transforms fitted to the control points by least squares, scored leave-one-out (fit on 9 points, test on the 10th). This shows how well each motion model can describe a pair without testing on the fitting points. |
| **Image similarity** | Normalised cross-correlation and mutual information of the green channels inside the overlapping field of view, before and after registration. |

Results are reported separately for the three FIRE categories:

- **S:** large overlap.
- **P:** small overlap.
- **A:** large overlap with anatomical changes due to disease progression.

A success curve plots the fraction of pairs whose TRE is below a threshold.

## Run it

1. Download `FIRE.7z` from the dataset page (link below) and extract it to a folder `FIRE/`.
2. Then run:

```bash
pip install -r requirements.txt
python retinal_registration.py --fire FIRE               # all 134 pairs
python retinal_registration.py --fire FIRE --category A  # one category
python -m pytest -q tests
```

**Outputs:**

- `results/per_pair.csv`
- `results/summary.md`: median TRE per category for each method, SIFT failures, and the share of
  pairs below 10 px.
- `figures/overlays.png`: before/after colour overlays.
- `figures/success_curve.png`

**Tests.** The tests use a synthetic fundus-like image warped by a known transform. They check that:

- the least-squares fits recover the transform;
- SIFT + RANSAC registers the pair to under 3 px without seeing the control points;
- correlation improves after registration;
- FIRE file names and point files are parsed correctly.

## Limitations

- **Planar models.** The retina is curved, so a homography is an approximation, especially for pairs
  with small overlap (category P). Methods built for FIRE use 3D eye models.
- **SIFT can fail.** It may not find enough correct matches on low-contrast or small-overlap pairs.
  Such pairs are counted as failures, not skipped.
- **No reported results yet.** The FIRE results have not been run in CI, because the FIRE server did
  not accept connections from GitHub's machines. Run the script locally to produce them.

## Data

C. Hernandez-Matas, X. Zabulis, A. Triantafyllou, P. Anyfanti, S. Douma, A. A. Argyros (2017).
*FIRE: Fundus Image Registration dataset.* Modeling and Artificial Intelligence in Ophthalmology,
1(4), 16–28. DOI: [10.35119/maio.v1i4.42](https://doi.org/10.35119/maio.v1i4.42).
Dataset page: <https://projects.ics.forth.gr/cvrl/fire/>.

134 pairs of 2912 × 2912 px images: 71 in category S, 49 in P and 14 in A.
