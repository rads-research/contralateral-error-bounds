"""Reproduce all analyses, Table 1 and Figures 1-2.

    python reproduce.py [--data data] [--out results]
"""
import argparse
import os
import platform
import subprocess
import urllib.request

import numpy as np
import pandas as pd
import scipy
import trimesh
from scipy.stats import spearmanr

import mirroring as m

OPENSKELETON = {
    "NM_LEXT.mat": "https://zenodo.org/records/23592083/files/NM_LEXT.mat?download=1",
    "Faces.mat": "https://zenodo.org/records/22833546/files/Faces.mat?download=1",
    "Metadata.mat": "https://zenodo.org/records/22833546/files/Metadata.mat?download=1",
}
VSD_REPO = "https://github.com/MCM-Fischer/VSDFullBodyBoneModels.git"


def get_data(root):
    osk, vsd = os.path.join(root, "OpenSkeleton"), os.path.join(root, "VSDFullBodyBoneModels")
    os.makedirs(osk, exist_ok=True)
    for name, url in OPENSKELETON.items():
        path = os.path.join(osk, name)
        if not os.path.exists(path):
            print("Downloading", name)
            urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
    if not os.path.isdir(os.path.join(vsd, "Bones")):
        subprocess.run(["git", "clone", "--depth", "1", VSD_REPO, vsd], check=True)
    return osk, os.path.join(vsd, "Bones")


def summarize(df, dsz, cohort):
    mir = df[df["mode"] == "mirror"]
    L = [cohort, ""]
    L.append("HD95 (mm): median (IQR), max | mean surface distance, median | cases with global fit > 2 mm")
    for (ds, b), g in mir.groupby(["dataset", "bone"]):
        q = np.percentile(g.hd95, [25, 50, 75])
        L.append(f"  {ds} {b}: {q[1]:.1f} ({q[0]:.1f}-{q[2]:.1f}), max {g.hd95.max():.1f} | "
                 f"{g.asd.median():.1f} | {(g.fit_global > 2).sum()}")
    L.append(f"  Self-reconstruction controls, max HD95: {df[df['mode'] == 'self'].hd95.max():.2f} mm")

    L += ["", "Spearman correlation with HD95: near-defect fit, global fit"]
    for (ds, b), g in mir.groupby(["dataset", "bone"]):
        rl, pl = spearmanr(g.fit_near_defect, g.hd95); rg, pg = spearmanr(g.fit_global, g.hd95)
        L.append(f"  {ds} {b}: {rl:.2f} (P = {pl:.2g}), {rg:.2f} (P = {pg:.2g})")

    table = m.evaluate_bounds(df)
    L += ["", "95% error bounds: internal coverage / width | external covered [95% CI] / width (range) | "
              "difference from fixed (95% CI)"]
    for _, r in table.iterrows():
        L.append(f"  {r.bone} {r.bound}: {100 * r.internal_coverage:.1f}% / {r.internal_mean_width:.1f} mm | "
                 f"{r.ext_covered}/{r.n_ext} [{100 * r.ext_ci_lo:.0f}-{100 * r.ext_ci_hi:.0f}%] / "
                 f"{r.ext_mean_width:.1f} mm ({r.ext_width_min:.1f}-{r.ext_width_max:.1f}) | "
                 f"{r.diff_vs_fixed:+.2f} ({r.diff_ci_lo:.2f} to {r.diff_ci_hi:.2f})")

    L += ["", "Regression bound at the 10th, 50th and 90th percentiles of near-defect fit"]
    for bone in ("Femur", "Tibia"):
        d = mir[(mir.dataset == "OpenSkeleton") & (mir.bone == bone)].reset_index(drop=True)
        e = mir[(mir.dataset == "VSD") & (mir.bone == bone)].reset_index(drop=True)
        x = np.maximum(d.fit_near_defect.values, m.FIT_FLOOR); qx = np.percentile(x, [10, 50, 90])
        qb = m.cvplus_upper(x[:, None], d.hd95.values, qx[:, None])
        L.append(f"  {bone}: fit {np.round(qx, 2)} mm -> bound {np.round(qb, 1)} mm")
        b = m.bounds(d, e); y = e.hd95.values
        for i in np.where(y > b["Regression"])[0]:
            L.append(f"    external case above regression bound by {y[i] - b['Regression'][i]:.2f} mm "
                     f"(near-defect fit {e.fit_near_defect[i]:.2f} mm)")
        miss = y > b["Fixed"]
        L.append(f"    above fixed bound: {miss.sum()}, of which within regression bound: "
                 f"{(miss & (y <= b['Regression'])).sum()}")

    ref = m.compare_refinement(df)
    L += ["", "Near-defect refinement (refined minus standard HD95)", ref.round(3).to_string(index=False),
          f"  Self-reconstruction controls after refinement, max HD95: "
          f"{df[df['mode'] == 'self'].hd95_refined.max():.2f} mm"]

    size = (dsz.groupby(["dataset", "bone", "defect_frac"]).hd95
            .agg(n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75))
            .round(2).reset_index())
    a = dsz[np.isclose(dsz.defect_frac, 1/3, atol=1e-3)].merge(mir, on=["dataset", "subject", "bone"])
    diff = (a.hd95_x - a.hd95_y).abs()
    L += ["", "Defect size: HD95 (mm)", size.to_string(index=False),
          f"  Sampling density at one-third (20,000 vs 15,000 points): median |difference| "
          f"{diff.median():.2f} mm, max {diff.max():.2f} mm"]
    L += ["", f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
              f"trimesh {trimesh.__version__}"]
    return "\n".join(L), table, ref, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    osk, vsd_dir = get_data(a.data)

    print("Loading data")
    vsd = m.load_vsd(vsd_dir)
    meta = m.load_metadata(os.path.join(osk, "Metadata.mat"))
    osk_subjects = m.load_openskeleton(os.path.join(osk, "NM_LEXT.mat"), os.path.join(osk, "Faces.mat"), meta)
    for name, s in (("VSD", vsd), ("OpenSkeleton", osk_subjects)):
        m.check_lengths(s, name)
    print(f"  Development (OpenSkeleton): {len(osk_subjects)}; external (VSD): {len(vsd)}")
    print("Handedness check")
    reflect = {"VSD": m.check_chirality(vsd), "OpenSkeleton": m.check_chirality(osk_subjects)}

    main_csv, size_csv = os.path.join(a.out, "results_main.csv"), os.path.join(a.out, "results_defect_size.csv")
    for name, s in (("VSD", vsd), ("OpenSkeleton", osk_subjects)):
        print(f"Main analysis: {name}")
        m.run(main_csv, m.MAIN_FIELDS, m.main_jobs(name, s, reflect[name]))
    for name, s in (("VSD", vsd), ("OpenSkeleton", osk_subjects)):
        print(f"Defect size: {name}")
        m.run(size_csv, m.DEFECT_FIELDS, m.defect_jobs(name, s, reflect[name]))

    df = pd.read_csv(main_csv, dtype={"subject": str})
    dsz = pd.read_csv(size_csv, dtype={"subject": str})
    cohort = "Development: " + m.cohort_summary(meta, [s for s, _ in osk_subjects])
    text, table, ref, size = summarize(df, dsz, cohort)
    table.to_csv(os.path.join(a.out, "table1_bounds.csv"), index=False)
    ref.to_csv(os.path.join(a.out, "refinement.csv"), index=False)
    size.to_csv(os.path.join(a.out, "defect_size.csv"), index=False)
    open(os.path.join(a.out, "summary.txt"), "w").write(text)
    print(text)

    e = df[(df.dataset == "VSD") & (df.bone == "Femur") & (df["mode"] == "mirror")]
    example = e.iloc[(e.hd95 - e.hd95.median()).abs().argsort().iloc[0]].subject
    m.figure1(dict(vsd)[example], example, os.path.join(a.out, "figure1"))
    m.figure2(table, os.path.join(a.out, "figure2"))


if __name__ == "__main__":
    main()
