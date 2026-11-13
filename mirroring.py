"""
Contralateral bone reconstruction with case-specific 95% error bounds.

Reconstruction
  The left femur or tibia is the target and the right bone, reflected across the sagittal plane,
  is the template. The long axis of the target is its first principal axis, oriented cranially,
  and the distal fraction of bone length is removed. The mirrored template is aligned to the
  retained surface by rigid ICP (no scaling) from 12 initialisations at 30-degree steps about
  the long axis; the lowest residual is kept.

Measures
  Global fit        RMS point-to-surface distance over the retained bone.
  Near-defect fit   RMS over the 15% of bone length immediately proximal to the defect.
  Error             Symmetric 95th-percentile surface distance (HD95, mm) within the missing segment.
  Refinement        ICP repeated on the retained third of length nearest the defect.

Error bounds (95%), derived in the development cohort
  Fixed            k-th smallest HD95, k = ceil((n + 1) x 0.95).
  Residual-scaled  Global fit x k-th smallest ratio HD95 / global fit.
  Regression       Linear model of HD95 on near-defect fit, calibrated by 10-fold CV+.
  Fit values are floored at 0.25 mm.
"""
import csv
import glob
import os
import zlib

import numpy as np
import scipy.io
import trimesh
from scipy.spatial import cKDTree

SEED = 2026
ALPHA = 0.05
N_SAMPLE = 15000          # points per surface for HD95
ICP_SAMPLE = 8000
ICP_TARGET_SAMPLE = 60000
ICP_MAX_ITER = 200
ICP_TOL = 1e-6
N_STARTS = 12
NEAR_DEFECT = 0.15        # band of length above the cut used for near-defect fit
REFINE_BAND = 1 / 3       # band of length above the cut used for refinement
FIT_FLOOR = 0.25          # mm
BONES = ("Femur_L", "Femur_R", "Tibia_L", "Tibia_R")

VSD_EXCLUDE = {"z013": "tibial intraosseous access",
               "z050": "knee arthroplasty",
               "z063": "hip and knee arthroplasty",
               "z066": "extensive metal artifact"}


# ----------------------------------------------------------------------------- geometry
def mirror_x(mesh):
    v = mesh.vertices.copy(); v[:, 0] *= -1
    return trimesh.Trimesh(v, mesh.faces[:, ::-1], process=False)


def principal_axis(points, toward):
    _, _, vt = np.linalg.svd(points - points.mean(0), full_matrices=False)
    return vt[0] if np.dot(vt[0], toward) > 0 else -vt[0]


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def axis_angle(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else axis_angle(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def to_h(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def submesh(mesh, vmask):
    return mesh.submesh([np.where(vmask[mesh.faces].all(1))[0]], append=True)


def surf_dist(points, mesh):
    return np.abs(trimesh.proximity.closest_point(mesh, points)[1])


def sample(mesh, n, rng):
    return trimesh.sample.sample_surface_even(mesh, n, seed=int(rng.integers(2**31 - 1)))[0]


def case_rng(*key):
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


def cranial_direction(bones, side="L"):
    d = bones[f"Femur_{side}"].vertices.mean(0) - bones[f"Tibia_{side}"].vertices.mean(0)
    return d / np.linalg.norm(d)


def icp(src, tgt_pts, tree, T0):
    T = T0.copy(); prev = np.inf
    for _ in range(ICP_MAX_ITER):
        P = src @ T[:3, :3].T + T[:3, 3]
        d, idx = tree.query(P)
        T = to_h(*kabsch(P, tgt_pts[idx])) @ T
        mse = float(np.mean(d ** 2))
        if abs(prev - mse) < ICP_TOL:
            break
        prev = mse
    P = src @ T[:3, :3].T + T[:3, 3]
    return T, float(np.sqrt(np.mean(tree.query(P)[0] ** 2)))


# ----------------------------------------------------------------------------- reconstruction
def _segment_error(defect, recon, target, ax, cut, n, rng):
    recon_def = submesh(recon, recon.vertices @ ax < cut)
    pa, pb = sample(defect, n, rng), sample(recon_def, n, rng)
    da, db = surf_dist(pa, recon), surf_dist(pb, target)
    return float(max(np.percentile(da, 95), np.percentile(db, 95))), float((da.mean() + db.mean()) / 2)


def reconstruct(target, template, cranial, rng, defect_frac=1/3, n_sample=N_SAMPLE,
                refine=True, return_meshes=False):
    """Reconstruct the distal `defect_frac` of `target` from `template` (already mirrored)."""
    ax = principal_axis(target.vertices, cranial)
    s = target.vertices @ ax
    L = s.max() - s.min()
    cut = s.min() + defect_frac * L
    retained, defect = submesh(target, s >= cut), submesh(target, s < cut)

    axC = principal_axis(template.vertices, cranial)
    sC = template.vertices @ axC
    cutC = sC.min() + defect_frac * (sC.max() - sC.min())
    src = sample(retained, ICP_SAMPLE, rng)
    tgt_pts = sample(template, ICP_TARGET_SAMPLE, rng)
    tree = cKDTree(tgt_pts)
    cR, cC = retained.vertices.mean(0), template.vertices[sC >= cutC].mean(0)
    best = (None, np.inf)
    for ang in np.linspace(0, 2 * np.pi, N_STARTS, endpoint=False):
        R0 = axis_angle(axC, ang) @ rot_a_to_b(ax, axC)
        T, r = icp(src, tgt_pts, tree, to_h(R0, cC - R0 @ cR))
        if r < best[1]:
            best = (T, r)
    T = best[0]

    s_src = src @ ax
    d_src = surf_dist(src @ T[:3, :3].T + T[:3, 3], template)
    band = s_src < cut + NEAR_DEFECT * L
    out = dict(length=float(L),
               fit_global=float(np.sqrt(np.mean(d_src ** 2))),
               fit_near_defect=float(np.sqrt(np.mean(d_src[band] ** 2))))
    recon = template.copy(); recon.apply_transform(np.linalg.inv(T))
    out["hd95"], out["asd"] = _segment_error(defect, recon, target, ax, cut, n_sample, rng)

    if refine:
        T2, _ = icp(src[s_src < cut + REFINE_BAND * L], tgt_pts, tree, T)
        recon2 = template.copy(); recon2.apply_transform(np.linalg.inv(T2))
        out["hd95_refined"], out["asd_refined"] = _segment_error(defect, recon2, target, ax, cut, n_sample, rng)
    if return_meshes:
        out.update(target=target, retained=retained, defect=defect, recon=recon,
                   defect_vertex_error=surf_dist(defect.vertices, recon))
    return out


def check_chirality(subjects, n_check=5):
    """True if right bones must be reflected to match left bones (registration residual test)."""
    votes = []
    for sid, b in subjects[:n_check]:
        L, R = b["Femur_L"], b["Femur_R"]
        cran = cranial_direction(b)
        res = []
        for tmpl in (R, mirror_x(R)):
            rng = case_rng("chir", sid)
            src = sample(L, ICP_SAMPLE, rng)
            tgt_pts = sample(tmpl, ICP_TARGET_SAMPLE, rng); tree = cKDTree(tgt_pts)
            a, aC = principal_axis(L.vertices, cran), principal_axis(tmpl.vertices, cran)
            best = min((icp(src, tgt_pts, tree, to_h(R0, tmpl.vertices.mean(0) - R0 @ L.vertices.mean(0)))
                        for R0 in (axis_angle(aC, g) @ rot_a_to_b(a, aC)
                                   for g in np.linspace(0, 2 * np.pi, N_STARTS, endpoint=False))),
                       key=lambda x: x[1])[0]
            P = src @ best[:3, :3].T + best[:3, 3]
            res.append(float(np.sqrt(np.mean(surf_dist(P, tmpl) ** 2))))
        votes.append(res[1] < res[0])
        print(f"  {sid}: residual as stored {res[0]:.2f} mm, reflected {res[1]:.2f} mm")
    return sum(votes) > len(votes) / 2


# ----------------------------------------------------------------------------- data
def _load_mat(path):
    try:
        return scipy.io.loadmat(path, simplify_cells=True)
    except (NotImplementedError, ValueError):
        import mat73
        return mat73.loadmat(path)


def load_vsd(bones_dir):
    subjects = []
    for f in sorted(glob.glob(os.path.join(bones_dir, "*.mat"))):
        sid = os.path.basename(f)[:-4]
        if "Phantom" in sid or sid in VSD_EXCLUDE:
            continue
        B = scipy.io.loadmat(f)["B"][0]
        bones = {}
        for b in B:
            if b[0].size and b[2].size and str(b[0].ravel()[0]) in BONES:
                v = np.asarray(b[2][0][0][0], float)
                fc = np.asarray(b[2][0][0][1], np.int64) - 1
                bones[str(b[0].ravel()[0])] = trimesh.Trimesh(v, fc, process=False)
        subjects.append((sid, bones))
    return subjects


def load_openskeleton(nm_lext_path, faces_path, metadata):
    """Decedents documented in the dataset metadata, in identifier order."""
    data = _load_mat(nm_lext_path)["NM_LEXT"]
    faces = _load_mat(faces_path)["Faces"]
    subjects = []
    for sid in sorted(data):
        if sid not in metadata:
            continue
        bones = {}
        for k in BONES:
            f = np.asarray(faces["F" + k], np.int64)
            bones[k] = trimesh.Trimesh(np.asarray(data[sid][k]["Registered"], float),
                                       f - 1 if f.min() == 1 else f, process=False)
        subjects.append((sid, bones))
    return subjects


def load_metadata(path):
    M = {k: v for k, v in _load_mat(path).items() if not str(k).startswith("__")}
    return list(M.values())[0] if len(M) == 1 else M


def cohort_summary(meta, ids):
    import collections
    ages = np.array([meta[i]["age_years"] for i in ids], float)
    sex = collections.Counter(meta[i]["sex_code"] for i in ids)
    return (f"n = {len(ids)}; sex {dict(sex)}; age median {np.nanmedian(ages):.1f} "
            f"(range {np.nanmin(ages):.0f}-{np.nanmax(ages):.0f})")


def check_lengths(subjects, name):
    for sid, b in subjects:
        for k in BONES:
            m = b[k]
            length = np.ptp(m.vertices @ principal_axis(m.vertices, np.array([0, 0, 1.0])))
            lo, hi = (330, 600) if k.startswith("Femur") else (270, 520)
            if not lo <= length <= hi:
                raise ValueError(f"{name} {sid} {k}: implausible length {length:.0f} mm")


# ----------------------------------------------------------------------------- batch runs
def run(rows_csv, fields, jobs):
    """Run jobs [(key_dict, fn)] and append one row each to rows_csv; rows already present are skipped."""
    keys = list(jobs[0][0]) if jobs else []
    done, good = set(), []
    if os.path.exists(rows_csv):
        for r in csv.DictReader(open(rows_csv)):
            if all(r.get(k) not in (None, "") for k in fields):
                good.append(r); done.add(tuple(r[k] for k in keys))
        with open(rows_csv + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(good)
        os.replace(rows_csv + ".tmp", rows_csv)
    todo = [(k, fn) for k, fn in jobs if tuple(str(k[x]) for x in keys) not in done]
    print(f"  {len(jobs)} reconstructions, {len(todo)} to run")
    new = not os.path.exists(rows_csv)
    with open(rows_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, (k, fn) in enumerate(todo, 1):
            r = fn()
            w.writerow({**k, **r}); fh.flush()
            print(f"  [{i}/{len(todo)}] {' '.join(map(str, k.values()))}: HD95 {r['hd95']:.2f} mm", flush=True)


MAIN_FIELDS = ["dataset", "subject", "bone", "mode", "length", "fit_global", "fit_near_defect",
               "hd95", "asd", "hd95_refined", "asd_refined"]
DEFECT_FIELDS = ["dataset", "subject", "bone", "defect_frac", "length", "fit_global", "hd95", "asd"]


def main_jobs(dataset, subjects, reflect, n_controls=3):
    """Distal-third reconstruction of every left femur and tibia, plus self-reconstruction controls."""
    jobs = []
    for i, (sid, b) in enumerate(subjects):
        for bone in ("Femur", "Tibia"):
            T = b[f"{bone}_L"]
            for mode in ("mirror", "self") if i < n_controls else ("mirror",):
                tmpl = T.copy() if mode == "self" else (mirror_x(b[f"{bone}_R"]) if reflect else b[f"{bone}_R"])
                jobs.append((dict(dataset=dataset, subject=sid, bone=bone, mode=mode),
                             lambda T=T, tmpl=tmpl, b=b, k=("v2", dataset, sid, bone, mode):
                             reconstruct(T, tmpl, cranial_direction(b), case_rng(*k))))
    jobs.sort(key=lambda j: j[0]["mode"] == "self")
    return jobs


def defect_jobs(dataset, subjects, reflect, fracs=(0.25, 1/3, 0.5)):
    jobs = []
    for sid, b in subjects:
        for bone in ("Femur", "Tibia"):
            for f in fracs:
                T, R = b[f"{bone}_L"], b[f"{bone}_R"]
                tmpl = mirror_x(R) if reflect else R
                jobs.append((dict(dataset=dataset, subject=sid, bone=bone, defect_frac=f"{f:.4f}"),
                             lambda T=T, tmpl=tmpl, b=b, f=f, k=(dataset, sid, bone, "L", "mirror", f"{f:.4f}"):
                             reconstruct(T, tmpl, cranial_direction(b), case_rng(*k), defect_frac=f,
                                         n_sample=20000, refine=False)))
    return jobs


# ----------------------------------------------------------------------------- statistics
def conformal_k(n, alpha=ALPHA):
    return int(np.ceil((n + 1) * (1 - alpha)))


def conformal_quantile(x, alpha=ALPHA):
    x = np.sort(np.asarray(x, float)); k = conformal_k(len(x), alpha)
    return np.inf if k > len(x) else float(x[k - 1])


def clopper_pearson(x, n, level=0.95):
    from scipy.stats import beta
    a = (1 - level) / 2
    return (0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1)),
            1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x)))


def cvplus_upper(Xd, yd, Xe, alpha=ALPHA, folds=10, seed=SEED):
    """One-sided CV+ upper bound (Barber et al., 2021) for linear regression of y on X."""
    n = len(yd)
    fold = np.random.default_rng(seed).permutation(np.arange(n) % folds)
    A = lambda X: np.column_stack([np.ones(len(X)), X])
    scores = np.empty((len(Xe), n))
    for k in range(folds):
        tr, te = fold != k, fold == k
        beta, *_ = np.linalg.lstsq(A(Xd[tr]), yd[tr], rcond=None)
        scores[:, te] = (A(Xe) @ beta)[:, None] + (yd[te] - A(Xd[te]) @ beta)[None, :]
    kq = conformal_k(n, alpha)
    return np.full(len(Xe), np.inf) if kq > n else np.sort(scores, 1)[:, kq - 1]


def bounds(dev, new):
    """95% upper bounds for the cases in `new`, derived from development cases `dev`."""
    y = dev.hd95.values
    g_d, g_n = np.maximum(dev.fit_global.values, FIT_FLOOR), np.maximum(new.fit_global.values, FIT_FLOOR)
    l_d, l_n = np.maximum(dev.fit_near_defect.values, FIT_FLOOR), np.maximum(new.fit_near_defect.values, FIT_FLOOR)
    return {
        "Fixed": np.full(len(new), conformal_quantile(y)),
        "Residual-scaled": conformal_quantile(y / g_d) * g_n,
        "Regression": cvplus_upper(l_d[:, None], y, l_n[:, None]),
        "Regression, global fit": cvplus_upper(g_d[:, None], y, g_n[:, None]),
    }


def evaluate_bounds(df, dev_name="OpenSkeleton", ext_name="VSD", n_boot=2000):
    """Internal 10-fold cross-validation and external validation of each bound."""
    import pandas as pd
    from scipy.stats import binom
    rows = []
    for bone in ("Femur", "Tibia"):
        rng = np.random.default_rng(SEED)
        sel = lambda ds: df[(df.dataset == ds) & (df.bone == bone) & (df["mode"] == "mirror")].reset_index(drop=True)
        d, e = sel(dev_name), sel(ext_name)
        ext = bounds(d, e)
        fold = np.random.default_rng(SEED + 1).permutation(np.arange(len(d)) % 10)
        cov, wid = {m: [] for m in ext}, {m: [] for m in ext}
        for k in range(10):
            for m, b in bounds(d[fold != k].reset_index(drop=True), d[fold == k].reset_index(drop=True)).items():
                cov[m] += list(d[fold == k].hd95.values <= b); wid[m] += list(b)
        y, n = e.hd95.values, len(e)
        for m, b in ext.items():
            x = int(np.sum(y <= b)); lo, hi = clopper_pearson(x, n)
            diff = b - ext["Fixed"]
            boot = [diff[i].mean() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
            rows.append(dict(bone=bone, bound=m, n_dev=len(d), n_ext=n,
                             internal_coverage=float(np.mean(cov[m])), internal_mean_width=float(np.mean(wid[m])),
                             ext_covered=x, ext_coverage=x / n, ext_ci_lo=lo, ext_ci_hi=hi,
                             transports=bool(binom.cdf(x, n, 1 - ALPHA) > 0.05),
                             ext_mean_width=float(b.mean()), ext_width_min=float(b.min()), ext_width_max=float(b.max()),
                             diff_vs_fixed=float(diff.mean()),
                             diff_ci_lo=float(np.percentile(boot, 2.5)), diff_ci_hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def compare_refinement(df):
    import pandas as pd
    from scipy.stats import wilcoxon
    rows, rng = [], np.random.default_rng(SEED)
    for (ds, bone), g in df[df["mode"] == "mirror"].groupby(["dataset", "bone"]):
        dlt = (g.hd95_refined - g.hd95).values
        boot = [np.median(dlt[rng.integers(0, len(dlt), len(dlt))]) for _ in range(2000)]
        rows.append(dict(dataset=ds, bone=bone, n=len(g), median_hd95=float(np.median(g.hd95)),
                         median_hd95_refined=float(np.median(g.hd95_refined)),
                         median_paired_diff=float(np.median(dlt)),
                         ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                         improved=int(np.sum(dlt < 0)), wilcoxon_p=float(wilcoxon(dlt).pvalue)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
NAVY, TAN, GREY, INK, MUTED, BONE = "#2E4A7D", "#C49A5B", "#9A9A9A", "#222222", "#5F5F5F", "#E4DFD5"
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
         "font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.edgecolor": "#BFBFBF", "pdf.fonttype": 42}


def _limits(ax, ref):
    lo, hi = ref.min(0), ref.max(0)
    pad = 0.06 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo); ax.view_init(elev=5, azim=-90); ax.set_axis_off(); ax.patch.set_alpha(0)


def _render(parts, ref, height_in=6.0, dpi=600):
    """Render meshes [(mesh_vertices, faces, colour or per-face values)] to a cropped RGBA image."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(height_in / 3, height_in), dpi=dpi)
    ax = fig.add_axes([-0.5, 0, 2, 1], projection="3d")
    for v, f, c in parts:
        if isinstance(c, str):
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color=c, linewidth=0, shade=True)
        else:
            sf = ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, cmap="cividis", linewidth=0, shade=False)
            sf.set_array(c); sf.set_clim(0, 6)
    _limits(ax, ref)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig)
    img[..., 3] = np.where(img[..., :3].min(-1) > 250, 0, img[..., 3])   # white background -> transparent
    return img


def _crop(imgs, pad=6):
    """Crop a set of images to a shared vertical extent and their own horizontal extent."""
    rows = [np.where(i[..., 3].any(1))[0] for i in imgs]
    top, bot = min(r[0] for r in rows) - pad, max(r[-1] for r in rows) + pad
    out = []
    for i in imgs:
        c = np.where(i[..., 3].any(0))[0]
        out.append(i[max(top, 0):bot, max(c[0] - pad, 0):c[-1] + pad])
    return out


def figure1(bones, sid, path):
    """Intact femur, distal-third defect, mirrored reconstruction and error map."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    T, Cm, cran = bones["Femur_L"], mirror_x(bones["Femur_R"]), cranial_direction(bones)
    r = reconstruct(T, Cm, cran, case_rng("fig1", sid), refine=False, return_meshes=True)
    R = rot_a_to_b(principal_axis(T.vertices, cran), np.array([0, 0, 1.0])); c0 = T.vertices.mean(0)
    tv = lambda m: (m.vertices - c0) @ R.T
    allv = tv(r["target"])
    bones3 = _crop([_render([(tv(m), m.faces, BONE)], allv) for m in (r["target"], r["retained"], r["recon"])])
    d = r["defect"]
    err = _crop([_render([(tv(d), d.faces, r["defect_vertex_error"][d.faces].mean(1))], tv(d))])[0]

    fig = plt.figure(figsize=(7.2, 4.0))
    titles = ["A  Intact\nfemur", "B  Distal third\nremoved", "C  Mirrored\nreconstruction", "D  Error over the\nmissing segment"]
    H = 0.8
    for i, im in enumerate(bones3):
        w = H * im.shape[1] / im.shape[0] * 4.0 / 7.2
        cx = 0.09 + 0.17 * i
        fig.add_axes([cx - w / 2, 0.03, w, H]).imshow(im); fig.axes[-1].set_axis_off()
        fig.text(cx, 0.97, titles[i], ha="center", va="top", fontsize=10)
    w = H * err.shape[1] / err.shape[0] * 4.0 / 7.2
    cx = 0.72
    fig.add_axes([cx - w / 2, 0.03, w, H]).imshow(err); fig.axes[-1].set_axis_off()
    fig.text(cx, 0.97, titles[3], ha="center", va="top", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 6), cmap="cividis")
    cb = fig.colorbar(sm, cax=fig.add_axes([0.93, 0.2, 0.015, 0.5]))
    cb.set_label("Surface distance (mm)", fontsize=9.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)


def figure2(table, path):
    """External coverage (95% CI) against mean width for the three bounds."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    t = table[table.bound.isin(["Fixed", "Residual-scaled", "Regression"])]
    style = {"Fixed": (GREY, "o"), "Residual-scaled": (TAN, "s"), "Regression": (NAVY, "D")}
    off = {("Femur", "Fixed"): (8, -4), ("Femur", "Residual-scaled"): (-8, -14), ("Femur", "Regression"): (-8, 4),
           ("Tibia", "Fixed"): (8, -4), ("Tibia", "Residual-scaled"): (8, -4), ("Tibia", "Regression"): (-8, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, constrained_layout=True)
    for ax, bone, xl in zip(axes, ("Femur", "Tibia"), ((6, 15), (4, 7))):
        ax.axhline(95, color="#BFBFBF", lw=1, ls=(0, (4, 3)), zorder=0)
        for _, r in t[t.bone == bone].iterrows():
            c, mk = style[r.bound]; x, y = r.ext_mean_width, 100 * r.ext_coverage
            ax.plot([x, x], [100 * r.ext_ci_lo, 100 * r.ext_ci_hi], color=c, lw=1.6, zorder=2)
            ax.scatter(x, y, s=52, marker=mk, color=c, edgecolor="white", linewidth=1.3, zorder=3)
            dx, dy = off[(bone, r.bound)]
            ax.annotate(f"{r.bound}\n{r.ext_covered}/{r.n_ext}, {x:.1f} mm", (x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8.5, ha="left" if dx > 0 else "right", va="center")
        ax.set_title(bone, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean bound width (mm)"); ax.set_xlim(*xl)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(65, 103); axes[0].set_ylabel("External coverage (%)")
    axes[0].text(14.9, 93.6, "95% target", ha="right", fontsize=8.5, color=MUTED)
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight"); fig.savefig(path + ".pdf", bbox_inches="tight")
