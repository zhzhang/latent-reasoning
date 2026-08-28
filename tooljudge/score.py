"""Agreement plus the alt-test, per condition. Answers RQ1, RQ2, RQ4."""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def cohen_kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def krippendorff_binary(M):
    n, k = M.shape
    # disagreeing unordered pairs / total unordered pairs. No extra halving:
    # De below is also a proportion of unordered pairs, so the two must match.
    Do = (M.sum(1) * (k - M.sum(1))).sum() / (n * k * (k - 1) / 2)
    p = M.mean()
    De = 2 * p * (1 - p) * (n * k) / (n * k - 1)
    return 1 - Do / De if De > 0 else np.nan


def alt_test(judge, R, eps=0.15, alpha=0.05):
    """Calderon et al. 2025. Leave one annotator out, compare judge against the
    rest to that annotator against the rest, with an epsilon advantage penalty."""
    n, k = R.shape
    pvals, diffs = [], []
    for j in range(k):
        others = [i for i in range(k) if i != j]
        f_llm = np.array([(judge[i] == R[i, others]).mean() for i in range(n)])
        f_hum = np.array([(R[i, j] == R[i, others]).mean() for i in range(n)])
        d = f_llm - (1 - eps) * f_hum
        diffs.append(d.mean())
        if np.allclose(d, 0):
            pvals.append(1.0)
        else:
            pvals.append(wilcoxon(d, alternative="greater", zero_method="zsplit").pvalue)
    rej, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_by")
    return {"omega": float(rej.mean()), "rho": float(np.mean([d + (1 - eps) for d in diffs]).clip(0, 1)),
            "passes": bool(rej.mean() >= 0.5), "per_annotator_p": [round(p, 4) for p in pvals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="one or more JSONL outputs from collect_rq4.py")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--labels-csv",
                    help="compute the human baseline over every labelled row rather "
                         "than only the rows a judge happened to parse")
    ap.add_argument("--drop-self-preference", action="store_true",
                    help="exclude shots whose answering model is also the judge")
    args = ap.parse_args()

    rows = [json.loads(l) for f in args.results for l in open(f) if l.strip()]
    df = pd.DataFrame(rows)
    if "error" in df:
        df = df[df["error"].isna()]

    # Parse rate belongs per condition, not pooled. Unparsed rows are dropped below,
    # and truncation clusters on the tool arms, so a pooled figure hides which arm
    # lost what. An arm far below the others is a failed run, not a strict judge.
    print("PARSE RATE BY CONDITION")
    for cond, sub in df.groupby("condition"):
        print(f"  {cond:12s} n={len(sub):5d}  parsed={int(sub.parsed.sum()):5d} "
              f"({sub.parsed.mean():.3f})")

    df = df[df["parsed"] == True].copy()
    df["judge_pass"] = df["pass"].astype(bool)

    if args.drop_self_preference and "self_preference" in df:
        n0 = len(df); df = df[~df.self_preference].copy()
        print(f"\ndropped {n0 - len(df)} self-preference rows (judge model under test)")

    full = df[df.ratings.apply(len) == 3].copy()

    # The baseline must not move with judge parse rate. Prefer the full label file.
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv)
        lab["ratings"] = lab["ratings"].apply(json.loads)
        M = np.array(lab[lab.ratings.apply(len) == 3].ratings.tolist()).astype(int)
        src = f"all labelled rows in {args.labels_csv}"
    else:
        M = np.array(full.drop_duplicates(["question_id", "model_label", "shot_index"])
                     .ratings.tolist()).astype(int) if "question_id" in full else \
            np.array(full.ratings.tolist()).astype(int)
        src = "judged rows (pass --labels-csv for the unbiased figure)"

    k = M.shape[1]
    loo = ((M.sum(1) * (M.sum(1) - 1) + (k - M.sum(1)) * (k - M.sum(1) - 1))
           / (k * (k - 1))).mean()
    print(f"\nHUMAN BASELINE on {len(M)} items, from {src}")
    print("  pairwise kappa:", [round(cohen_kappa(M[:, i], M[:, j]), 3)
                                for i, j in combinations(range(k), 2)])
    print("  krippendorff alpha:", round(krippendorff_binary(M), 3))
    print("  leave-one-out agreement:", round(float(loo), 3))
    print("  annotators split on:", round(float(((M.sum(1) > 0) & (M.sum(1) < k)).mean()), 3))

    print(f"\nJUDGE RESULTS (eps={args.eps})")
    for cond, sub in full.groupby("condition"):
        Ms = np.array(sub.ratings.tolist()).astype(int)
        jv = sub.judge_pass.values.astype(int)
        res = alt_test(jv, Ms, eps=args.eps)
        acc = (jv == (Ms.sum(1) > 1.5)).mean()
        print(f"  {cond:12s} n={len(sub):5d} acc_vs_majority={acc:.3f} "
              f"omega={res['omega']:.3f} rho={res['rho']:.3f} "
              f"{'PASS' if res['passes'] else 'FAIL'} "
              f"pass_rate={jv.mean():.3f} (human {(Ms.sum(1) > 1.5).mean():.3f})")
        # n_tool_calls is summed across samples by aggregate_samples and the arms run
        # different sample counts, so only the per-sample figure is comparable.
        if "tools_per_sample" in sub:
            print(f"               tools/sample={sub.tools_per_sample.mean():.2f} "
                  f"distinct/sample={sub.n_distinct_tools.mean():.2f} "
                  f"(raw summed field={sub.n_tool_calls.mean():.2f})")

    # The alt-test gains power with n, so a verdict is only comparable across analyses
    # run at the same n. Report the margin so a knife-edge PASS is visible as one.
    print("\nSENSITIVITY: alt-test verdict by epsilon")
    for cond, sub in full.groupby("condition"):
        Ms = np.array(sub.ratings.tolist()).astype(int)
        jv = sub.judge_pass.values.astype(int)
        marks = []
        for e in (0.05, 0.10, 0.15, 0.20):
            r = alt_test(jv, Ms, eps=e)
            marks.append(f"{e:.2f}:{'P' if r['passes'] else 'F'}(w={r['omega']:.2f})")
        print(f"  {cond:12s} " + "  ".join(marks))

    print("\nRQ2: BY CATEGORY AND MODALITY")
    for key in ("category", "modality"):
        if key not in full or full[key].isna().all():
            continue
        g = full.groupby(["condition", key]).apply(
            lambda s: pd.Series({
                "n": len(s),
                "acc": (s.judge_pass.values == (np.array(s.ratings.tolist()).sum(1) > 1.5)).mean(),
            }), include_groups=False)
        print(f"\n{key}:\n{g.round(3)}")

    print("\nAGREEMENT ON ITEMS WHERE HUMANS SPLIT")
    for cond, sub in full.groupby("condition"):
        sp = sub[sub.human_split]
        if len(sp):
            Ms = np.array(sp.ratings.tolist()).astype(int)
            acc = (sp.judge_pass.values == (Ms.sum(1) > 1.5)).mean()
            print(f"  {cond:12s} n={len(sp):4d} acc={acc:.3f}")

    if "self_preference" in full and full.self_preference.any():
        print("\nSELF-PREFERENCE (gemini-3.7-flash judging its own answers)")
        for cond, sub in full.groupby("condition"):
            a = sub[sub.self_preference]; b = sub[~sub.self_preference]
            if len(a) and len(b):
                f = lambda s: (s.judge_pass.values ==
                               (np.array(s.ratings.tolist()).sum(1) > 1.5)).mean()
                print(f"  {cond:12s} self n={len(a):4d} acc={f(a):.3f} "
                      f"pass={a.judge_pass.mean():.3f} | "
                      f"other n={len(b):4d} acc={f(b):.3f} pass={b.judge_pass.mean():.3f}")


if __name__ == "__main__":
    main()
