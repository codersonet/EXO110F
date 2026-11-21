"""Prints (and saves) a one-page summary of the finished run.
Reads files that earlier steps already made. It trains nothing 
and touches no test data."""
import json

import numpy as np
import pandas as pd

from exoplanet.preprocess import ROOT

RES = ROOT / "results"
TOP = 10   # how many ranked candidates to show

final = json.loads((RES / "final_test.json").read_text())
cfg = json.loads((ROOT / "configs" / "preprocess_v1.json").read_text())
d = np.load(ROOT / "data" / "processed" / "dataset_v1.npz")
ranked = pd.read_csv(RES / "ranked_candidates_test.csv")

tp, fp, fn, tn = final["TP"], final["FP"], final["FN"], final["TN"]
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0


def counts(split):
    y = d[f"y_{split}"]
    return int(y.sum()), int((y == 0).sum())


(tr_p, tr_n), (va_p, va_n), (te_p, te_n) = counts("train"), counts("val"), counts("test")

out = []
out.append(f"Run: seed={final['seed']}, preprocess_{cfg['version']}, model={final['model']}")
out.append("")
out.append("Split (by star):" + " " * 6 + f"{'train':>7}{'val':>7}{'test':>7}")
out.append("  positives (planets)    " + f"{tr_p:>7}{va_p:>7}{te_p:>7}")
out.append("  negatives (no planet)  " + f"{tr_n:>7}{va_n:>7}{te_n:>7}")
out.append("  (the final model was trained on train + val together)")
out.append("")
out.append(f"Threshold (chosen by cross-validation, never on test data): {final['threshold_from_cv']:.3f}")
out.append("")
out.append("Test results (locked test file, used for the final exam):")
out.append(f"  Precision {precision:.3f}   Recall {recall:.3f}   "
           f"PR-AUC {final['pr_auc']:.3f}   ROC-AUC {final['roc_auc']:.3f}")
out.append(f"  Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
out.append("")
out.append(f"Top {TOP} ranked candidates (held-out test stars):")
for _, r in ranked.head(TOP).iterrows():
    flag = "flagged" if r["flagged_as_candidate"] else "below threshold"
    out.append(f"  {int(r['rank']):>2}. exoTest.csv row {int(r['csv_row']):<4} "
               f"score {r['planet_score']:.3f}   dataset label: {r['known_label']:<10} ({flag})")
out.append("")
out.append("Notes:")
out.append("  - The Kaggle data has no star names (such as KIC numbers), so stars are identified by row number.")
out.append(f"  - Only {te_p} planets are in the test set, so treat every number as rough.")
out.append("  - These are ranked candidates, not discoveries or confirmed planets.")

text = "\n".join(out)
print(text)
(RES / "final_summary.txt").write_text(text + "\n")
print("\nsaved", RES / "final_summary.txt")