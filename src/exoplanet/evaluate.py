from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

def report(y, p, thr=0.5, name="model"):
    pred = (p >= thr).astype(int)
    print(f"[{name}] positives: {int(y.sum())} of {len(y)}")
    print(f"PR-AUC : {average_precision_score(y, p):.3f}  (random is about {y.mean():.3f})")
    print(f"ROC-AUC: {roc_auc_score(y, p):.3f}")
    print(f"precision: {precision_score(y, pred, zero_division=0):.3f}, "
          f"recall: {recall_score(y, pred):.3f}  (threshold {thr})")
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, pred))