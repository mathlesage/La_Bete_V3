# pip install -U datasets huggingface_hub numpy pyarrow
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import numpy as np

# --------- à personnaliser ----------

repo_in   = "matheoqtb/dataset_merged"   # dataset source
new_repo  = "matheoqtb/dataset_pos_sup2" # dataset filtré sur le Hub
split     = "train"                        # None pour traiter tous les splits communs

col_A = "cos_sim_pos"        # colonne à comparer
col_B = "cos_pos__neg_from_positive"        # 1re colonne de référence
col_C = "cos_pos__neg_from_anchor"        # 2e colonne de référence
col_D = "cos_sim_pos"        # colonne soumise au seuil

threshold_D = 0.7
compare_to_sum = False   # False -> A > B et A > C ; True -> A > (B + C)
num_proc = 4             # parallélisme pour accélérer
# ------------------------------------

def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

def _prepare_numeric(ds, cols):
    # convertit en float de manière robuste
    return ds.map(lambda x: {c: _safe_float(x.get(c, None)) for c in cols},
                  desc="Cast en float",
                  num_proc=num_proc)

def _filter_one_split(ds, col_A, col_B, col_C, col_D, threshold_D, compare_to_sum):
    ds = _prepare_numeric(ds, [col_A, col_B, col_C, col_D])

    def predicate(batch):
        A = np.asarray(batch[col_A], dtype="float64")
        B = np.asarray(batch[col_B], dtype="float64")
        C = np.asarray(batch[col_C], dtype="float64")
        D = np.asarray(batch[col_D], dtype="float64")

        if compare_to_sum:
            condA = A > (B + C)
        else:
            condA = (A > B + 0.1) & (A > C + 0.1)

        condD = D > threshold_D

        # invalide si NaN/inf
        valid = np.isfinite(A) & np.isfinite(B) & np.isfinite(C) & np.isfinite(D)
        mask = valid & condA & condD
        return mask.tolist()

    return ds.filter(predicate, batched=True, desc="Filtrage", num_proc=num_proc)

if split:
    ds = load_dataset(repo_in, split=split)
    ds_out = _filter_one_split(ds, col_A, col_B, col_C, col_D, threshold_D, compare_to_sum)
    ds_out.push_to_hub(new_repo, private=False, max_shard_size="500MB")
else:
    dsd = load_dataset(repo_in)  # DatasetDict
    common_splits = list(dsd.keys())
    out = {s: _filter_one_split(dsd[s], col_A, col_B, col_C, col_D, threshold_D, compare_to_sum)
           for s in common_splits}
    DatasetDict(out).push_to_hub(new_repo, private=False, max_shard_size="500MB")
