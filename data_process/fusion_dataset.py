# pip install -U datasets huggingface_hub pandas pyarrow
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import pandas as pd

# ---------- à personnaliser ----------
login()  # colle ton token

repo_left  = "matheoqtb/ancre_querry_neg_cos_test"     # dataset principal
repo_right = "matheoqtb/ancre_querry_filtre"     # dataset à joindre
new_repo   = "matheoqtb/dataset_merged2"
split      = "train"                     # None pour traiter tous les splits communs
how        = "left"                     # 'inner' 'left' 'right' 'outer'
left_key   = None                        # ex "id" si tu connais la clé côté gauche
right_key  = None                        # ex "uid" si le nom diffère côté droit
keep_right_key = False                   # garde ou non la clé droite si elle a un nom différent
# -------------------------------------

def merge_one_split(ds_left, ds_right, left_key=None, right_key=None,
                    how="outer", keep_right_key=False):
    dfL = ds_left.to_pandas()
    dfR = ds_right.to_pandas()

    # choix auto de la clé si non fournie
    if left_key is None or right_key is None:
        if "id" in dfL.columns and "id" in dfR.columns:
            left_key = right_key = "id"
        else:
            # fallback: jointure sur l'index de ligne
            dfL["_row_idx"] = range(len(dfL))
            dfR["_row_idx"] = range(len(dfR))
            left_key = right_key = "_row_idx"

    # cast doux des clés pour éviter les conflits de types
    dfL[left_key] = dfL[left_key].astype("object")
    dfR[right_key] = dfR[right_key].astype("object")

    # ignorer les colonnes de droite qui ont le même nom que celles de gauche
    overlap = set(dfL.columns) & set(dfR.columns)
    if left_key == right_key:
        overlap -= {left_key}  # on laisse passer la clé si identique
    right_cols = [c for c in dfR.columns if c not in overlap]

    merged = pd.merge(dfL, dfR[right_cols], left_on=left_key, right_on=right_key, how=how)

    # optionnel: retirer la clé droite si les noms diffèrent
    if left_key != right_key and not keep_right_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

    # nettoyage si jointure par index
    if "_row_idx" in merged.columns:
        merged = merged.drop(columns=["_row_idx"])

    return Dataset.from_pandas(merged, preserve_index=False)

if split:
    dsL = load_dataset(repo_left, split=split)
    dsR = load_dataset(repo_right, split=split)
    out = merge_one_split(dsL, dsR, left_key, right_key, how, keep_right_key)
    out.push_to_hub(new_repo, private=False, max_shard_size="500MB")
else:
    dL = load_dataset(repo_left)   # DatasetDict
    dR = load_dataset(repo_right)  # DatasetDict
    common = set(dL.keys()) & set(dR.keys())
    merged = {s: merge_one_split(dL[s], dR[s], left_key, right_key, how, keep_right_key) for s in common}
    DatasetDict(merged).push_to_hub(new_repo, private=False, max_shard_size="500MB")
