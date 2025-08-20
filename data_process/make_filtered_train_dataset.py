# pip install datasets huggingface_hub

from typing import Union, Optional, Dict
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import math


# ---------- Utils ----------

def _is_empty_cell(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float):
        try:
            return math.isnan(val)
        except Exception:
            pass
    if isinstance(val, str) and val.strip().lower() in {"", "nan"}:
        return True
    return False


def _ensure_datasetdict_train_only(ds: Union[Dataset, DatasetDict]) -> DatasetDict:
    """
    Garantit un DatasetDict avec un seul split 'train'.
    - si ds est un Dataset: le wrap dans {'train': ds}
    - si ds est un DatasetDict et contient 'train': ne garde que ce split
    - sinon concatène tous les splits dans un seul Dataset 'train'
      (suppose des schémas compatibles)
    """
    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})

    if not isinstance(ds, DatasetDict):
        raise TypeError("ds doit être un Dataset ou DatasetDict")

    if "train" in ds:
        return DatasetDict({"train": ds["train"]})

    # concatène tous les splits
    if len(ds) == 0:
        return DatasetDict({"train": Dataset.from_dict({})})

    base = None
    for split in ds.values():
        if base is None:
            base = split
        else:
            base = concatenate_datasets([base, split])
    return DatasetDict({"train": base})


def _filter_by_threshold(
    split: Dataset,
    cos_col: str,
    threshold: float
) -> Dataset:
    """
    Garde uniquement les lignes où cos_col >= threshold
    Les cellules vides ou non convertibles sont filtrées (exclues)
    """
    if cos_col not in split.column_names:
        raise ValueError(f"La colonne '{cos_col}' est absente du split")

    def keep_fn(batch: Dict[str, list]):
        vals = batch[cos_col]
        mask = []
        for v in vals:
            try:
                if _is_empty_cell(v):
                    mask.append(False)
                else:
                    mask.append(float(v) >= threshold)
            except Exception:
                mask.append(False)
        return mask

    return split.filter(keep_fn, batched=True)


# ---------- API principale ----------

def make_filtered_train_dataset(
    ds: Union[str, Dataset, DatasetDict],
    cos_col: str = "cos_sim",
    threshold: float = 0.7,
    **load_kwargs
) -> DatasetDict:
    """
    Charge un dataset (Hub ou local), ne garde qu'un split 'train', filtre par seuil.

    Paramètres
    - ds: nom de repo HF Hub (str) ou Dataset/DatasetDict
    - cos_col: nom de la colonne de similarité cosinus
    - threshold: seuil minimal à conserver (>= threshold)
    - **load_kwargs: pass-through pour load_dataset (ex: name=..., revision=..., token=...)

    Retour
    - DatasetDict({"train": <Dataset filtré>})
    """
    # 1) charger
    if isinstance(ds, str):
        ds = load_dataset(ds, **load_kwargs)

    # 2) n'avoir qu'un split train
    dd = _ensure_datasetdict_train_only(ds)
    train = dd["train"]

    # 3) filtrer
    filtered = _filter_by_threshold(train, cos_col=cos_col, threshold=threshold)

    return DatasetDict({"train": filtered})


def make_and_push_filtered_train_dataset(
    repo_id_in: str,
    cos_col: str,
    threshold: float,
    repo_id_out: str,
    private: Optional[bool] = None,
    hf_token: Optional[str] = None,
    commit_message: str = "Create filtered train split by cosine similarity threshold",
    max_shard_size: str = "500MB",
    **load_kwargs
) -> str:
    """
    Version Hub-to-Hub:
    - charge repo_id_in
    - crée un unique split 'train' filtré par seuil sur cos_col
    - push vers repo_id_out
    Retourne repo_id_out
    """
    dd = make_filtered_train_dataset(
        ds=repo_id_in,
        cos_col=cos_col,
        threshold=threshold,
        **load_kwargs
    )

    push_args = dict(
        repo_id=repo_id_out,
        private=private,
        token=hf_token,
        max_shard_size=max_shard_size,
        commit_message=commit_message
    )
    # enlever None
    push_args = {k: v for k, v in push_args.items() if v is not None}

    dd.push_to_hub(**push_args)
    return repo_id_out


# ---------- Exemple d’utilisation ----------

if __name__ == "__main__":
    """
    Exemple 1: local -> juste créer le DatasetDict filtré
    """
    make_and_push_filtered_train_dataset(
        repo_id_in="matheoqtb/ancre_querry_neg_all",
        cos_col="cos_pos__neg_from_positive",
        threshold=0.00,
        repo_id_out="matheoqtb/ancre_querry_neg_cos_all",
        revision="main"
    )
