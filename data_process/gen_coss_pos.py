# pip install datasets sentence-transformers torch huggingface_hub
from typing import Union, Callable, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
import math

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:
    torch = None
    SentenceTransformer = None

# ---------------- Utils ----------------

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


def _build_embedder(embedding_model: Union[str, Any, Callable], device: Optional[str] = None, batch_size: int = 64):
    """
    Retourne une fonction embed(texts: list[str]) -> torch.Tensor [n, d] normalisé L2
    embedding_model peut être:
      - str: nom d’un modèle Sentence-Transformers
      - instance SentenceTransformer
      - callable: retourne np.array/list/torch.Tensor d’embeddings
    """
    if callable(embedding_model):
        def embed(texts):
            arr = embedding_model(texts)
            if 'torch' in str(type(arr)):
                X = arr
            else:
                X = np.asarray(arr)
                if torch is None:
                    raise RuntimeError("torch est requis si la fonction ne retourne pas un tenseur torch")
                X = torch.tensor(X)
            X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            return X
        return embed

    if isinstance(embedding_model, str):
        if SentenceTransformer is None:
            raise RuntimeError("Installe sentence-transformers pour utiliser un modèle par nom")
        st = SentenceTransformer(embedding_model, device=device)
        def embed(texts):
            return st.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embed

    if hasattr(embedding_model, "encode"):
        def embed(texts):
            return embedding_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embed

    raise ValueError("embedding_model doit être un nom, une instance SentenceTransformer, ou un callable")


def _map_split(
    split: Dataset,
    embed_fn: Callable,
    col_a: str,
    col_b: str,
    out_col: str,
    batch_size: int,
    num_proc: Optional[int] = None,
) -> Dataset:
    def fn(batch: Dict[str, list]):
        n = len(batch[col_a])
        existing = batch.get(out_col, [None] * n)

        idxs_to_compute = []
        out_vals = list(existing)

        for i in range(n):
            # calcule seulement si la cellule cible est vide (ou si la colonne n'existait pas)
            if _is_empty_cell(existing[i]):
                a = batch[col_a][i]
                b = batch[col_b][i]
                if not _is_empty_cell(a) and not _is_empty_cell(b):
                    idxs_to_compute.append(i)

        if idxs_to_compute:
            texts_a = [str(batch[col_a][i]) for i in idxs_to_compute]
            texts_b = [str(batch[col_b][i]) for i in idxs_to_compute]

            A = embed_fn(texts_a)
            B = embed_fn(texts_b)

            # vecteurs normalisés -> cosinus = produit scalaire
            sims = (A * B).sum(dim=1).cpu().numpy() if hasattr(A, "cpu") else (A * B).sum(axis=1)

            for k, i in enumerate(idxs_to_compute):
                out_vals[i] = float(sims[k])

        return {out_col: out_vals}

    return split.map(fn, batched=True, batch_size=batch_size, num_proc=num_proc)


def compute_cossim_on_dataset(
    ds: Union[str, Dataset, DatasetDict],
    embedding_model: Union[str, Any, Callable],
    col_a: str,
    col_b: str,
    out_col: str = "cos_sim",
    batch_size: int = 64,
    device: Optional[str] = None,
    num_proc: Optional[int] = None,
    **load_kwargs
) -> DatasetDict:
    """
    Calcule la similarité cosinus entre col_a et col_b sur tous les splits
    Écrit dans out_col uniquement si la cellule est vide ou si la colonne n'existe pas
    """
    if isinstance(ds, str):
        ds = load_dataset(ds, **load_kwargs)

    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    if device is None and torch is not None and torch.cuda.is_available():
        device = "cuda"

    embed_fn = _build_embedder(embedding_model, device=device, batch_size=batch_size)

    # vérifie que les colonnes existent partout
    for name, split in ds.items():
        for col in (col_a, col_b):
            if col not in split.column_names:
                raise ValueError(f"Colonne '{col}' absente du split '{name}'")

    updated = {}
    for name, split in ds.items():
        updated[name] = _map_split(split, embed_fn, col_a, col_b, out_col, batch_size, num_proc=num_proc)

    return DatasetDict(updated)

# ---------------- Push to Hub ----------------

def compute_and_push_cossim_on_hub(
    repo_id_in: str,
    col_a: str,
    col_b: str,
    embedding_model: Union[str, Any, Callable],
    out_col: str = "cos_sim",
    repo_id_out: Optional[str] = None,     # si None, réécrit dans le même repo
    private: Optional[bool] = None,        # hérite du repo existant si None
    revision: Optional[str] = None,        # ex: "main"
    config_name: Optional[str] = None,     # ex: "default"
    hf_token: Optional[str] = None,        # sinon utilise le cache de login local
    batch_size: int = 64,
    device: Optional[str] = None,
    num_proc: Optional[int] = None,
    commit_message: str = "Add/update cosine similarity column",
    max_shard_size: str = "500MB",
    **extra_load_kwargs
):
    """
    1) charge le dataset depuis le Hub
    2) calcule la cos-similarité entre col_a et col_b sur tous les splits
    3) pousse le DatasetDict résultat sur le Hub (même repo par défaut)
    """
    load_args = dict(name=config_name, revision=revision, token=hf_token, **extra_load_kwargs)
    # supprime les clés None pour éviter les warnings
    load_args = {k: v for k, v in load_args.items() if v is not None}

    ds = load_dataset(repo_id_in, **load_args)

    updated = compute_cossim_on_dataset(
        ds=ds,
        embedding_model=embedding_model,
        col_a=col_a,
        col_b=col_b,
        out_col=out_col,
        batch_size=batch_size,
        device=device,
        num_proc=num_proc
    )

    target_repo = repo_id_out or repo_id_in

    # push sur le Hub
    # DatasetDict.push_to_hub gère la création ou la mise à jour du repo
    push_args = dict(
        repo_id=target_repo,
        private=private,
        token=hf_token,
        max_shard_size=max_shard_size,
        commit_message=commit_message
    )
    # enlève les None
    push_args = {k: v for k, v in push_args.items() if v is not None}

    updated.push_to_hub(**push_args)
    return target_repo

# ------------- Exemple d’utilisation -------------

if __name__ == "__main__":
    """
    Mets ton token Hugging Face dans la variable d’environnement HF_TOKEN
    ou passe-le en paramètre hf_token="hf_xxx"
    """
    import os

    repo_final = compute_and_push_cossim_on_hub(
        repo_id_in="matheoqtb/ancre_querry",                  # dataset source sur le Hub
        col_a="anchor_text",                         # ex: "question"
        col_b="generated_text",                         # ex: "reponse"
        out_col="cos_sim_pos",                                # colonne cible
        embedding_model="Lajavaness/bilingual-embedding-large",
        repo_id_out="matheoqtb/ancre_querry_cos",                                 # None -> push dans le même repo
        private=None,                                     # None -> conserve la visibilité existante
        revision="main",                                  # branche/révision à lire
        config_name=None,                                 # config si le dataset en a une
        hf_token=os.environ.get("HF_TOKEN"),              # ou mets directement "hf_xxx"
        batch_size=128,
        device="cuda",                                    # ou "cpu"
        num_proc=None,                                    # ou un int pour paralléliser
        commit_message="Compute cos_sim between colonne_source_1 and colonne_source_2"
    )

    print(f"Dataset poussé sur: {repo_final}")
