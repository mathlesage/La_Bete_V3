# pip install datasets sentence-transformers torch huggingface_hub
from typing import Union, Callable, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
import math
import hashlib
import json

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
            # accepte un tenseur torch, sinon convertit en torch
            if 'torch' in str(type(arr)):
                X = arr
            else:
                X = np.asarray(arr)
                if torch is None:
                    raise RuntimeError("torch est requis si la fonction ne retourne pas un tenseur torch")
                X = torch.tensor(X)
            # normalisation L2
            X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
            return X
        return embed

    if isinstance(embedding_model, str):
        if SentenceTransformer is None:
            raise RuntimeError("Installe sentence-transformers pour utiliser un modèle par nom")
        st = SentenceTransformer(embedding_model, device=device, trust_remote_code=True)
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


def _make_row_key(a, b, id_val=None) -> str:
    """
    Construit une clé stable de ligne pour reconnaître ce qui a déjà été calculé.

    - Si id_val (id unique) est fourni et non vide, utilise 'id::<id>'
    - Sinon, utilise un hash SHA1 du couple (a, b) après normalisation/serialization
    """
    if id_val is not None and not _is_empty_cell(id_val):
        return f"id::{str(id_val)}"

    s = json.dumps(
        [
            None if _is_empty_cell(a) else str(a),
            None if _is_empty_cell(b) else str(b),
        ],
        ensure_ascii=False,
        separators=(",", ":")
    )
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _load_existing_map_from_repo(
    repo_id: str,
    col_a: str,
    col_b: str,
    out_col: str,
    id_col: Optional[str],
    revision: Optional[str],
    config_name: Optional[str],
    hf_token: Optional[str],
    **extra_load_kwargs
) -> Dict[str, float]:
    """
    Retourne {row_key -> valeur out_col} pour toutes les lignes du repo_id
    où out_col n'est pas vide.

    Tolérant:
    - Si le repo n'existe pas: renvoie {}
    - Si col_a/col_b manquent dans le repo de sortie mais id_col est présent: utilise id_col
    - Sinon: ignore les lignes où la clé ne peut pas être construite
    """
    load_args = dict(name=config_name, revision=revision, token=hf_token, **extra_load_kwargs)
    load_args = {k: v for k, v in load_args.items() if v is not None}

    try:
        ds_out = load_dataset(repo_id, **load_args)
    except Exception:
        return {}

    existing: Dict[str, float] = {}

    def collect(batch):
        # Si out_col n'est pas dans ce batch, rien à faire
        if out_col not in batch:
            return {}

        n = len(batch[out_col])
        # Prépare accès défensif aux colonnes
        has_a = col_a in batch
        has_b = col_b in batch
        has_id = bool(id_col) and (id_col in batch)

        for i in range(n):
            val = batch[out_col][i]
            if _is_empty_cell(val):
                continue

            # essaie de construire la clé
            a = batch[col_a][i] if has_a else None
            b = batch[col_b][i] if has_b else None
            idv = batch[id_col][i] if has_id else None

            # si pas d'id et pas de textes, on ne peut pas indexer
            if _is_empty_cell(idv) and (a is None and b is None):
                continue

            key = _make_row_key(a, b, idv)
            existing[key] = float(val)
        return {}

    for split in ds_out.values():
        split.map(collect, batched=True, batch_size=1000)

    return existing


def _map_split(
    split: Dataset,
    embed_fn: Callable,
    col_a: str,
    col_b: str,
    out_col: str,
    batch_size: int,
    num_proc: Optional[int] = None,
    id_col: Optional[str] = None,
    existing_map: Optional[Dict[str, float]] = None,
) -> Dataset:
    """
    Map sur un split:
    - copie la valeur out_col depuis existing_map si disponible et cellule vide
    - sinon calcule seulement si cellule vide et (a,b) non vides
    """
    existing_map = existing_map or {}

    def fn(batch: Dict[str, list]):
        n = len(batch[col_a])
        existing_col = batch.get(out_col, [None] * n)
        out_vals = list(existing_col)

        idxs_to_compute = []

        # pre-check colonnes optionnelles
        has_id = bool(id_col) and (id_col in batch)

        for i in range(n):
            a = batch[col_a][i]
            b = batch[col_b][i]
            idv = batch[id_col][i] if has_id else None
            key = _make_row_key(a, b, idv)

            # 1) si on a déjà une valeur dans le repo de sortie, on la copie si la cellule est vide
            if key in existing_map and _is_empty_cell(existing_col[i]):
                out_vals[i] = float(existing_map[key])
                continue

            # 2) sinon, on calcule seulement si la cellule est vide ET que a,b sont non vides
            if _is_empty_cell(existing_col[i]) and (not _is_empty_cell(a)) and (not _is_empty_cell(b)):
                idxs_to_compute.append(i)

        # calcule en batch si besoin
        if idxs_to_compute:
            texts_a = [str(batch[col_a][i]) for i in idxs_to_compute]
            texts_b = [str(batch[col_b][i]) for i in idxs_to_compute]

            A = embed_fn(texts_a)
            B = embed_fn(texts_b)

            # vecteurs normalisés -> cosinus = produit scalaire
            if hasattr(A, "cpu"):
                sims = (A * B).sum(dim=1).cpu().numpy()
            else:
                sims = (A * B).sum(axis=1)

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
    id_col: Optional[str] = None,
    existing_map: Optional[Dict[str, float]] = None,
    **load_kwargs
) -> DatasetDict:
    """
    Calcule la similarité cosinus entre col_a et col_b sur tous les splits
    Écrit dans out_col uniquement si la cellule est vide OU si une valeur existe dans existing_map
    (copie) pour la clé correspondante.

    - ds peut être: nom de dataset HF Hub, Dataset, ou DatasetDict
    - id_col: si présent, sert à construire la clé de ligne prioritaire
    - existing_map: {row_key -> valeur out_col} pour réutiliser les calculs déjà effectués
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
        # id_col est optionnel, pas bloquant s'il manque

    updated = {}
    for name, split in ds.items():
        updated[name] = _map_split(
            split=split,
            embed_fn=embed_fn,
            col_a=col_a,
            col_b=col_b,
            out_col=out_col,
            batch_size=batch_size,
            num_proc=num_proc,
            id_col=id_col,
            existing_map=existing_map
        )

    return DatasetDict(updated)


# ---------------- Push to Hub (incrémental) ----------------

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
    id_col: Optional[str] = None,          # identifiant stable pour l'incrémental (facultatif)
    **extra_load_kwargs
):
    """
    Version incrémentale:
    1) charge le dataset d'entrée depuis le Hub
    2) charge (si présent) le dataset de sortie et construit {row_key -> out_col}
    3) calcule la cos-sim entre col_a et col_b sur tous les splits, uniquement pour les cellules vides non couvertes par existing_map
    4) pousse le résultat vers le repo de sortie
    """
    # 1) charge le dataset d'entrée
    load_args = dict(name=config_name, revision=revision, token=hf_token, **extra_load_kwargs)
    load_args = {k: v for k, v in load_args.items() if v is not None}
    ds_in = load_dataset(repo_id_in, **load_args)

    # 2) construit la mémoire depuis le repo de sortie (s'il existe)
    target_repo = repo_id_out or repo_id_in
    existing_map = _load_existing_map_from_repo(
        repo_id=target_repo,
        col_a=col_a,
        col_b=col_b,
        out_col=out_col,
        id_col=id_col,
        revision=revision,
        config_name=config_name,
        hf_token=hf_token,
        **extra_load_kwargs
    )

    # 3) calcule en ne complétant que ce qui manque (en se basant sur la sortie existante)
    updated = compute_cossim_on_dataset(
        ds=ds_in,
        embedding_model=embedding_model,
        col_a=col_a,
        col_b=col_b,
        out_col=out_col,
        batch_size=batch_size,
        device=device,
        num_proc=num_proc,
        id_col=id_col,
        existing_map=existing_map
    )

    # 4) push sur le Hub
    push_args = dict(
        repo_id=target_repo,
        private=private,
        token=hf_token,
        max_shard_size=max_shard_size,
        commit_message=commit_message
    )
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
        repo_id_in="matheoqtb/ancre_querry2_validated",                  # dataset source sur le Hub
        col_a="anchor_text",                                  # ex: "question"
        col_b="generated_text",                               # ex: "reponse"
        out_col="cos_sim_pos",                                # colonne cible
        embedding_model="Lajavaness/bilingual-embedding-large",
        repo_id_out="matheoqtb/ancre_querry_cos",             # None -> push dans le même repo
        private=None,                                         # None -> conserve la visibilité existante
        revision="main",                                      # branche/révision à lire
        config_name=None,                                     # config si le dataset en a une
        hf_token=os.environ.get("HF_TOKEN"),                  # ou mets directement "hf_xxx"
        batch_size=128,
        device="cuda",                                        # ou "cpu"
        num_proc=None,                                        # ou un int pour paralléliser
        commit_message="Compute cos_sim between anchor_text and generated_text",
        id_col=None                                           # si tu as un ID stable, mets le nom de la colonne ici (ex: "id")
    )

    print(f"Dataset poussé sur: {repo_final}")
