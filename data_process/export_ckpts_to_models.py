# export_ckpts_to_models.py
import os, re, tempfile, shutil
from typing import List
from huggingface_hub import HfApi, create_repo, snapshot_download, hf_hub_url

def sanitize(name: str) -> str:
    name = name.strip().replace(" ", "-")
    name = re.sub(r"[^A-Za-z0-9._\-]", "-", name)
    # éviter noms trop longs
    return name[:96].strip("-._")

def detect_subfolders(api: HfApi, repo_id: str, token: str) -> List[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    subs = set()
    for f in files:
        # on considère un sous-dossier s’il contient un config.json (ou model.safetensors/bin)
        if "/" in f:
            top = f.split("/", 1)[0]
            if f.startswith(top + "/config.json") or f.startswith(top + "/model.safetensors") or f.startswith(top + "/pytorch_model.bin"):
                subs.add(top)
    # ignorer dossiers techniques
    subs = [s for s in subs if not s.startswith(".") and s not in ("snapshots",)]
    return sorted(subs)

def build_readme(src_repo: str, sub: str, base_code_repo: str) -> str:
    return (
        f"# Checkpoint exporté: {sub}\n\n"
        f"Ce dépôt contient un checkpoint extrait de `{src_repo}` (sous-dossier `{sub}`) et les fichiers de code "
        f"nécessaires provenant de `{base_code_repo}`.\n\n"
        "Chargement:\n"
        "    from transformers import AutoTokenizer, AutoModel\n"
        f"    tok = AutoTokenizer.from_pretrained('<THIS_REPO>', trust_remote_code=True)\n"
        f"    mdl = AutoModel.from_pretrained('<THIS_REPO>', trust_remote_code=True)\n"
        "\n"
        "Tâche: feature-extraction (embeddings)\n"
    )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_repo", required=True, help="Repo source des checkpoints (ex: Mathlesage/mmarco-eurobert-ckpts)")
    ap.add_argument("--base_code_repo", default="EuroBERT/EuroBERT-610m", help="Repo qui contient le remote code (fichiers .py EuroBERT)")
    ap.add_argument("--target_namespace", required=True, help="Namespace (utilisateur/org) pour les nouveaux repos")
    ap.add_argument("--repo_prefix", default="mmarco-eurobert-", help="Préfixe des nouveaux repos")
    ap.add_argument("--only_subfolders", nargs="*", default=None, help="Sous-dossiers à exporter (sinon auto-detection)")
    ap.add_argument("--private", action="store_true", help="Créer les repos en privé")
    ap.add_argument("--overwrite", action="store_true", help="Ecraser le contenu si le repo existe déjà")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("❌ HF_TOKEN non défini dans l'environnement.")

    # accélérateur transfert
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    api = HfApi()

    # détecte les sous-dossiers si non fournis
    if args.only_subfolders:
        subfolders = args.only_subfolders
    else:
        subfolders = detect_subfolders(api, args.src_repo, token)
        if not subfolders:
            raise SystemExit(f"❌ Aucun sous-dossier détecté dans {args.src_repo}. Vérifie qu'il y a p.ex. '5M_pairs/config.json'.")

    print(f"➡️  Sous-dossiers trouvés: {subfolders}")

    # télécharger une fois le code .py du modèle de base
    with tempfile.TemporaryDirectory() as tmp_code:
        code_dir = snapshot_download(
            repo_id=args.base_code_repo,
            repo_type="model",
            allow_patterns=["*.py"],  # tous les fichiers python nécessaires au remote code
            local_dir=tmp_code,
            local_dir_use_symlinks=False,
            token=token,
        )

        for sub in subfolders:
            print(f"\n=== Traitement: {sub} ===")
            with tempfile.TemporaryDirectory() as tmp_ckpt:
                # 1) snapshot du sous-dossier
                ckpt_dir = snapshot_download(
                    repo_id=args.src_repo,
                    repo_type="model",
                    allow_patterns=[f"{sub}/*"],
                    local_dir=tmp_ckpt,
                    local_dir_use_symlinks=False,
                    token=token,
                )
                src_sub_path = os.path.join(ckpt_dir, sub)
                if not os.path.isdir(src_sub_path):
                    print(f"⚠️  Introuvable localement: {src_sub_path} (skip)")
                    continue

                # 2) staging: merge ckpt + code
                with tempfile.TemporaryDirectory() as tmp_stage:
                    # copier le contenu du checkpoint
                    for root, _, files in os.walk(src_sub_path):
                        rel = os.path.relpath(root, src_sub_path)
                        dst = os.path.join(tmp_stage, rel) if rel != "." else tmp_stage
                        os.makedirs(dst, exist_ok=True)
                        for fn in files:
                            shutil.copy2(os.path.join(root, fn), os.path.join(dst, fn))

                    # copier les .py (remote code) à la racine du staging
                    for root, _, files in os.walk(code_dir):
                        for fn in files:
                            if fn.endswith(".py"):
                                shutil.copy2(os.path.join(root, fn), os.path.join(tmp_stage, fn))

                    # 3) README minimal
                    readme_path = os.path.join(tmp_stage, "README.md")
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(build_readme(args.src_repo, sub, args.base_code_repo))

                    # 4) créer repo cible et upload
                    repo_name = sanitize(args.repo_prefix + sub)
                    target_repo = f"{args.target_namespace}/{repo_name}"
                    create_repo(
                        repo_id=target_repo,
                        repo_type="model",
                        private=args.private,
                        exist_ok=True,
                        token=token,
                    )

                    print(f"⬆️  Upload vers {target_repo} …")
                    api.upload_folder(
                        repo_id=target_repo,
                        repo_type="model",
                        folder_path=tmp_stage,
                        path_in_repo=".",  # à la racine
                        token=token,
                        commit_message=f"Import checkpoint from {args.src_repo}/{sub} + remote code {args.base_code_repo}",
                        # si --overwrite n'est pas passé, on laisse HF faire des deltas; sinon on pourrait nettoyer avant
                    )

                    print(f"✅ Terminé: {target_repo}")
                    print(f"   → {hf_hub_url(target_repo, '')}")

    print("\n🎉 Tous les sous-dossiers ont été traités.")
if __name__ == "__main__":
    main()
