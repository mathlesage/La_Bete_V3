#!/bin/bash
# Ou #!/usr/bin/env bash
# Ou #!/bin/zsh si vous préférez Zsh explicitement

# Arrête le script immédiatement si une commande échoue
set -e

VENV_NAME=".venv"
# Chemin où uv est typiquement installé par son script d'installation
UV_INSTALL_DIR="$HOME/.uv/bin"

echo "--- Configuration de l'environnement Python avec uv ---"

# 1. Vérifier si uv est installé et accessible
if ! command -v uv &> /dev/null; then
    echo "'uv' n'est pas trouvé dans le PATH."
    # Vérifier s'il est installé mais pas dans le PATH de cette session
    if [ -x "$UV_INSTALL_DIR/uv" ]; then
        echo "'uv' trouvé dans $UV_INSTALL_DIR. Ajout au PATH pour cette session."
        export PATH="$UV_INSTALL_DIR:$PATH"
    else
        echo "Installation de 'uv'..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            echo "'uv' a été installé. Ajout de $UV_INSTALL_DIR au PATH pour cette session."
            export PATH="$UV_INSTALL_DIR:$PATH"
            # L'installateur devrait avoir mis à jour .bashrc/.zshrc pour les futures sessions.
        else
            echo "ERREUR : L'installation de 'uv' a échoué."
            exit 1
        fi
    fi

    # Vérification finale après tentative d'installation/mise à jour du PATH
    if ! command -v uv &> /dev/null; then
        echo "ERREUR : 'uv' n'est toujours pas accessible après l'installation."
        echo "Veuillez ouvrir un nouveau terminal ou ajouter manuellement '$UV_INSTALL_DIR' à votre PATH."
        echo "Par exemple, ajoutez 'export PATH=\"\$HOME/.uv/bin:\$PATH\"' à votre ~/.bashrc ou ~/.zshrc"
        exit 1
    fi
    echo "'uv' est maintenant disponible."
else
    echo "'uv' est déjà installé et accessible."
fi

echo "Version de uv :"
uv --version
echo ""

# 2. Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "$VENV_NAME" ]; then
    echo "Création de l'environnement virtuel '$VENV_NAME'..."
    if uv venv "$VENV_NAME"; then
        echo "Environnement virtuel '$VENV_NAME' créé avec succès."
    else
        echo "ERREUR : La création de l'environnement virtuel '$VENV_NAME' a échoué."
        exit 1
    fi
else
    echo "L'environnement virtuel '$VENV_NAME' existe déjà."
fi
echo ""

# 3. Instructions pour l'activation
echo "--- ACTION REQUISE ---"
echo "L'environnement est prêt. Pour l'activer dans votre session terminal actuelle, exécutez :"
echo ""
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Si vous avez exécuté ce script en le 'sourçant' (ex: 'source ./setup_uv_env.sh'),"
echo "l'environnement pourrait déjà être actif (vérifiez votre prompt)."
echo "----------------------"
echo ""
echo "Une fois activé, votre prompt devrait afficher '($VENV_NAME)'."
echo "Vous pourrez alors installer des paquets avec : uv pip install <paquet>"
echo "Ou depuis un fichier de dépendances : uv pip install -r requirements.txt"
echo "Pour désactiver l'environnement : deactivate"

# Fin du script
exit 0