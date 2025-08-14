"""
Générateur de Hard Negatives avec OpenRouter
"""
import os
import yaml
import json
import random
from typing import List, Dict, Optional
from openai import OpenAI

# Liste des modèles disponibles par provider
MODELS = {
    "openai": [
        "openai/gpt-4.1-nano",
    ],
    "google-vertex": [
        "google/gemini-2.5-flash-lite",
    ]
}
def load_prompts():
    """Charge les prompts depuis le fichier YAML"""
    try:
        with open('hard_negatives_prompts.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)['prompts']
    except:
        print("Erreur chargement prompts, utilisation prompts par défaut")
        return {
            'query_based': [{
                'name': 'default',
                'prompt': 'Génère une variation de: {query}\nVariation:'
            }],
            'positive_based': [{
                'name': 'default',
                'prompt': 'Modifie: {positive}\nPour la query: {query}\nModification:'
            }]
        }

def apply_aggressive_modification(text: str, original: str) -> str:
    """
    Applique des modifications agressives si le texte est trop similaire
    """
    if not text:
        return None
    
    # Liste de transformations radicales par domaine
    transformations = {
        "python": ["jardinage", "cuisine", "astronomie", "danse"],
        "react": ["chimie", "océan", "histoire", "musique"],
        "installer": ["détruire", "manger", "observer", "dessiner"],
        "javascript": ["biologie", "géologie", "poésie", "sport"],
        "gâteau": ["moteur", "équation", "planète", "symphonie"],
        "faire": ["défaire", "étudier", "rêver", "oublier"]
    }
    
    modified = text.lower()
    original_lower = original.lower()
    
    # Remplacer les mots trop similaires
    for key, replacements in transformations.items():
        if key in original_lower and key in modified:
            replacement = random.choice(replacements)
            modified = modified.replace(key, replacement)
    
    # Si toujours trop de mots en commun, générer quelque chose de radical
    original_words = set(original_lower.split())
    modified_words = set(modified.split())
    common_words = original_words & modified_words
    
    # Enlever les mots communs triviaux
    trivial = {"le", "la", "un", "une", "de", "du", "des", "et", "ou", "à", "?", "comment", "qu'est-ce", "que"}
    common_words = common_words - trivial
    
    if len(common_words) > 2:
        # Trop similaire, créer quelque chose de complètement différent
        templates = [
            "Les oiseaux migrent vers le sud en hiver.",
            "La théorie quantique explique les particules subatomiques.",
            "Les champignons poussent dans les forêts humides.",
            "Le jazz est né à la Nouvelle-Orléans.",
            "Les glaciers fondent à cause du réchauffement.",
            "La photosynthèse transforme la lumière en énergie.",
            "Les volcans entrent en éruption de façon imprévisible.",
            "L'architecture gothique date du Moyen Âge."
        ]
        return random.choice(templates)
    
    return modified.capitalize()

def generate_hard_negative(prompt: str, model: str = None, provider: str = None, original_text: str = "") -> str:
    """
    Génère un hard negative en utilisant OpenRouter
    
    Args:
        prompt: Le prompt complet à envoyer
        model: Le modèle à utiliser (si None, sélection aléatoire)
        provider: Le provider à utiliser (si spécifié, force ce provider)
        original_text: Le texte original pour vérification de similarité
    
    Returns:
        Le texte généré ou None si erreur
    """
    # Sélection du modèle
    if model is None:
        if provider and provider in MODELS:
            model = random.choice(MODELS[provider])
        else:
            all_models = [m for models in MODELS.values() for m in models]
            model = random.choice(all_models)
    
    # Configuration pour forcer le provider si spécifié
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    
    # System prompt plus agressif
    system_prompt = """Tu dois créer des phrases TRÈS DIFFÉRENTES de l'original.
    Règles STRICTES:
    1. Change COMPLÈTEMENT de domaine (tech→nature, cuisine→mécanique, etc.)
    2. Maximum 2 mots en commun avec l'original (hors mots comme le/la/un/de)
    3. Ne JAMAIS répondre à la question originale
    4. Le sens doit être TOTALEMENT différent
    Réponds avec UNE SEULE phrase courte."""
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # Plus de créativité
            max_tokens=150,
            extra_body=extra_body
        )
        
        result = response.choices[0].message.content.strip()
        
        # Appliquer des modifications agressives si nécessaire
        if original_text:
            result = apply_aggressive_modification(result, original_text)
        
        return result
    
    except Exception as e:
        print(f"\n❌ Erreur détaillée: {str(e)}")
        print(f"   Model: {model}")
        print(f"   Provider: {provider}")
        return None

def process_sample(query: str, positive: str, provider: str = None) -> Dict:
    """
    Traite un échantillon pour générer ses hard negatives
    
    Args:
        query: La query originale
        positive: La réponse positive
        provider: Le provider à utiliser (optionnel)
    
    Returns:
        Dict avec query, positive, et les hard negatives générés
    """
    prompts = load_prompts()
    
    # Sélectionner un prompt aléatoire pour query-based
    query_prompt_config = random.choice(prompts['query_based'])
    query_prompt = query_prompt_config['prompt'].format(query=query)
    
    # Sélectionner un prompt aléatoire pour positive-based
    positive_prompt_config = random.choice(prompts['positive_based'])
    positive_prompt = positive_prompt_config['prompt'].format(query=query, positive=positive)
    
    # Générer les hard negatives
    hard_neg_from_query = generate_hard_negative(query_prompt, provider=provider)
    hard_neg_from_positive = generate_hard_negative(positive_prompt, provider=provider)
    
    # Messages d'erreur plus informatifs
    if hard_neg_from_query is None:
        print(f"  ⚠️ Échec génération hard_neg_from_query")
        hard_neg_from_query = f"[Erreur] Variation de: {query[:50]}..."
    
    if hard_neg_from_positive is None:
        print(f"  ⚠️ Échec génération hard_neg_from_positive")
        hard_neg_from_positive = f"[Erreur] Modification de: {positive[:50]}..."
    
    return {
        'query': query,
        'positive': positive,
        'hard_negative_from_query': hard_neg_from_query,
        'hard_negative_from_positive': hard_neg_from_positive,
        'prompt_used_query': query_prompt_config['name'],
        'prompt_used_positive': positive_prompt_config['name']
    }

def main():
    """Fonction principale"""
    
    # Vérifier la clé API
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY non définie")
        print("export OPENROUTER_API_KEY='votre_clé'")
        return
    
    # Dataset à utiliser
    # Vérifier d'abord si dataset_100.json existe
    try:
        with open('dataset_100.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✓ Dataset chargé depuis dataset_100.json: {len(dataset)} échantillons")
    except FileNotFoundError:
        print("dataset_100.json non trouvé, utilisation du dataset d'exemple")
        dataset = [
            {
                "query": "Comment installer Python ?",
                "positive": "Téléchargez Python depuis python.org et lancez l'installateur."
            },
            {
                "query": "Qu'est-ce que React ?",
                "positive": "React est une bibliothèque JavaScript pour créer des interfaces utilisateur."
            },
            {
                "query": "Comment faire un gâteau ?",
                "positive": "Mélangez farine, œufs, sucre et beurre, puis enfournez 30 minutes à 180°C."
            },
            {
                "query": "Quelle est la capitale de la France ?",
                "positive": "La capitale de la France est Paris, située sur la Seine."
            },
            {
                "query": "Comment déboguer du code JavaScript ?",
                "positive": "Utilisez console.log, les breakpoints et les outils de développement du navigateur."
            },
            {
                "query": "Qu'est-ce que Docker ?",
                "positive": "Docker est une plateforme de conteneurisation pour empaqueter des applications."
            },
            {
                "query": "Comment apprendre le machine learning ?",
                "positive": "Commencez par les mathématiques, puis Python, et suivez des cours en ligne."
            },
            {
                "query": "Pourquoi le ciel est bleu ?",
                "positive": "La lumière bleue est diffusée par les molécules d'air dans l'atmosphère."
            },
            {
                "query": "Comment créer une API REST ?",
                "positive": "Définissez les endpoints, utilisez HTTP verbs et retournez du JSON."
            },
            {
                "query": "Qu'est-ce que Git ?",
                "positive": "Git est un système de contrôle de version distribué pour le code source."
            }
        ]
    
    # Demander le provider
    print("\nProviders disponibles:")
    providers_list = list(MODELS.keys())
    for i, p in enumerate(providers_list, 1):
        models_count = len(MODELS[p])
        print(f"  {i}. {p} ({models_count} modèle{'s' if models_count > 1 else ''})")
    print(f"  {len(providers_list)+1}. Tous (aléatoire)")
    
    choice = input("\nChoisir (numéro): ").strip()
    
    provider = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(providers_list):
            provider = providers_list[idx]
            print(f"✓ Provider: {provider}")
        else:
            print("✓ Tous les providers")
    
    # Traiter le dataset
    print("\nGénération des hard negatives...")
    results = []
    
    for sample in dataset:
        print(f"\nTraitement: {sample['query'][:50]}...")
        result = process_sample(sample['query'], sample['positive'], provider)
        results.append(result)
        
        # Afficher le résultat
        print(f"  ✓ Hard neg (query): {result['hard_negative_from_query'][:60]}...")
        print(f"  ✓ Hard neg (positive): {result['hard_negative_from_positive'][:60]}...")
    
    # Sauvegarder les résultats
    with open('hard_negatives_output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ {len(results)} échantillons générés")
    print("📁 Sauvegardé dans: hard_negatives_output.json")

if __name__ == "__main__":
    main()