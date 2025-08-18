"""
Test des hard negatives avec embeddings
"""
import json
import numpy as np
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingTester:
    """Teste la qualité des hard negatives avec des embeddings"""
    
    def __init__(self, model_name: str = "Lajavaness/bilingual-embedding-large"):
        """Initialise avec le modèle d'embedding
        
        Note: Utilise gte-Qwen2-1.5B-instruct par défaut.
        Vous pouvez changer pour 'Qwen/Qwen2-0.5B' ou tout autre modèle d'embedding.
        """
        print(f"Chargement du modèle {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Modèle chargé sur {self.device}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode des textes en embeddings"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = torch.sum(embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            return embeddings.cpu().numpy()
    
    def test_hard_negatives(self, data: List[Dict]) -> Dict:
        """
        Teste les hard negatives et retourne les métriques
        
        Args:
            data: Liste de dicts avec query, positive, hard_negative_from_query, hard_negative_from_positive
        
        Returns:
            Dict avec les similarités calculées
        """
        all_texts = []
        for sample in data:
            all_texts.extend([
                sample['query'],
                sample['positive'],
                sample['hard_negative_from_query'],
                sample['hard_negative_from_positive']
            ])
        
        # Encoder tous les textes
        print("Encodage des textes...")
        embeddings = self.encode(all_texts)
        
        # Calculer les similarités pour chaque échantillon
        results = []
        for i, sample in enumerate(data):
            idx = i * 4
            query_emb = embeddings[idx]
            positive_emb = embeddings[idx + 1]
            hard_neg_q_emb = embeddings[idx + 2]
            hard_neg_p_emb = embeddings[idx + 3]
            
            result = {
                'query': sample['query'][:50],
                'query_full': sample['query'],
                'positive': sample['positive'][:50],
                'positive_full': sample['positive'],
                'hard_neg_query': sample['hard_negative_from_query'][:50],
                'hard_neg_query_full': sample['hard_negative_from_query'],
                'hard_neg_positive': sample['hard_negative_from_positive'][:50],
                'hard_neg_positive_full': sample['hard_negative_from_positive'],
                'sim_query_positive': float(cosine_similarity([query_emb], [positive_emb])[0][0]),
                'sim_query_hardneg_q': float(cosine_similarity([query_emb], [hard_neg_q_emb])[0][0]),
                'sim_query_hardneg_p': float(cosine_similarity([query_emb], [hard_neg_p_emb])[0][0]),
                'prompt': sample.get('prompt_used_query', 'N/A'),
                'prompt_used_positive': sample.get('prompt_used_positive', 'N/A'),
            }
            results.append(result)
        
        return results

def evaluate_quality(results: List[Dict]) -> None:
    """Évalue et affiche la qualité des hard negatives"""
    
    # Afficher un exemple de résultat pour vérification
    print("\n" + "-"*60)
    print("EXEMPLE DE TRANSFORMATION:")
    if results:
        ex = results[0]
        print(f"Query: {ex['query_full']}")
        print(f"Positive: {ex['positive_full']}")
        print(f"Hard Neg (Query): {ex['hard_neg_query_full']} [Sim: {ex['sim_query_hardneg_q']:.3f}]")
        print(f"Hard Neg (Positive): {ex['hard_neg_positive_full']} [Sim: {ex['sim_query_hardneg_p']:.3f}]")
    
    print("\n" + "="*60)
    print("ÉVALUATION DES HARD NEGATIVES")
    print("="*60)
    
    # Critères idéaux
    ideal_ranges = {
        'sim_query_positive': (0.7, 1.0, "Devrait être élevé"),
        'sim_query_hardneg_q': (0.3, 0.7, "Hard negative idéal"),
        'sim_query_hardneg_p': (0.3, 0.7, "Hard negative idéal")
    }
    
    # Analyser chaque métrique
    for metric, (min_val, max_val, desc) in ideal_ranges.items():
        values = [r[metric] for r in results]
        mean_val = np.mean(values)
        
        # Compter combien sont dans la plage idéale
        in_range = sum(1 for v in values if min_val <= v <= max_val)
        percentage = (in_range / len(values)) * 100
        
        print(f"\n{metric}:")
        print(f"  Moyenne: {mean_val:.3f}")
        print(f"  Range idéal: [{min_val}, {max_val}] - {desc}")
        print(f"  Dans le range: {percentage:.0f}% ({in_range}/{len(values)})")
    
    # Afficher les échantillons problématiques
    print("\n" + "-"*60)
    print("ÉCHANTILLONS PROBLÉMATIQUES:")
    
    for i, result in enumerate(results):
        problems = []
        
        if result['sim_query_positive'] < 0.7:
            problems.append(f"Similarité query-positive faible: {result['sim_query_positive']:.3f}")
        
        if result['sim_query_hardneg_q'] > 0.75:
            problems.append(f"Hard neg (query) trop similaire: {result['sim_query_hardneg_q']:.3f}")
        elif result['sim_query_hardneg_q'] < 0.25:
            problems.append(f"Hard neg (query) trop différent: {result['sim_query_hardneg_q']:.3f}")
        
        if result['sim_query_hardneg_p'] > 0.75:
            problems.append(f"Hard neg (positive) trop similaire: {result['sim_query_hardneg_p']:.3f}")
        elif result['sim_query_hardneg_p'] < 0.25:
            problems.append(f"Hard neg (positive) trop différent: {result['sim_query_hardneg_p']:.3f}")
        
        if problems:
            print(f"\n#{i+1} Query: {result['query_full']}")
            print(f"     Hard neg (Q): {result['hard_neg_query_full']}")
            print(f"     Hard neg (P): {result['hard_neg_positive_full']}")
            print(f"     Prompts utilisés: {result['prompt']} / {result['prompt_used_positive']}")
            for p in problems:
                print(f"  ⚠️ {p}")
    
    # Recommandations
    print("\n" + "-"*60)
    print("RECOMMANDATIONS:")
    
    avg_hardneg = np.mean([r['sim_query_hardneg_q'] for r in results] + 
                          [r['sim_query_hardneg_p'] for r in results])
    
    if avg_hardneg < 0.3:
        print("• Les hard negatives sont trop faciles (trop différents)")
        print("  → Ajuster les prompts pour plus de similarité")
    elif avg_hardneg > 0.7:
        print("• Les hard negatives sont trop similaires")
        print("  → Augmenter la différenciation dans les prompts")
    else:
        print("• ✅ Les hard negatives sont dans une bonne plage")
    
    print("\n" + "="*60)

def main():
    """Fonction principale de test"""
    
    # Charger les données générées
    try:
        with open('hard_negatives_output.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ {len(data)} échantillons chargés")
    except FileNotFoundError:
        # Données de test par défaut
        print("Fichier non trouvé, utilisation de données de test")
        data = [
            {
                "query": "Comment installer Python ?",
                "positive": "Téléchargez Python depuis python.org et lancez l'installateur.",
                "hard_negative_from_query": "Comment désinstaller Python de votre système ?",
                "hard_negative_from_positive": "Python est un langage disponible sur python.org."
            },
            {
                "query": "Qu'est-ce que React ?",
                "positive": "React est une bibliothèque JavaScript pour créer des interfaces.",
                "hard_negative_from_query": "Qu'est-ce que Angular dans le développement web ?",
                "hard_negative_from_positive": "JavaScript utilise React comme l'une de ses nombreuses bibliothèques."
            }
        ]
    
    # Tester avec les embeddings
    tester = EmbeddingTester()
    results = tester.test_hard_negatives(data)
    
    # Évaluer la qualité
    evaluate_quality(results)
    
    # Sauvegarder les résultats détaillés
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Résultats détaillés sauvés dans: test_results.json")

if __name__ == "__main__":
    main()