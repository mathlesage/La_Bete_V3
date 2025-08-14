"""
Script d'analyse détaillée des hard negatives par prompt
Teste sur 100 échantillons et analyse les performances de chaque prompt
"""

import json
import numpy as np
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from datasets import load_dataset

def load_mathlesage_dataset(n_samples=100):
    """Charge le dataset Mathlesage depuis HuggingFace"""
    print("Chargement du dataset Mathlesage/La_Bete_data_2...")
    
    try:
        # Charger le dataset depuis HuggingFace
        dataset = load_dataset("Mathlesage/La_Bete_data_2", split="train")
        
        # Préparer les données pour le générateur
        samples = []
        for i in range(min(n_samples, len(dataset))):
            sample = {
                "query": dataset[i]["anchor"],
                "positive": dataset[i]["response"]
            }
            samples.append(sample)
        
        print(f"✓ {len(samples)} échantillons chargés")
        return samples
    
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        print("Utilisation d'échantillons de test...")
        # Fallback sur des échantillons de test
        return [
            {
                "query": "Comment installer Python ?",
                "positive": "Téléchargez Python depuis python.org et lancez l'installateur."
            },
            {
                "query": "Qu'est-ce que React ?",
                "positive": "React est une bibliothèque JavaScript pour créer des interfaces."
            }
        ]

def analyze_by_prompt(results: List[Dict]) -> Dict:
    """
    Analyse les résultats par prompt
    
    Returns:
        Dict avec statistiques par prompt
    """
    # Organiser par prompt
    by_prompt_query = defaultdict(list)
    by_prompt_positive = defaultdict(list)
    
    for result in results:
        prompt_q = result.get('prompt', 'unknown')
        prompt_p = result.get('prompt_used_positive', 'unknown')
        
        by_prompt_query[prompt_q].append(result['sim_query_hardneg_q'])
        by_prompt_positive[prompt_p].append(result['sim_query_hardneg_p'])
    
    # Analyser chaque prompt
    analysis = {
        'query_based': {},
        'positive_based': {}
    }
    
    # Analyse des prompts query-based
    for prompt, sims in by_prompt_query.items():
        if sims:
            sims_array = np.array(sims)
            analysis['query_based'][prompt] = {
                'count': len(sims),
                'mean': float(np.mean(sims_array)),
                'std': float(np.std(sims_array)),
                'min': float(np.min(sims_array)),
                'max': float(np.max(sims_array)),
                'below_0.3': int(np.sum(sims_array < 0.3)),
                'in_range_0.3_0.6': int(np.sum((sims_array >= 0.3) & (sims_array <= 0.6))),
                'above_0.6': int(np.sum(sims_array > 0.6)),
                'percent_in_range': float(np.sum((sims_array >= 0.3) & (sims_array <= 0.6)) / len(sims) * 100)
            }
    
    # Analyse des prompts positive-based
    for prompt, sims in by_prompt_positive.items():
        if sims:
            sims_array = np.array(sims)
            analysis['positive_based'][prompt] = {
                'count': len(sims),
                'mean': float(np.mean(sims_array)),
                'std': float(np.std(sims_array)),
                'min': float(np.min(sims_array)),
                'max': float(np.max(sims_array)),
                'below_0.3': int(np.sum(sims_array < 0.3)),
                'in_range_0.3_0.6': int(np.sum((sims_array >= 0.3) & (sims_array <= 0.6))),
                'above_0.6': int(np.sum(sims_array > 0.6)),
                'percent_in_range': float(np.sum((sims_array >= 0.3) & (sims_array <= 0.6)) / len(sims) * 100)
            }
    
    return analysis

def print_detailed_analysis(analysis: Dict):
    """Affiche l'analyse détaillée par prompt"""
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE PAR PROMPT")
    print("="*80)
    
    # Analyse des prompts query-based
    print("\n📊 PROMPTS QUERY-BASED")
    print("-"*40)
    
    for prompt, stats in sorted(analysis['query_based'].items(), 
                                key=lambda x: x[1]['percent_in_range'], reverse=True):
        print(f"\n✦ {prompt} ({stats['count']} utilisations)")
        print(f"  Moyenne: {stats['mean']:.3f} (±{stats['std']:.3f})")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Distribution:")
        print(f"    • < 0.3: {stats['below_0.3']} ({stats['below_0.3']/stats['count']*100:.1f}%)")
        print(f"    • 0.3-0.6: {stats['in_range_0.3_0.6']} ({stats['percent_in_range']:.1f}%) ✓")
        print(f"    • > 0.6: {stats['above_0.6']} ({stats['above_0.6']/stats['count']*100:.1f}%)")
        
        # Évaluation
        if stats['percent_in_range'] >= 70:
            print(f"  🟢 Excellent - {stats['percent_in_range']:.0f}% dans la cible")
        elif stats['percent_in_range'] >= 50:
            print(f"  🟡 Bon - {stats['percent_in_range']:.0f}% dans la cible")
        else:
            print(f"  🔴 À améliorer - seulement {stats['percent_in_range']:.0f}% dans la cible")
    
    # Analyse des prompts positive-based
    print("\n\n📊 PROMPTS POSITIVE-BASED")
    print("-"*40)
    
    for prompt, stats in sorted(analysis['positive_based'].items(), 
                                key=lambda x: x[1]['percent_in_range'], reverse=True):
        print(f"\n✦ {prompt} ({stats['count']} utilisations)")
        print(f"  Moyenne: {stats['mean']:.3f} (±{stats['std']:.3f})")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Distribution:")
        print(f"    • < 0.3: {stats['below_0.3']} ({stats['below_0.3']/stats['count']*100:.1f}%)")
        print(f"    • 0.3-0.6: {stats['in_range_0.3_0.6']} ({stats['percent_in_range']:.1f}%) ✓")
        print(f"    • > 0.6: {stats['above_0.6']} ({stats['above_0.6']/stats['count']*100:.1f}%)")
        
        # Évaluation
        if stats['percent_in_range'] >= 70:
            print(f"  🟢 Excellent - {stats['percent_in_range']:.0f}% dans la cible")
        elif stats['percent_in_range'] >= 50:
            print(f"  🟡 Bon - {stats['percent_in_range']:.0f}% dans la cible")
        else:
            print(f"  🔴 À améliorer - seulement {stats['percent_in_range']:.0f}% dans la cible")

def calculate_global_stats(results: List[Dict]) -> Dict:
    """Calcule les statistiques globales"""
    
    all_query_sims = [r['sim_query_hardneg_q'] for r in results]
    all_positive_sims = [r['sim_query_hardneg_p'] for r in results]
    all_sims = all_query_sims + all_positive_sims
    
    return {
        'total_samples': len(results),
        'total_hard_negatives': len(all_sims),
        'global_mean': float(np.mean(all_sims)),
        'global_std': float(np.std(all_sims)),
        'global_below_0.3': int(np.sum(np.array(all_sims) < 0.3)),
        'global_in_range': int(np.sum((np.array(all_sims) >= 0.3) & (np.array(all_sims) <= 0.6))),
        'global_above_0.6': int(np.sum(np.array(all_sims) > 0.6)),
        'global_percent_in_range': float(np.sum((np.array(all_sims) >= 0.3) & (np.array(all_sims) <= 0.6)) / len(all_sims) * 100)
    }

def save_analysis_report(analysis: Dict, global_stats: Dict, output_file: str = "analysis_report.json"):
    """Sauvegarde le rapport d'analyse"""
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "global_statistics": global_stats,
        "prompt_analysis": analysis
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Rapport détaillé sauvegardé dans: {output_file}")

def main():
    """Fonction principale"""
    
    print("🚀 Analyse des Hard Negatives sur 100 échantillons")
    print("="*60)
    
    # 1. Charger le dataset
    dataset = load_mathlesage_dataset(n_samples=100)
    
    # 2. Sauvegarder pour le générateur
    with open('dataset_100.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("📁 Dataset sauvegardé dans: dataset_100.json")
    
    # 3. Vérifier si les résultats existent déjà
    try:
        with open('test_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✓ Résultats chargés: {len(results)} échantillons")
    except FileNotFoundError:
        print("\n⚠️ Pas de résultats trouvés.")
        print("Veuillez d'abord:")
        print("1. Générer les hard negatives: python hard_negatives_generator.py")
        print("2. Tester les embeddings: python test_hard_negatives.py")
        return
    
    # 4. Analyser par prompt
    analysis = analyze_by_prompt(results)
    
    # 5. Calculer les statistiques globales
    global_stats = calculate_global_stats(results)
    
    # 6. Afficher l'analyse détaillée
    print_detailed_analysis(analysis)
    
    # 7. Afficher le résumé global
    print("\n\n" + "="*80)
    print("RÉSUMÉ GLOBAL")
    print("="*80)
    print(f"\n📈 Sur {global_stats['total_hard_negatives']} hard negatives générés:")
    print(f"  • < 0.3: {global_stats['global_below_0.3']} ({global_stats['global_below_0.3']/global_stats['total_hard_negatives']*100:.1f}%)")
    print(f"  • 0.3-0.6: {global_stats['global_in_range']} ({global_stats['global_percent_in_range']:.1f}%) ✓")
    print(f"  • > 0.6: {global_stats['global_above_0.6']} ({global_stats['global_above_0.6']/global_stats['total_hard_negatives']*100:.1f}%)")
    print(f"\n  Moyenne globale: {global_stats['global_mean']:.3f}")
    
    # Évaluation finale
    if global_stats['global_percent_in_range'] >= 70:
        print(f"\n🎯 EXCELLENT: {global_stats['global_percent_in_range']:.0f}% des hard negatives sont dans la cible!")
    elif global_stats['global_percent_in_range'] >= 50:
        print(f"\n✅ BON: {global_stats['global_percent_in_range']:.0f}% des hard negatives sont dans la cible")
    else:
        print(f"\n⚠️ À AMÉLIORER: Seulement {global_stats['global_percent_in_range']:.0f}% dans la cible")
    
    # 8. Sauvegarder le rapport
    save_analysis_report(analysis, global_stats)
    
    print("\n" + "="*80)
    print("Analyse terminée!")

if __name__ == "__main__":
    main()