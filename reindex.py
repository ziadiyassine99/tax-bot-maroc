"""
Script de réindexation des documents dans Qdrant Cloud.
Réindexe tous les modules ou un module spécifique.

Usage:
    python reindex.py              # Réindexe tous les modules
    python reindex.py cnss         # Réindexe uniquement le module CNSS
    python reindex.py regulation   # Réindexe uniquement le module Régulation
"""

import sys
import os
import time

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODULES, get_module_config
from document_loader import DocumentProcessor, PDFLoadError
from vector_store import VectorStoreManager, create_vector_store_manager


def reindex_module(module_id: str) -> dict:
    """
    Réindexe un module spécifique.
    
    Args:
        module_id: L'identifiant du module (cnss, regulation, travail, conventions)
        
    Returns:
        dict avec les résultats de l'indexation
    """
    print(f"\n{'='*60}")
    print(f"📚 Réindexation du module: {module_id.upper()}")
    print(f"{'='*60}")
    
    try:
        module_config = get_module_config(module_id)
        print(f"📁 Dossier source: {module_config['pdf_path']}")
        print(f"🗄️  Collection: {module_config['collection_name']}")
        
        # 1. Charger les documents
        print(f"\n🔄 Chargement des documents...")
        start_time = time.time()
        
        doc_processor = DocumentProcessor(pdf_path=module_config["pdf_path"])
        documents = doc_processor.load_and_split()
        
        load_time = time.time() - start_time
        print(f"✅ {len(documents)} chunks créés en {load_time:.1f}s")
        
        # 2. Créer le vector store
        print(f"\n🔄 Création du vector store dans Qdrant Cloud...")
        start_time = time.time()
        
        vs_manager = create_vector_store_manager(module_config)
        
        # Afficher le mode de connexion
        mode = vs_manager.get_connection_mode()
        print(f"📡 Mode de connexion: {mode.upper()}")
        
        # Créer/recréer le vector store (supprime l'existant)
        vs_manager.create_vector_store(documents)
        
        index_time = time.time() - start_time
        print(f"✅ Indexation terminée en {index_time:.1f}s")
        
        # 3. Vérifier
        info = vs_manager.get_collection_info()
        print(f"\n📊 Statistiques de la collection:")
        print(f"   - Nom: {info.get('name', 'N/A')}")
        print(f"   - Vecteurs: {info.get('vectors_count', 'N/A')}")
        print(f"   - Points: {info.get('points_count', 'N/A')}")
        
        return {
            "module": module_id,
            "success": True,
            "chunks": len(documents),
            "load_time": load_time,
            "index_time": index_time,
            "collection_info": info
        }
        
    except PDFLoadError as e:
        print(f"❌ Erreur de chargement: {e}")
        return {"module": module_id, "success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {"module": module_id, "success": False, "error": str(e)}


def reindex_all():
    """Réindexe tous les modules."""
    print("\n" + "="*60)
    print("🚀 RÉINDEXATION COMPLÈTE DE TOUS LES MODULES")
    print("="*60)
    
    results = []
    total_start = time.time()
    
    for module_id in MODULES.keys():
        result = reindex_module(module_id)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Résumé
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DE L'INDEXATION")
    print("="*60)
    
    success_count = sum(1 for r in results if r.get("success"))
    total_chunks = sum(r.get("chunks", 0) for r in results if r.get("success"))
    
    for r in results:
        status = "✅" if r.get("success") else "❌"
        chunks = r.get("chunks", 0)
        print(f"  {status} {r['module'].upper()}: {chunks} chunks")
    
    print(f"\n⏱️  Temps total: {total_time:.1f}s")
    print(f"📊 Modules réussis: {success_count}/{len(results)}")
    print(f"📚 Total chunks indexés: {total_chunks}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        module_id = sys.argv[1].lower()
        if module_id in MODULES:
            reindex_module(module_id)
        else:
            print(f"❌ Module inconnu: {module_id}")
            print(f"   Modules disponibles: {list(MODULES.keys())}")
            sys.exit(1)
    else:
        reindex_all()

