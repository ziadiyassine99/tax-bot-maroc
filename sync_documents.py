"""
Document synchronization script for incremental indexing.
Run this script to detect and index only changed documents.

Usage:
    python sync_documents.py                    # Sync all modules
    python sync_documents.py --module cgi       # Sync specific module
    python sync_documents.py --force            # Force full re-index
    python sync_documents.py --dry-run          # Show what would change
"""

import argparse
import os
import sys
from typing import List, Dict, Any

from config import MODULES, get_module_config
from document_loader import DocumentProcessor
from document_tracker import DocumentTracker, get_pdf_files_in_directory
from vector_store import create_vector_store_manager


def get_module_documents_dir(module_config: Dict[str, Any]) -> str:
    """Get the directory containing documents for a module."""
    pdf_path = module_config["pdf_path"]
    return os.path.dirname(pdf_path)


def sync_module(
    module_id: str,
    tracker: DocumentTracker,
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Synchronize documents for a single module.
    
    Args:
        module_id: Module identifier
        tracker: Document tracker instance
        force: Force full re-index
        dry_run: Only show what would change
        
    Returns:
        Dict with sync statistics
    """
    module_config = get_module_config(module_id)
    collection_name = module_config["collection_name"]
    docs_dir = get_module_documents_dir(module_config)
    
    print(f"\n{'='*60}")
    print(f"Module: {module_config['name']} ({module_id})")
    print(f"Directory: {docs_dir}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}")
    
    # Get current PDF files
    current_files = get_pdf_files_in_directory(docs_dir)
    print(f"\nFichiers trouvés: {len(current_files)}")
    for f in current_files:
        print(f"  - {os.path.basename(f)}")
    
    # Detect changes
    added, modified, deleted = tracker.detect_changes(collection_name, current_files)
    
    print(f"\nChangements détectés:")
    print(f"  + Nouveaux: {len(added)}")
    for f in added:
        print(f"    - {os.path.basename(f)}")
    print(f"  ~ Modifiés: {len(modified)}")
    for f in modified:
        print(f"    - {os.path.basename(f)}")
    print(f"  - Supprimés: {len(deleted)}")
    for f in deleted:
        print(f"    - {os.path.basename(f)}")
    
    # If force, treat all as new
    if force:
        print("\n⚠️  Mode FORCE: réindexation complète")
        added = current_files
        modified = []
        deleted = list(tracker.get_collection_docs(collection_name).keys())
    
    # Check if any changes
    if not added and not modified and not deleted:
        print("\n✅ Aucun changement détecté. Base à jour.")
        return {"status": "up_to_date", "changes": 0}
    
    if dry_run:
        print("\n🔍 Mode DRY-RUN: aucune modification effectuée")
        return {
            "status": "dry_run",
            "would_add": len(added),
            "would_modify": len(modified),
            "would_delete": len(deleted)
        }
    
    # Process documents
    print("\n📄 Traitement des documents...")
    
    vs_manager = create_vector_store_manager(module_config)
    
    # Check if vector store is in memory mode
    if vs_manager.is_memory_mode():
        print("⚠️  Mode mémoire détecté (pas de Docker Qdrant)")
        print("   L'indexation incrémentale n'est pas supportée.")
        print("   Utilisation de l'indexation complète.")
        
        # Full rebuild in memory mode
        all_files = current_files
        all_docs = []
        for file_path in all_files:
            processor = DocumentProcessor(pdf_path=file_path)
            docs = processor.load_and_split()
            all_docs.extend(docs)
            tracker.update_document(collection_name, file_path, len(docs))
        
        vs_manager.create_vector_store(all_docs)
        print(f"\n✅ Base recréée avec {len(all_docs)} chunks")
        return {"status": "rebuilt", "chunks": len(all_docs)}
    
    # Docker mode: incremental sync
    added_docs = []
    modified_docs = []
    
    # Process new files
    for file_path in added:
        print(f"  + Indexation: {os.path.basename(file_path)}")
        processor = DocumentProcessor(pdf_path=file_path)
        docs = processor.load_and_split()
        added_docs.extend(docs)
        tracker.update_document(collection_name, file_path, len(docs))
    
    # Process modified files
    for file_path in modified:
        print(f"  ~ Mise à jour: {os.path.basename(file_path)}")
        processor = DocumentProcessor(pdf_path=file_path)
        docs = processor.load_and_split()
        modified_docs.extend(docs)
        tracker.update_document(collection_name, file_path, len(docs))
    
    # Remove deleted files from tracker
    for file_path in deleted:
        print(f"  - Suppression: {os.path.basename(file_path)}")
        tracker.remove_document(collection_name, file_path)
    
    # Ensure vector store is loaded or created
    if not vs_manager.vector_store_exists() and not added_docs and not modified_docs:
        print("\n⚠️  Aucune base vectorielle existante et aucun document à ajouter.")
        return {"status": "no_documents", "changes": 0}
    
    # Load existing store if it exists
    if vs_manager.vector_store_exists():
        vs_manager.load_vector_store()
    
    # Perform sync
    stats = vs_manager.sync_documents(
        added_docs=added_docs,
        modified_sources=[f for f in modified],
        modified_docs=modified_docs,
        deleted_sources=deleted
    )
    
    print(f"\n✅ Synchronisation terminée:")
    print(f"   Ajoutés: {stats['added']} chunks")
    print(f"   Mis à jour: {stats['updated']} fichiers")
    print(f"   Supprimés: {stats['deleted']} fichiers")
    
    # Show tracker stats
    tracker_stats = tracker.get_stats(collection_name)
    print(f"\n📊 État de la base:")
    print(f"   Fichiers indexés: {tracker_stats['tracked_files']}")
    print(f"   Total chunks: {tracker_stats['total_chunks']}")
    
    return {"status": "synced", **stats}


def main():
    parser = argparse.ArgumentParser(description="Synchroniser les documents pour l'indexation")
    parser.add_argument("--module", "-m", help="Module spécifique à synchroniser (cgi, cdt)")
    parser.add_argument("--force", "-f", action="store_true", help="Forcer la réindexation complète")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Afficher les changements sans les appliquer")
    
    args = parser.parse_args()
    
    print("🔄 Synchronisation des documents")
    print(f"   Force: {args.force}")
    print(f"   Dry-run: {args.dry_run}")
    
    tracker = DocumentTracker()
    
    # Determine which modules to sync
    if args.module:
        if args.module not in MODULES:
            print(f"❌ Module inconnu: {args.module}")
            print(f"   Modules disponibles: {list(MODULES.keys())}")
            sys.exit(1)
        modules_to_sync = [args.module]
    else:
        modules_to_sync = list(MODULES.keys())
    
    # Sync each module
    results = {}
    for module_id in modules_to_sync:
        try:
            results[module_id] = sync_module(
                module_id, 
                tracker, 
                force=args.force,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"\n❌ Erreur lors de la synchronisation de {module_id}: {e}")
            results[module_id] = {"status": "error", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    for module_id, result in results.items():
        status_emoji = "✅" if result["status"] in ["synced", "up_to_date", "rebuilt"] else "❌"
        print(f"{status_emoji} {module_id}: {result['status']}")
    
    return 0 if all(r["status"] != "error" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

