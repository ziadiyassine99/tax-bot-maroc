"""
Script simple pour supprimer les collections Qdrant.
L'app Streamlit les reconstruira automatiquement au prochain chargement.
"""
from qdrant_client import QdrantClient
from config import get_qdrant_config, MODULES

# Connexion à Qdrant Cloud
config = get_qdrant_config()
client = QdrantClient(url=config["url"], api_key=config["api_key"])

print("Collections actuelles:")
for col in client.get_collections().collections:
    print(f"  - {col.name}")

print("\nSuppression des collections...")
for module_id, module_config in MODULES.items():
    collection_name = module_config["collection_name"]
    try:
        client.delete_collection(collection_name)
        print(f"  [OK] {collection_name} supprimee")
    except Exception as e:
        print(f"  [--] {collection_name}: {e}")

print("\nCollections restantes:")
for col in client.get_collections().collections:
    print(f"  - {col.name}")

print("\n=> Relancez l'app Streamlit pour reconstruire les index automatiquement")

