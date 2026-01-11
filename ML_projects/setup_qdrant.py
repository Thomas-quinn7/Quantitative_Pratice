import qdrant_client as qdrant_module
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
import numpy as np
import kagglehub
import os
import time

def wait_for_qdrant(client, max_retries=30):
    """Wait for Qdrant to be ready"""
    for i in range(max_retries):
        try:
            client.get_collections()
            return True
        except Exception as e:
            print(f"Waiting for Qdrant to start... ({i+1}/{max_retries})")
            time.sleep(2)
    return False

def setup_qdrant_database(force_recreate=False):
    """Initialize Qdrant with crypto data for new users
    
    Args:
        force_recreate: If True, delete existing collection and recreate
    """

    result = os.system("docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant 2>/dev/null")
    if result != 0:
        print("ℹ️  Qdrant container might already be running")
    
    client = qdrant_module.QdrantClient("localhost", port=6333)

    if not wait_for_qdrant(client):
        print("❌ Could not connect to Qdrant")
        return None
    
    collections = client.get_collections()
    collection_exists = any(col.name == "crypto_data" for col in collections.collections)
    
    if collection_exists and not force_recreate:
        print("✅ Crypto data collection already exists in Qdrant")
        collection_info = client.get_collection("crypto_data")
        print(f"   Collection has {collection_info.points_count} points")
        return client
    
    if collection_exists and force_recreate:
        print("🗑️  Deleting existing collection to recreate...")
        client.delete_collection("crypto_data")
        time.sleep(2) 
    
    path = kagglehub.dataset_download("adrianjuliusaluoch/crypto-and-stock-market-data-for-financial-analysis")
    crypto_df = pd.read_csv(os.path.join(path, "cryptocurrency.csv"))
    print(f"✅ Loaded {len(crypto_df)} crypto records")
    print(f"   Columns: {list(crypto_df.columns)}")
    
    try:
        client.create_collection(
            collection_name="crypto_data",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        print("Created 'crypto_data' collection")
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None
    
    # TODO: Add your data processing and insertion logic here
    # For now, just create empty collection
    print("📝 TODO: Add data insertion logic")
    print("   - Process crypto_df into vectors")
    print("   - Insert points into collection")
    
    print("✅ Qdrant database initialized successfully")
    return client

def check_qdrant_status():
    """Check current status of Qdrant database"""
    try:
        client = qdrant_module.QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        
        print("📊 Qdrant Status:")
        print(f"   Total collections: {len(collections.collections)}")
        
        for col in collections.collections:
            info = client.get_collection(col.name)
            print(f"   - {col.name}: {info.points_count} points")
            
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("🔄 Force recreating database...")
        setup_qdrant_database(force_recreate=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--status":
        check_qdrant_status()
    else:
        setup_qdrant_database(force_recreate=False)