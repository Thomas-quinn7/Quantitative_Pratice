import qdrant_client as qdrant_module
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd
import numpy as np
import kagglehub
import os
import time
from datetime import datetime

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

def create_crypto_vector(row, feature_columns, vector_size=128):
    """Create a normalized vector from crypto data row"""
    features = []
    for col in feature_columns:
        if col in row.index and pd.notna(row[col]):
            try:
                features.append(float(row[col]))
            except (ValueError, TypeError):
                features.append(0.0)
        else:
            features.append(0.0)
    
    # Create vector and pad to desired size
    vector = np.zeros(vector_size)
    vector[:min(len(features), vector_size)] = features[:vector_size]
    
    # Add derived features if space available
    if len(features) < vector_size:
        idx = len(features)
        
        # Daily return (Close - Open) / Open
        if 'Close' in row.index and 'Open' in row.index and idx < vector_size:
            try:
                close = float(row['Close'])
                open_price = float(row['Open'])
                if open_price > 0:
                    vector[idx] = (close - open_price) / open_price
                    idx += 1
            except (ValueError, TypeError):
                pass
        
        # Daily volatility (High - Low) / Open
        if 'High' in row.index and 'Low' in row.index and 'Open' in row.index and idx < vector_size:
            try:
                high = float(row['High'])
                low = float(row['Low'])
                open_price = float(row['Open'])
                if open_price > 0:
                    vector[idx] = (high - low) / open_price
                    idx += 1
            except (ValueError, TypeError):
                pass
        
        # Volume normalized by close price
        if 'Volume' in row.index and 'Close' in row.index and idx < vector_size:
            try:
                volume = float(row['Volume'])
                close = float(row['Close'])
                if close > 0:
                    vector[idx] = volume / close
                    idx += 1
            except (ValueError, TypeError):
                pass
    
    # L2 normalization
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()

def setup_qdrant_database(force_recreate=False):
    """Initialize Qdrant with Top 50 crypto data from multiple CSV files"""
    print("\n" + "="*60)
    print("  TOP 50 CRYPTO DATA SETUP FOR QDRANT")
    print("="*60 + "\n")

    # Connect to Qdrant
    client = qdrant_module.QdrantClient("localhost", port=6333)

    if not wait_for_qdrant(client):
        print("❌ Could not connect to Qdrant")
        print("   Make sure Docker container is running: docker ps")
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
    
    # Download dataset
    print("📥 Downloading Top 50 Cryptocurrency dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("dhrubangtalukdar/top-50-cryptocurrency-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} cryptocurrency files")
        
        if not csv_files:
            print("❌ No CSV files found in dataset")
            return None
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None
    
    # Create collection
    print("\n📊 Creating 'crypto_data' collection...")
    try:
        client.create_collection(
            collection_name="crypto_data",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        print("✅ Collection created successfully")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return None
    
    # Process all CSV files
    print(f"\n📝 Processing {len(csv_files)} cryptocurrencies...")
    
    all_points = []
    point_id = 0
    total_records = 0
    skipped_files = 0
    
    for csv_file in sorted(csv_files):
        try:
            # Extract crypto name from filename (e.g., "bitcoin.csv" -> "Bitcoin")
            crypto_name = csv_file.replace('.csv', '').replace('_', ' ').title()
            
            # Read CSV
            df = pd.read_csv(os.path.join(path, csv_file))
            
            # Determine feature columns (should be: Date, Open, High, Low, Close, Volume)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns = [col for col in numeric_cols if col in df.columns]
            
            # Process each row (each day of data)
            for idx, row in df.iterrows():
                try:
                    # Create vector
                    vector = create_crypto_vector(row, feature_columns)
                    
                    # Create payload
                    payload = {
                        "crypto": crypto_name,
                        "symbol": crypto_name.replace(' ', '').upper()[:10],  # Approximate symbol
                        "date": str(row.get('Date', '')),
                    }
                    
                    # Add OHLCV data
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in row.index and pd.notna(row[col]):
                            try:
                                payload[col.lower()] = float(row[col])
                            except (ValueError, TypeError):
                                pass
                    
                    # Create point
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                    all_points.append(point)
                    point_id += 1
                    
                except Exception as e:
                    continue
            
            total_records += len(df)
            print(f"   ✓ {crypto_name}: {len(df)} records", end='\r')
            
        except Exception as e:
            skipped_files += 1
            print(f"\n   ⚠️  Skipped {csv_file}: {e}")
            continue
    
    print(f"\n\n📤 Inserting {len(all_points)} data points into Qdrant...")
    
    # Insert in batches for efficiency
    batch_size = 100
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i+batch_size]
        try:
            client.upsert(collection_name="crypto_data", points=batch)
            print(f"   Inserted {min(i+batch_size, len(all_points))}/{len(all_points)} points...", end='\r')
        except Exception as e:
            print(f"\n   ⚠️  Error inserting batch at {i}: {e}")
    
    print(f"\n\n✅ Data insertion complete!")
    print(f"   Cryptocurrencies processed: {len(csv_files) - skipped_files}")
    print(f"   Total data points inserted: {len(all_points)}")
    if skipped_files > 0:
        print(f"   Files skipped: {skipped_files}")
    
    # Verify collection
    collection_info = client.get_collection("crypto_data")
    print(f"\n📊 Collection Info:")
    print(f"   Collection: crypto_data")
    print(f"   Points: {collection_info.points_count}")
    print(f"   Vector size: {collection_info.config.params.vectors.size}")
    print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
    
    # Show sample data from different cryptos
    if collection_info.points_count > 0:
        print("\n📋 Sample data (recent records from different cryptos):")
        
        # Get unique cryptos
        unique_cryptos = set()
        results = client.scroll(
            collection_name="crypto_data",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        for point in results[0]:
            crypto = point.payload.get('crypto', 'Unknown')
            if crypto not in unique_cryptos and len(unique_cryptos) < 5:
                unique_cryptos.add(crypto)
                date = point.payload.get('date', 'N/A')
                close = point.payload.get('close', 0)
                volume = point.payload.get('volume', 0)
                print(f"   • {crypto} ({date})")
                print(f"     Close: ${close:,.2f}, Volume: {volume:,.0f}")
    
    print("\n✅ Qdrant database initialized successfully!")
    print(f"   Dashboard: http://localhost:6333/dashboard")
    print(f"\n💡 You can now:")
    print(f"   - Search for similar price patterns across cryptos")
    print(f"   - Find cryptos with similar volatility profiles")
    print(f"   - Build trading strategies based on historical patterns")
    
    return client

def check_qdrant_status():
    """Check current status of Qdrant database"""
    try:
        client = qdrant_module.QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        
        print("\n📊 Qdrant Status:")
        print(f"   Total collections: {len(collections.collections)}")
        
        for col in collections.collections:
            info = client.get_collection(col.name)
            print(f"\n   Collection: {col.name}")
            print(f"   - Points: {info.points_count}")
            print(f"   - Vector size: {info.config.params.vectors.size}")
            print(f"   - Distance: {info.config.params.vectors.distance}")
            
            # Count unique cryptos
            if info.points_count > 0:
                results = client.scroll(
                    collection_name=col.name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )
                
                cryptos = set()
                for point in results[0]:
                    cryptos.add(point.payload.get('crypto', 'Unknown'))
                
                print(f"   - Unique cryptocurrencies: {len(cryptos)}")
                print(f"   - Examples: {', '.join(list(cryptos)[:5])}")
        
        print()
            
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