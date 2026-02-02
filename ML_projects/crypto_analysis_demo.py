"""
Crypto Trading Analysis with Qdrant
Demo script showing how to use the vector database for trading insights
"""

import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def connect_to_qdrant():
    """Connect to Qdrant database"""
    try:
        client = qdrant_client.QdrantClient("localhost", port=6333)
        info = client.get_collection("crypto_data")
        print(f"✅ Connected to Qdrant")
        print(f"   Total points: {info.points_count:,}")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def get_crypto_stats(client):
    """Get overview statistics of available cryptos"""
    print("\n" + "="*60)
    print("  CRYPTOCURRENCY DATABASE OVERVIEW")
    print("="*60)
    
    # Scroll through records to gather stats
    results = client.scroll(
        collection_name="crypto_data",
        limit=10000,  # Get a large sample
        with_payload=True,
        with_vectors=False
    )
    
    crypto_records = {}
    for point in results[0]:
        crypto = point.payload.get('crypto', 'Unknown')
        if crypto not in crypto_records:
            crypto_records[crypto] = []
        crypto_records[crypto].append(point.payload)
    
    print(f"\n📊 Available Cryptocurrencies: {len(crypto_records)}")
    
    # Show top cryptos by number of records
    sorted_cryptos = sorted(crypto_records.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n🏆 Top 10 by data coverage:")
    for i, (crypto, records) in enumerate(sorted_cryptos[:10], 1):
        avg_close = np.mean([r.get('close', 0) for r in records if r.get('close', 0) > 0])
        print(f"   {i:2d}. {crypto:20s} - {len(records):,} records, Avg: ${avg_close:,.2f}")
    
    return crypto_records

def find_similar_patterns(client, crypto_name, date=None, top_k=5):
    """Find cryptos with similar price patterns to a given crypto on a specific date"""
    print(f"\n" + "="*60)
    print(f"  FINDING SIMILAR PATTERNS TO {crypto_name.upper()}")
    print("="*60)
    
    # Find the reference point
    filter_conditions = Filter(
        must=[
            FieldCondition(
                key="crypto",
                match=MatchValue(value=crypto_name)
            )
        ]
    )
    
    # Get a sample point from this crypto
    results = client.scroll(
        collection_name="crypto_data",
        scroll_filter=filter_conditions,
        limit=1,
        with_payload=True,
        with_vectors=True
    )
    
    if not results[0]:
        print(f"❌ No data found for {crypto_name}")
        return
    
    reference_point = results[0][0]
    reference_vector = reference_point.vector
    
    print(f"\n🔍 Reference point:")
    print(f"   Crypto: {reference_point.payload.get('crypto')}")
    print(f"   Date: {reference_point.payload.get('date')}")
    print(f"   Close: ${reference_point.payload.get('close', 0):,.2f}")
    print(f"   Volume: {reference_point.payload.get('volume', 0):,.0f}")
    
    # Search for similar patterns
    print(f"\n🎯 Top {top_k} similar patterns:")
    
    similar = client.search(
        collection_name="crypto_data",
        query_vector=reference_vector,
        limit=top_k + 1,  # +1 because first result is usually itself
        with_payload=True
    )
    
    for i, hit in enumerate(similar[1:top_k+1], 1):  # Skip first (itself)
        payload = hit.payload
        similarity = hit.score
        print(f"\n   {i}. {payload.get('crypto', 'Unknown')} ({payload.get('date', 'N/A')})")
        print(f"      Similarity: {similarity:.4f}")
        print(f"      Close: ${payload.get('close', 0):,.2f}")
        print(f"      Volume: {payload.get('volume', 0):,.0f}")

def compare_cryptos(client, crypto1, crypto2, num_days=30):
    """Compare two cryptocurrencies across multiple days"""
    print(f"\n" + "="*60)
    print(f"  COMPARING {crypto1.upper()} vs {crypto2.upper()}")
    print("="*60)
    
    for crypto_name in [crypto1, crypto2]:
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="crypto",
                    match=MatchValue(value=crypto_name)
                )
            ]
        )
        
        results = client.scroll(
            collection_name="crypto_data",
            scroll_filter=filter_conditions,
            limit=num_days,
            with_payload=True,
            with_vectors=False
        )
        
        if results[0]:
            prices = [p.payload.get('close', 0) for p in results[0]]
            volumes = [p.payload.get('volume', 0) for p in results[0]]
            
            print(f"\n📈 {crypto_name}:")
            print(f"   Records found: {len(prices)}")
            if prices and any(p > 0 for p in prices):
                valid_prices = [p for p in prices if p > 0]
                print(f"   Avg Price: ${np.mean(valid_prices):,.2f}")
                print(f"   Price Range: ${min(valid_prices):,.2f} - ${max(valid_prices):,.2f}")
                print(f"   Avg Volume: {np.mean([v for v in volumes if v > 0]):,.0f}")

def find_high_volatility_days(client, min_volatility_ratio=0.05, limit=10):
    """Find days with highest volatility across all cryptos"""
    print(f"\n" + "="*60)
    print(f"  HIGH VOLATILITY DAYS (>5% daily range)")
    print("="*60)
    
    # Scroll through data
    results = client.scroll(
        collection_name="crypto_data",
        limit=5000,
        with_payload=True,
        with_vectors=False
    )
    
    high_vol_days = []
    for point in results[0]:
        payload = point.payload
        high = payload.get('high', 0)
        low = payload.get('low', 0)
        open_price = payload.get('open', 0)
        
        if open_price > 0:
            volatility = (high - low) / open_price
            if volatility > min_volatility_ratio:
                high_vol_days.append({
                    'crypto': payload.get('crypto'),
                    'date': payload.get('date'),
                    'volatility': volatility,
                    'close': payload.get('close', 0),
                    'volume': payload.get('volume', 0)
                })
    
    # Sort by volatility
    high_vol_days.sort(key=lambda x: x['volatility'], reverse=True)
    
    print(f"\n🔥 Top {limit} most volatile days:")
    for i, day in enumerate(high_vol_days[:limit], 1):
        print(f"\n   {i:2d}. {day['crypto']} ({day['date']})")
        print(f"       Volatility: {day['volatility']*100:.2f}%")
        print(f"       Close: ${day['close']:,.2f}")
        print(f"       Volume: {day['volume']:,.0f}")

def get_crypto_by_name(client, crypto_name, limit=10):
    """Get recent data for a specific cryptocurrency"""
    print(f"\n" + "="*60)
    print(f"  RECENT DATA FOR {crypto_name.upper()}")
    print("="*60)
    
    filter_conditions = Filter(
        must=[
            FieldCondition(
                key="crypto",
                match=MatchValue(value=crypto_name)
            )
        ]
    )
    
    results = client.scroll(
        collection_name="crypto_data",
        scroll_filter=filter_conditions,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    if not results[0]:
        print(f"❌ No data found for {crypto_name}")
        return
    
    print(f"\n📊 Last {len(results[0])} records:")
    for i, point in enumerate(results[0], 1):
        p = point.payload
        print(f"\n   {i}. Date: {p.get('date', 'N/A')}")
        print(f"      Open: ${p.get('open', 0):,.2f}")
        print(f"      High: ${p.get('high', 0):,.2f}")
        print(f"      Low: ${p.get('low', 0):,.2f}")
        print(f"      Close: ${p.get('close', 0):,.2f}")
        print(f"      Volume: {p.get('volume', 0):,.0f}")

def main():
    """Run demo queries"""
    print("\n" + "="*60)
    print("  QDRANT CRYPTO TRADING ANALYSIS - DEMO")
    print("="*60)
    
    # Connect
    client = connect_to_qdrant()
    if not client:
        return
    
    # Get overview
    crypto_records = get_crypto_stats(client)
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("  ANALYSIS OPTIONS")
        print("="*60)
        print("\n1. View specific crypto data")
        print("2. Find similar patterns")
        print("3. Compare two cryptos")
        print("4. Find high volatility days")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            crypto = input("Enter crypto name (e.g., Bitcoin, Ethereum): ").strip().title()
            get_crypto_by_name(client, crypto, limit=10)
        
        elif choice == "2":
            crypto = input("Enter crypto name to find similar patterns: ").strip().title()
            find_similar_patterns(client, crypto, top_k=5)
        
        elif choice == "3":
            crypto1 = input("Enter first crypto name: ").strip().title()
            crypto2 = input("Enter second crypto name: ").strip().title()
            compare_cryptos(client, crypto1, crypto2, num_days=30)
        
        elif choice == "4":
            find_high_volatility_days(client, min_volatility_ratio=0.05, limit=10)
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()