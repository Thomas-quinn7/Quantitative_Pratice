#!/usr/bin/env python3
"""
Quick Qdrant Connection Test
Run this to verify Qdrant is working properly
"""

import sys
import time

def test_qdrant_connection():
    print("="*60)
    print("  QDRANT CONNECTION TEST")
    print("="*60)
    
    # Test 1: Import qdrant_client
    print("\n[1/4] Checking qdrant_client package...")
    try:
        import qdrant_client
        print(f"✅ qdrant_client installed successfully")
    except ImportError:
        print("❌ qdrant_client not installed")
        print("   Run: pip install qdrant-client")
        return False
    
    # Test 2: Connect to Qdrant
    print("\n[2/4] Connecting to Qdrant server...")
    try:
        client = qdrant_client.QdrantClient("localhost", port=6333)
        print("✅ Connected to Qdrant at localhost:6333")
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        print("   Is Docker running? Check: docker ps | Select-String qdrant")
        return False
    
    # Test 3: Get collections
    print("\n[3/4] Retrieving collections...")
    try:
        collections = client.get_collections()
        print(f"✅ Found {len(collections.collections)} collection(s)")
        
        if collections.collections:
            for col in collections.collections:
                info = client.get_collection(col.name)
                print(f"   - {col.name}: {info.points_count} points")
        else:
            print("   ℹ️  No collections yet (this is normal for new setup)")
    except Exception as e:
        print(f"❌ Error getting collections: {e}")
        return False
    
    # Test 4: Check health
    print("\n[4/4] Checking Qdrant health...")
    try:
        # Simple health check by listing collections again
        client.get_collections()
        print("✅ Qdrant is healthy and responding")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("  ✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n📊 Qdrant Dashboard: http://localhost:6333/dashboard")
    print("\n📝 Next steps:")
    print("   1. Run: python setup_qdrant.py")
    print("   2. Run: python Tester_ML.py")
    print("\n")
    
    return True

if __name__ == "__main__":
    success = test_qdrant_connection()
    sys.exit(0 if success else 1)