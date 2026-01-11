import qdrant_client as QdrantClient
import numpy as np
import kagglehub
import pandas as pd
import os
import csv

path = kagglehub.dataset_download("adrianjuliusaluoch/crypto-and-stock-market-data-for-financial-analysis")

try:
    crypto_df = pd.read_csv(os.path.join(path, "cryptocurrency.csv"))
    print(f"Successfully loaded crypto_data.csv with shape: {crypto_df.shape}")
except FileNotFoundError:
    print("crypto_data.csv not found in the dataset")
    print("Available files:", os.listdir(path))
except Exception as e:
    print(f"Error loading crypto_data.csv: {e}")

try:
    client = QdrantClient.QdrantClient("localhost", port=6333)
    print("✅ Connected to local Qdrant server!")
    collections = client.get_collections()
    print("Collections:", collections)
except Exception as e:
    print(f"❌ Local server connection failed: {e}")
    print("This means Docker/Qdrant server is not running")