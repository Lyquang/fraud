import pandas as pd
import os
from sklearn.model_selection import train_test_split

DEBUG_FRAC = 0.005  # 10%
DATA_PATH = 'data/train_merged.csv'
DEBUG_PATH = 'data/test_debug.csv'

print("🔧 Creating Debug Dataset")
print("="*60)

os.makedirs('data', exist_ok=True)

# Load dataset
df_full = pd.read_csv(DATA_PATH)
print(f"📊 Original dataset: {df_full.shape}")
print(f"   Fraud rate: {df_full['isFraud'].mean()*100:.2f}%")

# Split stratified by 'isFraud'
_, df_debug = train_test_split(
    df_full,
    test_size=DEBUG_FRAC,
    random_state=42,
    stratify=df_full['isFraud']
)

print(f"\n✅ Debug dataset created: {df_debug.shape}")
print(f"   Fraud rate: {df_debug['isFraud'].mean()*100:.2f}%")

# Save
df_debug.to_csv(DEBUG_PATH, index=False)
size_mb = os.path.getsize(DEBUG_PATH)/(1024*1024)
print(f"💾 Saved to: {DEBUG_PATH} ({size_mb:.2f} MB)")

# Compare distribution
print("\n📊 Class Distribution:")
print(f"{'Dataset':<15} {'Normal':<10} {'Fraud':<10} {'Fraud %':<8}")
print("-"*45)
print(f"{'Original':<15} {(df_full['isFraud']==0).sum():<10,} {(df_full['isFraud']==1).sum():<10,} {df_full['isFraud'].mean()*100:<8.2f}")
print(f"{'Debug':<15} {(df_debug['isFraud']==0).sum():<10,} {(df_debug['isFraud']==1).sum():<10,} {df_debug['isFraud'].mean()*100:<8.2f}")

print("\n✅ Debug dataset ready for testing!")
