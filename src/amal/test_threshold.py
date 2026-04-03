# test_threshold.py
import pandas as pd
from database import get_engine

engine = get_engine()
df = pd.read_sql(
    "SELECT bsadf_score FROM bubble_analysis WHERE ticker_id='NIFTY50'",
    engine
)
df = df.dropna()

print(f"Total rows: {len(df)}")
print(f"Min:    {df['bsadf_score'].min():.3f}")
print(f"Max:    {df['bsadf_score'].max():.3f}")
print(f"Mean:   {df['bsadf_score'].mean():.3f}")

print("\nLabel counts at different thresholds:")
for thresh in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    bubble = (df['bsadf_score'] > thresh).sum()
    crash  = (df['bsadf_score'] < -thresh).sum()
    normal = len(df) - bubble - crash
    print(f"  ±{thresh}: Bubble={bubble}({100*bubble/len(df):.1f}%) "
          f"Crash={crash}({100*crash/len(df):.1f}%) "
          f"Normal={normal}({100*normal/len(df):.1f}%)")