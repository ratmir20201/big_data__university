import os
import numpy as np
import pandas as pd

os.makedirs('data', exist_ok=True)

rng = np.random.default_rng(42)

n_sensors = 10
n_minutes = 100000
sensor_ids = [f's{i:02d}' for i in range(1, n_sensors + 1)]
base = rng.uniform(5, 25, n_sensors)

times = pd.date_range('2024-01-01 00:00:00', periods=n_minutes, freq='min')
minutes = np.arange(n_minutes)

frames = []
for i, sid in enumerate(sensor_ids):
    daily = 3.0 * np.sin(2 * np.pi * minutes / 1440 + i)
    noise = rng.normal(0, 0.4, n_minutes)
    measurement = base[i] + daily + noise
    quality = np.clip(rng.beta(12, 1.2, n_minutes), 0, 1)
    frames.append(pd.DataFrame({
        'sensor_id': sid,
        'timestamp': times,
        'measurement': np.round(measurement, 2),
        'unit': 'C',
        'quality': np.round(quality, 2),
    }))

df = pd.concat(frames, ignore_index=True)

missing_idx = rng.choice(len(df), size=int(len(df) * 0.01), replace=False)
df.loc[missing_idx, 'measurement'] = np.nan

df = df.sort_values(['timestamp', 'sensor_id']).reset_index(drop=True)

df.to_csv('data/sensor_data.csv', index=False)

n_parts = 4
size = -(-len(df) // n_parts)
for k in range(n_parts):
    df.iloc[k * size:(k + 1) * size].to_csv(f'data/sensor_data-{k:03d}.csv', index=False)

print("Строк:", len(df))
print("Датчиков:", df['sensor_id'].nunique())
print("Пропусков measurement:", int(df['measurement'].isna().sum()))
print("Файлов-частей:", n_parts)
