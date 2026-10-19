import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
proc = psutil.Process(os.getpid())

dtypes = {'sensor_id': 'category', 'measurement': 'float32', 'unit': 'category', 'quality': 'float32'}

mem0 = proc.memory_info().rss / 1024 ** 2
t0 = time.perf_counter()
df = pd.read_csv('data/sensor_data.csv', dtype=dtypes, parse_dates=['timestamp'])
t_read = time.perf_counter() - t0
mem1 = proc.memory_info().rss / 1024 ** 2

print("=== Задание 1. Загрузка и обзор ===")
print(df.head())
print(df.info())
print(df.describe())
print("Уникальных датчиков:", df['sensor_id'].nunique())
print("Пропусков по столбцам:")
print(df.isna().sum())
print(f"Время чтения CSV: {t_read:.2f} с")
print(f"Память процесса: до {mem0:.0f} МБ, после чтения {mem1:.0f} МБ")
print(f"Память DataFrame: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} МБ")

print("\n=== Задание 2. Предобработка ===")
n_before = len(df)
df = df.sort_values(['sensor_id', 'timestamp'])
df = df[df['quality'] >= 0.8].copy()
print("Строк до фильтрации:", n_before)
print("Строк после фильтрации quality >= 0.8:", len(df))
print("Удалено:", n_before - len(df))
print("Пропусков measurement до заполнения:", int(df['measurement'].isna().sum()))
df['measurement'] = df.groupby('sensor_id', observed=True)['measurement'].ffill()
print("Пропусков measurement после ffill:", int(df['measurement'].isna().sum()))
df['hour'] = df['timestamp'].dt.floor('1h')
print(df.head())

print("\n=== Задание 3. Агрегация ===")
t0 = time.perf_counter()
hourly = df.groupby(['sensor_id', 'hour'], observed=True)['measurement'].mean().reset_index()
t_agg = time.perf_counter() - t0
print(f"Pandas aggregation time: {t_agg:.2f} s")
print(hourly.head())
print("Размер результата:", hourly.shape)

indexed = df.set_index('timestamp')
t0 = time.perf_counter()
hourly_rs = indexed.groupby('sensor_id', observed=True).resample('1h')['measurement'].mean().reset_index()
t_rs = time.perf_counter() - t0
print(f"Pandas resample time: {t_rs:.2f} s")
print("Размер результата resample:", hourly_rs.shape)

hourly.to_parquet('results/hourly_pandas.parquet', index=False)
print("Сохранено: results/hourly_pandas.parquet")

print("\n=== Скользящее среднее (5 минут), датчик s01 ===")
s = indexed[indexed['sensor_id'] == 's01'].copy()
s['rolling_5min'] = s['measurement'].rolling('5min').mean()
print(s[['measurement', 'rolling_5min']].head(8))

fig, ax = plt.subplots(figsize=(10, 4))
part = s.iloc[:1440 * 2]
ax.plot(part.index, part['measurement'], alpha=0.4, label='measurement')
ax.plot(part.index, part['rolling_5min'], label='rolling 5 min')
ax.set_title('Датчик s01: измерения и скользящее среднее (2 суток)')
ax.set_xlabel('Время')
ax.set_ylabel('Температура, °C')
ax.legend()
plt.tight_layout()
plt.savefig('results/rolling_s01.png', dpi=150)
plt.show()

print("\nСреднее по датчикам:")
print(df.groupby('sensor_id', observed=True)['measurement'].agg(['mean', 'min', 'max']).round(2))
