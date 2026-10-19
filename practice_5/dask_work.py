import os
import time
import psutil
import pandas as pd
import dask.dataframe as dd

os.makedirs('results', exist_ok=True)
proc = psutil.Process(os.getpid())

dtypes = {'sensor_id': 'object', 'measurement': 'float32', 'unit': 'object', 'quality': 'float32'}

mem0 = proc.memory_info().rss / 1024 ** 2

ddf = dd.read_csv('data/sensor_data-*.csv', dtype=dtypes, parse_dates=['timestamp'], assume_missing=True)

print("=== Задание 4. Dask ===")
print(ddf)
print("Количество партиций:", ddf.npartitions)
print(ddf.dtypes)

n_total = len(ddf)
print("Всего строк:", n_total)

ddf = ddf[ddf['quality'] >= 0.8]
ddf['hour'] = ddf['timestamp'].dt.floor('1h')
n_filtered = len(ddf)
print("Строк после фильтрации quality >= 0.8:", n_filtered)

hourly_ddf = ddf.groupby(['sensor_id', 'hour'])['measurement'].mean()

t0 = time.perf_counter()
hourly_res = hourly_ddf.compute()
t_dask = time.perf_counter() - t0
mem1 = proc.memory_info().rss / 1024 ** 2

print(f"Dask aggregation time (compute): {t_dask:.2f} s")
print(f"Память процесса: до {mem0:.0f} МБ, после compute {mem1:.0f} МБ")

hourly_res = hourly_res.reset_index().sort_values(['sensor_id', 'hour']).reset_index(drop=True)
print(hourly_res.head())
print("Размер результата:", hourly_res.shape)

print("\n=== Сравнение с Pandas ===")
dtypes_pd = {'sensor_id': 'category', 'measurement': 'float32', 'unit': 'category', 'quality': 'float32'}
t0 = time.perf_counter()
pdf = pd.read_csv('data/sensor_data.csv', dtype=dtypes_pd, parse_dates=['timestamp'])
pdf = pdf[pdf['quality'] >= 0.8].copy()
pdf['hour'] = pdf['timestamp'].dt.floor('1h')
hourly_pd = pdf.groupby(['sensor_id', 'hour'], observed=True)['measurement'].mean().reset_index()
t_pandas_total = time.perf_counter() - t0
print(f"Pandas (чтение + фильтрация + агрегация): {t_pandas_total:.2f} s")

t0 = time.perf_counter()
ddf2 = dd.read_csv('data/sensor_data-*.csv', dtype=dtypes, parse_dates=['timestamp'], assume_missing=True)
ddf2 = ddf2[ddf2['quality'] >= 0.8]
ddf2['hour'] = ddf2['timestamp'].dt.floor('1h')
hourly_dask_total = ddf2.groupby(['sensor_id', 'hour'])['measurement'].mean().compute().reset_index()
t_dask_total = time.perf_counter() - t0
print(f"Dask (чтение + фильтрация + агрегация): {t_dask_total:.2f} s")

a = hourly_pd.sort_values(['sensor_id', 'hour']).reset_index(drop=True)
b = hourly_dask_total.sort_values(['sensor_id', 'hour']).reset_index(drop=True)
print("Размеры совпадают:", a.shape == b.shape)
print("Макс. расхождение средних:", float((a['measurement'].astype('float64') - b['measurement'].astype('float64')).abs().max()))

print("\n=== Задание 5. Сохранение в parquet ===")
out = ddf2.groupby(['sensor_id', 'hour'])['measurement'].mean().reset_index()
out.to_parquet('results/hourly_dask', write_index=False)
print("Файлы:", sorted(os.listdir('results/hourly_dask')))
