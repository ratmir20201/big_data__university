import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
(BASE / 'output').mkdir(exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


sys.stdout = Tee(sys.stdout, open(BASE / 'output' / 'console.txt', 'w', encoding='utf-8'))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 20)

print('=== Шаг 1. Загрузка данных ===')
df = pd.read_csv(BASE / 'students.csv')
print(df)
print('Размер таблицы:', df.shape)

print('\n=== Шаг 2. Первичный осмотр ===')
df.info()
print('Пропуски по столбцам:')
print(df.isna().sum())
print('Количество полных дубликатов:', df.duplicated().sum())

print('\n=== Шаг 3. Очистка данных ===')
df = df.drop_duplicates().reset_index(drop=True)
print('После удаления дубликатов:', df.shape)
for column in ('study_hours', 'attendance', 'exam_score'):
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    print(f'Пропуски в {column} заменены медианой: {median_value}')
mode_group = df['group'].mode()[0]
df['group'] = df['group'].fillna(mode_group)
print('Пропуски в group заменены модой:', mode_group)
print('Пропусков осталось:', int(df.isna().sum().sum()))
print('Размер таблицы после очистки:', df.shape)

print('\n=== Шаг 4. Статистика ===')
numeric = ['study_hours', 'attendance', 'sleep_hours', 'exam_score']
summary = pd.DataFrame({
    'среднее': df[numeric].mean(),
    'медиана': df[numeric].median(),
    'мода': df[numeric].mode().iloc[0],
    'ст.откл.': df[numeric].std(),
})
print(summary.round(2))
print('Средний балл по группам:')
print(df.groupby('group')['exam_score'].mean().round(2))

print('\n=== Шаг 5. Корреляция ===')
correlation = df[numeric].corr()
print(correlation.round(3))
print('Корреляция study_hours и exam_score:', round(df['study_hours'].corr(df['exam_score']), 3))
slope, intercept = np.polyfit(df['study_hours'], df['exam_score'], 1)
print(f'Линейная регрессия: exam_score = {slope:.2f} * study_hours + {intercept:.2f}')

print('\n=== Шаг 6. Графики ===')
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df['exam_score'], bins=6, color='#4C72B0', edgecolor='white')
ax.axvline(df['exam_score'].mean(), color='red', linestyle='--', label='Среднее')
ax.set_title('Распределение баллов за экзамен')
ax.set_xlabel('Баллы')
ax.set_ylabel('Количество студентов')
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'output' / 'hist_scores.png', dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df['study_hours'], df['exam_score'], color='#4C72B0', label='Студенты')
xs = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 50)
ax.plot(xs, slope * xs + intercept, color='red', label='Линия тренда')
ax.set_title('Зависимость баллов от часов самоподготовки')
ax.set_xlabel('Часов подготовки в неделю')
ax.set_ylabel('Баллы за экзамен')
ax.legend()
plt.tight_layout()
plt.savefig(BASE / 'output' / 'scatter_hours_score.png', dpi=130)
plt.close()
df.to_csv(BASE / 'output' / 'students_clean.csv', index=False)
print('Графики и очищенный файл сохранены в папку output')
