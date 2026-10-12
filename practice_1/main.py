import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
print("=== Первые 5 строк ===")
print(df.head())

print("\n=== df.info() ===")
df.info()
print("\n=== df.describe() ===")
print(df.describe())

print("\n=== Пропуски по столбцам ===")
print(df.isna().sum())
print("\nКоличество дубликатов:", df.duplicated().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])
df = df.drop_duplicates()

print("\n=== После очистки ===")
print("Пропусков всего:", df.isna().sum().sum())
print("Размер таблицы:", df.shape)

print("\n=== Статистика по Age ===")
print("Среднее :", round(df['Age'].mean(), 2))
print("Минимум :", df['Age'].min())
print("Максимум:", df['Age'].max())

print("\n=== Статистика по числовым колонкам ===")
print(df[['Age', 'Fare', 'SibSp', 'Parch']].agg(['mean', 'min', 'max']).round(2))

plt.figure(figsize=(7, 4))
plt.hist(df['Age'], bins=20, color='#4C72B0', edgecolor='white')
plt.axvline(df['Age'].mean(), color='red', linestyle='--', label='Среднее')
plt.title('Распределение возраста пассажиров')
plt.xlabel('Возраст, лет')
plt.ylabel('Количество пассажиров')
plt.legend()
plt.tight_layout()
plt.savefig('age_hist.png', dpi=150)
plt.show()