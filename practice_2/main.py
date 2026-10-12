import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('titanic_seaborn.csv')
print(df.head())
print(df.shape)

print(df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isnull().sum())

le = LabelEncoder()
df['class_label'] = le.fit_transform(df['class'])
print(df[['class', 'class_label']].drop_duplicates().sort_values('class_label'))

df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
print(df.filter(like='sex').head())
print(df.filter(like='embarked').head())

print(df[['age', 'fare']].describe().round(2))
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
print(df[['age', 'fare']].describe().round(2))
print(df[['age', 'fare']].head())

sns.histplot(df['age'], bins=30, kde=True)
plt.title('Распределение возраста (стандартизованного)')
plt.savefig('hist_age.png', dpi=150)
plt.show()

sns.boxplot(x='class', y='fare', data=df)
plt.title('Стоимость билета по классам (стандартизованная)')
plt.savefig('boxplot_fare.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150)
plt.show()