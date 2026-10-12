import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('titanic_seaborn.csv')
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked'])
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

features = ['pclass', 'age', 'fare', 'sibsp', 'parch', 'sex_male', 'embarked_Q', 'embarked_S']
X = df[features].astype(float)
y = df['survived']
print(X.shape)
print(y.value_counts())

kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters
print(df['cluster'].value_counts())
print(pd.crosstab(df['cluster'], df['survived']))
print(df.groupby('cluster')[features].mean().round(2))

sns.scatterplot(x='age', y='fare', hue='cluster', data=df)
plt.title('Кластеризация пассажиров')
plt.savefig('clusters.png', dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(pd.Series(model.coef_[0], index=features).round(2))