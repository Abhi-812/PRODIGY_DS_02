import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


url='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(url)

print("First 5 rows of the dataset: ")
print(df.head())

print("\n Summary of the dataset: ")
print(df.info())

print("\nmissing values in the dataset: ")
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df.drop(columns=['Cabin'],inplace=True)

print("\nmissing values after cleaning the dataset: ")
print(df.isnull().sum())

#EDA
# 1
survival_rate=df['Survived'].mean()*100
print(f"\nSurvival rate:{survival_rate:.2f}%")

# 2
plt.figure(figsize=(8,6))
sns.countplot(x='Survived',data=df)
plt.title('Survival count')
plt.xticks([0,1],['Not survived','survived'])
plt.ylabel('Count')
plt.show()

# 3
plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue='Sex', data=df)
plt.title('Survival count by grnder')
plt.xticks([0,1],['Not survived','survived'])
plt.ylabel('Count')
plt.show()

# 4
plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue='Pclass',data=df)
plt.title('Survival count by Passanger Class')
plt.xticks([0,1],['Not survived','survived'])
plt.ylabel('Count')
plt.show()

# 5
plt.figure(figsize=(8,6))
sns.histplot(df['Age'], bins=30, edgecolor='Black')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 6
plt.figure(figsize=(8,6))
sns.countplot(x='Survived',hue='Embarked',data=df)
plt.title('Survival count by Embarkation point')
plt.xticks([0,1],['Not survived','survived'])
plt.ylabel('Count')
plt.show()

# 7
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
