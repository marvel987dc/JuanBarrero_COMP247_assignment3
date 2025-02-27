import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import false

# Load data
data = pd.read_csv("./Data/breast_cancer.csv")
data_juanDavid = pd.DataFrame(data)

print("\nColumn and names: ", data_juanDavid.columns, data_juanDavid.dtypes)
print("\nmissing values: ", data_juanDavid.isnull().sum())
print("\nStatistic of numeric fields: ", data_juanDavid.describe())

#preprocessing and visualization
data_juanDavid['bare'] = data_juanDavid['bare'].replace('?', np.nan)
#fill any missing value with the median of the column

data_juanDavid["bare"] = pd.to_numeric(data_juanDavid["bare"])
print(data_juanDavid['bare'].dtype)

data_juanDavid = data_juanDavid.drop('ID', axis=1)

data_juanDavid['bare'] = data_juanDavid['bare'].fillna(data_juanDavid['bare'].mean())

#plotting the data sns
plt.figure(figsize=(10, 5))
sns.countplot(data=data_juanDavid, legend=False,palette="coolwarm")
plt.title("Distribution of Classes")
plt.xlabel("Class (2=Benign, 4=Malignant)")
plt.ylabel("Count")
plt.show()

#plotting the data mathplotlib
plt.figure(figsize=(10, 5))
plt.hist(data_juanDavid['bare'], bins=10, color='blue')
plt.title("Distribution of Bare Nuclei")
plt.xlabel("Bare Nuclei")
plt.ylabel("Count")
plt.show()

#plottin the data pandas
plt.figure(figsize=(10, 5))
#bar plot to vizualize teh distribution of values int he 'Class' column
data_juanDavid['class'].value_counts().plot(kind='bar', color='green')
plt.title("Distribution of Classes")
plt.xlabel("Class (2=Benign, 4=Malignant)")
plt.ylabel("Count")
plt.show()
#Separate the features from the class.
X = data_juanDavid.drop('class', axis=1)
y = data_juanDavid['class']
#splitting the data 80% training and 20% testing
from sklearn.model_selection import train_test_split
#The seed are the two last digits of my student ID
seed = 53
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Training the model
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel
#Create the model
clf_Juan  = SVC(kernel='linear', C = 0.1)
clf_Juan.fit(X_train, y_train)



