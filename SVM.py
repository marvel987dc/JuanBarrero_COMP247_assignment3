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
data_juanDavid.plot(kind= 'scatter', x='clump', y='size', color='red')
plt.show()
