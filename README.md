# ODD2023-Datascience-Ex06AIM:
# Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
# ALGORITHM:
### Step 1:Read the given Data
### Step 2:Perform Data cleaning process on the dataset.
### Step 3:Apply Feature Transformation techniques to all the features of the data set
### step 4:Analyse the transformed features

# CODE AND OUTPUT:
```py
Developed by : Kanishka.V.S
Register no:212222230061
```
```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/e29355e6-30ad-4bcb-a7e6-4e35aa4e9cf8)
```py
df.info()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/8610adb9-9a7d-48f3-853f-02d22a614bd0)

```py
df.skew()

```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/9b4702ee-76dc-4168-9383-20ef4b509057)
```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/aab2ff28-b999-4ca7-8de1-44f43ddebcd7)
```py

np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/b5c8e172-2b65-430c-8732-354fc199be53)
```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/26d76bdf-f080-46ee-a00c-bb827cc9536f)
```py
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/b2845902-6dcd-4ccd-914d-c982ca074150)
```py
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/9cc128f2-22f1-4e93-bf4e-65a87af90861)
```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[['Moderate Negative Skew']])
df
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/bab84b66-b2e7-4f10-9ea3-4b2190ac7fab)
```py

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/e150fd33-5f00-4e62-878c-f21d21b47a9e)
```py

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/5845528f-3703-449b-af04-2e154a84787c)
```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/fded0806-4461-46d8-8de0-d96781989a3e)
```py
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex06/assets/113497357/fc04c9ab-7a63-441a-a9db-08cf5a650128)

# RESULT:
Thus feature transformation is done for the given dataset.


