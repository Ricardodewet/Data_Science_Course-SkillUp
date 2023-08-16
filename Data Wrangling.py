# DATA WRANGLING

# Correlation Analysis

import pandas as pd

df = pd.readf = pd.read_csv("C:\dataset\Pakistan Districts Profile.csv")

df.describe()

df[['Railway Route Kilometrage', 'Learning Score Percentage']].groupby(['Learning Score Percentage']).describe().unstack()

df[['Railway Route Kilometrage', 'Learning Score Percentage']].corr()


# --------------------------------------------------------------------

#  Find missing data

import pandas as pd

df = pd.read_csv("C:\dataset\class-grades.csv")

df.describe()

df.isna().any()  # if the reply is true, then there is values missing

#  Replacing missing data

from sklearn.impute import SimpleImputer as si
import numpy as np

mean_imputer = si(missing_values= np.nan, strategy = 'mean')
mean_imputer = mean_imputer.fit(df)
imputed_df = mean_imputer.transform(df.values)
df = pd.DataFrame(data=imputed_df, columns = ['Prefix', 'Assignment', 'Tutorial', 'Midterm', 'TakeHome', 'Final'])

median_imputer = si(missing_values= np.nan, strategy = 'mean')
median_imputer = median_imputer.fit(df)
imputed_df = median_imputer.transform(df.values)
df = pd.DataFrame(data=imputed_df, columns = ['Prefix', 'Assignment', 'Tutorial', 'Midterm', 'TakeHome', 'Final'])

df.head(9)

# --------------------------------------------------------------------

# Detect an outlier

import seaborn as sb

df = pd.read_csv("C:\dataset\class-grades.csv")

sb.boxplot(x = df["Assignment"]) # all values <60 are outliers

# Treat Outliers

filter = df ['Assignment'].values > 60
df_outlier_rem = df[filter]
print(df_outlier_rem)

sb.boxplot(x = df_outlier_rem["Assignment"])

# --------------------------------------------------------------------

#  Outlier and Missing Value Treatment

from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = load_diabetes()

dataset.data
dataset.target
dataset['feature_names']

df = pd.DataFrame(data = np.c_[dataset['data'], dataset['target']], columns =dataset['feature_names'] + ['target'])
df.isnull().any()

for column in df:
    plt.figure()
    df.boxplot([column])
