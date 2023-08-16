# STATISTICS

# Mean Calculations

def calc_mean(x):
	total = 0
	for i in range(len(x)):
		total += x[i]
	mean = total / len(x)
	return print(mean)


calc_mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# OR

import statistics as stats

stats.mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# --------------------------------------------------------------------

# Median

import statistics as stats
x = [1,2,3,4,5,6,7,8,9,10]
stats.median(x)

# --------------------------------------------------------------------

# Mode
import statistics as stats
x = [4,3,5,6,7,8,2,5,4,2,3,6,78,5,3423,1,1,436,56,8,3,1,1,0]
stats.mode(x)

# --------------------------------------------------------------------

# Standard Deviation

import statistics as stats
x = [1,2,3,4,5]
b = stats.stdev(x)
print(b)

# Variance
#               Standard Deviation Squared
c = b**2

print(c)

# --------------------------------------------------------------------

# Skewness

import numpy as np
from statsmodels.stats.stattools import robust_skewness, medcouple

x = np.array([2,3,5,7,8,9,15])

skewness = medcouple(x)
print(skewness)

# --------------------------------------------------------------------

# Kurtosis

import numpy as np
from statsmodels.stats.stattools import robust_kurtosis

x = np.array([2,4,5,7,0,9,11,15])
kurtosis = robust_kurtosis(x)
kurtosis


# --------------------------------------------------------------------

# Covariance

def cov(x, y):
	meanx = sum(x) / float(len(x))
	meany = sum(y) / float(len(x))
	xpart = [i - meanx for i in x]
	ypart = [i - meany for i in y]

	numerator = sum([xpart[i] * ypart[i] for i in range(len(xpart))])
	denominator = len(x) - 1

	covariance = numerator / denominator
	return covariance


cov([1.23, 2.12, 3.34, 4.5], [2.56, 2.89, 3.76, 3.95])

# from geeks for geeks


# Python code to demonstrate the
# use of numpy.cov
import numpy as np

x = [1.23, 2.12, 3.34, 4.5]

y = [2.56, 2.89, 3.76, 3.95]

# find out covariance with respect  rows
cov_mat = np.stack((x, y), axis=1)

print("shape of matrix x and y:", np.shape(cov_mat))

print("shape of covariance matrix:", np.shape(np.cov(cov_mat)))

print(np.cov(cov_mat))

# --------------------------------------------------------------------

# Correlation

import statistics as stats


def std_dev(x):
	b = stats.stdev(x)
	return b


def cov(x, y):
	meanx = sum(x) / float(len(x))
	meany = sum(y) / float(len(x))
	xpart = [i - meanx for i in x]
	ypart = [i - meany for i in y]

	numerator = sum([xpart[i] * ypart[i] for i in range(len(xpart))])
	denominator = len(x) - 1

	covariance = numerator / denominator
	return covariance


x = [1.23, 2.12, 3.34, 4.5]

y = [2.56, 2.89, 3.76, 3.95]

correlation = cov(x, y) / std_dev(x) * std_dev(y)

print(correlation)

# --------------------------------------------------------------------

# Project

# libraries needed for EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# % matplotlib inline
sb.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')

# reading dataset
df = pd.read_csv("C:\dataset\Sales.csv")

# setting date col as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'], inplace=True)

# check distribution and skewness
# for col in df.columns:
#     sb.distplot(df[col])
# plt.show()

df.describe()

# -------------------------------------------------------------------------------
# this block does not work with this dataset, use as eg
products = ['Product Name' str(x) for x in range(len(df) + 1)]
for product in products:
	print(product, end=' -')
	print(np.mean(df[product + '_Price'] / df[product]))

# returs --> product1 - product_price ect.

# -------------------------------------------------------------------------------

# represent data weekly
wdf = df.resample('W').mean()
wdf.head()
wdf.describe()

plt.figure(figsize=(10, 5), dpi=100)
plt.tick_params(axis='x', rotation=45)
plt.title("profit during each year")
plt.bar(x=wdf.index, height=wdf.Profit)

# represent data yearly
ydf = df.resample('Y').mean()
ydf.head()
ydf.describe()

plt.figure(figsize=(10, 5), dpi=100)
plt.tick_params(axis='x', rotation=45)
plt.title("profit during each year")
plt.bar(x=ydf.Year, height=ydf.Profit)

# --------------------------------------------------------------------

# Binomail Distribution

from scipy.stats import binom
import matplotlib.pyplot as plt
# set values
n = 6
p = 0.6
# drfine r values list
r_values = list(range(n + 1))
# get mean and var
mean, var = binom.stats(n, p)
# pmf values
dist = [binom.pmf(r,n,p) for r in r_values]
# print table
print('r\tp(r)')
for i in range(n + 1):
    print(str(r_values[i]) + '\t' + str(dist[i]))
# plot graph
plt.bar(r_values, dist)
plt.show()

# --------------------------------------------------------------------

# Poisson's Distribution

from scipy.stats import poisson
import matplotlib.pyplot as plt

x = [0,1,2,3,4,5]
lmbda = 2

poisson_pd = poisson.pmf(x, lmbda)

fig, ax = plt.subplots(1,1, figsize = (8, 6))
ax.plot(x, poisson_pd, 'bo', ms = 8, label = 'poisson pmf')
plt.ylabel('probalility', fontsize = '18')
plt.xlabel('x - No. of restoraunts', fontsize = '18')
ax.vlines(x, 0, poisson_pd, colors = 'b', lw = 5, alpha = 0.5)

# --------------------------------------------------------------------

# Sample Mean
data = [2,3,4,5,6,5,34,5,56,45,34,23,1,23,34,465,6574,54,34,24,354,435,56,75465,24,1,23,34,34,435,56,7,67,56,45,42,342,131,331,2,44,345,5,566,67,6576,3452,423424,5,65,6,5,432,2345,5467,67]

sample_mean = sum(data) / len(data)

print(sample_mean)


# --------------------------------------------------------------------

# T-test
                # used when vales are les than 30
import random
import seaborn as sb
import scipy.stats as stats
import numpy as np

a= [random.gauss(50, 20) for x in range(30)]
b= [random.gauss(55, 15) for x in range(30)]

sb.set_style('darkgrid')
sb.kdeplot(a, shade = True)
sb.kdeplot(b, shade = True)

t_stat, p_value = stats.ttest_ind(a, b, equal_var = False)
print(f'ttest_ind: {t_stat, p_value}')
# if p_value is > 0.05, we can accept the null hypothises (H0). meaning the means for both distributions are almost the same.

print(np.mean(a), np.mean(b))

t_stat, p_value = stats.ttest_rel(a, b)
print(f'ttest_rel: {t_stat, p_value}')

# The ttest_ind() is an unpaired t-test that is used to compare two independent sets of data.
# While ttest-rel() is a paired t-test that is used to compare the dependent sets of data.
# And the ttest_1samp() is used to compare the mean of one group against some unique value.

print(np.mean(a))

t_stat, p_value = stats.ttest_1samp(a, np.mean(a) - 5, axis =0)
print(f'ttest_1samp: {t_stat, p_value}')
# if p_value is > 0.05, we can accept the null hypothises (H0). meaning the means for both distributions are almost the same.

# --------------------------------------------------------------------

# Z-tests
                # used when vales are more than 30

import random
random.seed(20)
from statsmodels.stats.weightstats import ztest as ztest

mean = 100
standard_deviation = 15

a = [random.gauss(mean, standard_deviation) for x in range(40)]

print(ztest(a, value = 100)) # H0 <0.05, H0 rejected

a = [random.gauss(mean+20, standard_deviation) for x in range(40)]
print(ztest(a, value = 100, alternative = 'larger')) # H0 <0.05, H0 rejected

a = [random.gauss(mean-20, standard_deviation) for x in range(40)]
print(ztest(a, value = 100, alternative = 'smaller')) # H0 <0.05, H0 rejected


# ztest-2

print('ztest-2')

a = [random.gauss(mean, standard_deviation) for x in range(40)]
b = [random.gauss(mean+20, standard_deviation) for x in range(40)]
print(ztest(a,b, value = 0)) # H0 <0.05, H0 rejected


# but if mean is the same

print('')

a = [random.gauss(mean, standard_deviation) for x in range(40)]
b = [random.gauss(mean, standard_deviation) for x in range(40)]
print(ztest(a,b, value = 0)) # H0 >0.05, H0 accepted

# --------------------------------------------------------------------

# Chi-Squared Distribution

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

data1 = np.random.chisquare(df = 1, size = 1000)
data2 = np.random.chisquare(df = 2, size = 1000)
data3 = np.random.chisquare(df = 3, size = 1000)

sb.distplot(data1, hist = False, label = 'dof 1') # only want line graph, thus hist = false
sb.distplot(data2, hist = False, label = 'dof 2')
sb.distplot(data3, hist = False, label = 'dof 3')
plt.legend()


# --------------------------------------------------------------------

# chi-square test
                # maily used to determine if there is a relation between 2 sets of data
from scipy.stats import chi2_contingency

data = [[10,20,30], [6,9,17]]

stat, p_value, dof, chi_array = chi2_contingency(data)
print(p_value) # value > than 0,05 thus accepted

# but

data = [[10,20,30], [9,1,8]]

stat, p_value, dof, chi_array = chi2_contingency(data)
print(p_value) # value < than 0,05 thus rejected


# --------------------------------------------------------------------

# F-distribution

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f

x = np.linspace(0, 4.5, 1000)
data_1 = f(1,1,0)
data_2 = f(5,8,0)
data_3 = f(4,4,0)
data_4 = f(100,200,0)

plt.figure(figsize = (12,6))
plt.plot(x, data_1.pdf(x), label = '1,1')
plt.plot(x, data_2.pdf(x), label = '5.8')
plt.plot(x, data_3.pdf(x), label = '4.4')
plt.plot(x, data_4.pdf(x), label = '100,200')
plt.legend()

# --------------------------------------------------------------------

# F-test

import random
random.seed(20)

x = np.array([random.gauss(100,15) for x in range(20)])
y = np.array([random.gauss(100,15) for x in range(20)])

f_test_stat  = np.var(x, ddof = 1) / np.var(y, ddof=1)

dfn = x.size -1
dfd = y.size -1

p_value = 1-f.cdf(f_test_stat, dfn, dfd)
print(p_value)
print('')

x = np.array([random.gauss(100,15) for x in range(20)])
y = np.array([random.gauss(200,15) for x in range(20)])

f_test_stat  = np.var(x, ddof = 1) / np.var(y, ddof=1)

dfn = x.size -1
dfd = y.size -1

p_value = 1-f.cdf(f_test_stat, dfn, dfd)
print(p_value)
print('')

# --------------------------------------------------------------------

# Advanced Statistics Project

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as snf
import warnings


warnings.filterwarnings('ignore')
# %matplotlib inline

google = yf.Ticker('GOOG')

df = google.history('5Y')
df.head()
df.shape
df.tail()
df.describe()


sb.set_style('darkgrid')
plt.figure(figsize = (7,5), dpi = 150)
plt.title('cloding price')
plt.plot(df['Close'])


#  calculate 50DMA
df['fiftyDMA'] = df['Close'].rolling(50).mean()

#  calculate 200DMA
df['thDMA'] = df['Close'].rolling(200).mean()
df.describe()

# plot closing price vs 50DMMA VS 200DMA
sb.set_style('darkgrid')
plt.figure(figsize = (7,5), dpi = 150)
plt.title('closing price vs 50DMMA VS 200DMA')
plt.plot(df['Close'], label = 'close')
plt.plot(df['fiftyDMA'], label = '50DMA')
plt.plot(df['thDMA'], label = '200DMA')
plt.legend()


# anilyse correlation between each vatialbe
plt.figure(figsize = (7,7), dpi = 100)
sb.heatmap(df.corr(), annot = True)

# plot distplot of 50DMA
sb.set_style('darkgrid')
plt.figure(figsize = (7,5), dpi = 150)
plt.title('distplot of 50DMA')
sb.distplot(df['fiftyDMA'])


# plot distplot of close price
sb.set_style('darkgrid')
plt.figure(figsize = (7,5), dpi = 150)
plt.title('distplot of close price')
sb.distplot(df['Close'])

# fit ot a stats model
model = snf.ols(formula = 'Close ~ fiftyDMA', data = df)
model = model.fit()
model.summary()


