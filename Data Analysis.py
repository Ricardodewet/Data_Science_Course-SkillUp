# EXPLORATORY DATA ANALYSIS

# Histogram and Scatter plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
# %matplotlib inline         # to show plot on notebook - jupyter-lab

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


x_axis = data
y_axis = target


# hist

style.use('ggplot')
plt.figure(figsize = (7,7))
plt.hist(y_axis, bins = 50)
plt.xlabel('price in 1000s usd')
plt.ylabel('number of houses')
plt.show()

# scatter

style.use('ggplot')
plt.figure(figsize = (7,7))
plt.scatter(x_axis[:,5], y_axis)
plt.xlabel('number of houses')
plt.ylabel('price in 1000s usd')
plt.show()


# --------------------------------------------------------------------

# Heatmap

import matplotlib.pyplot as plt
import seaborn as sb

flight_d = sb.load_dataset('flights')

flight_d.head(5)

flight_d = flight_d.pivot('month', 'year', 'passengers')
flight_d

sb.heatmap(flight_d)

# --------------------------------------------------------------------

# Pie Charts

import matplotlib.pyplot as plt

job_d = ['40', '20', '17', '8', '5', '10']
lbs = 'IT', 'Finance', 'Marketing', 'Admin', 'HR', 'Operations'

exp = (0.06,0.0,0,0,0,0)
plt.pie(job_d, labels = lbs, explode = exp)
plt.show()

# --------------------------------------------------------------------

# 2D-Plots

import matplotlib.pyplot as plt
from matplotlib import style

web_customers = [123,456,950,1290, 1630, 1450, 1045, 1295, 465, 205, 80]
time = [7,8,9,10,11,12,13,14,15,16,17]

style.use('ggplot')
plt.plot(time, web_customers, color = 'b', linestyle = '--', linewidth = 2.5, label = 'web traffic')
plt.axis([6.5,17.5,50,2000])
plt.title('web traffic')
plt.xlabel('Hours')
plt.ylabel('Num of users')
plt.legend()
plt.show()


style.use('ggplot')
plt.plot(time, web_customers, alpha = .4)
plt.annotate('Max', ha = 'center', va = 'bottom', xytext = (8, 1500), xy = (11, 1630), arrowprops = {'facecolor' : 'green'})
plt.title('web traffic')
plt.xlabel('Hours')
plt.ylabel('Num of users')
plt.show()


# --------------------------------------------------------------------

# Multiple Plots

import matplotlib.pyplot as plt
from matplotlib import style

web_mon = [123,456,950,1290, 1630, 1450, 1045, 1295, 465, 205, 80]
web_tue = [45 ,67,342,456,654,678,777,954,1200,1223, 1253]
web_wed = [56,67,345,456,865,932,1134,1234,1567,1765,1990]
time = [7,8,9,10,11,12,13,14,15,16,17]

style.use('ggplot')
plt.plot(time, web_mon, 'r', label = 'monday')
plt.plot(time, web_tue, 'b', label = 'tue')
plt.plot(time, web_wed, 'g', label = 'wed')
plt.title('web traffic')
plt.xlabel('Hours')
plt.ylabel('Num of users')
plt.show()

# --------------------------------------------------------------------

# Subplot - Display graphs in a single frame

import matplotlib.pyplot as plt
from matplotlib import style

temp = [79,75,74,73,81,77,81,95,93,95,97,98,99,98,98,97,92,94,92,83,83,81,72,84]
wind = [14,12,10,13,9,13,12,13,17,13,17,1,18,7,25,10,10,16,0,16,9,9,9,5]
time = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
humidity = [73,76,78,81,81,84,84,79,71,64,58,55,51,48,46,46,45,46,48,453,60,67,69,73]
percipitation = [26,42,+9,48,11,16,5,11,3,48,26,21,58,69,74,53,79,58,26,74,95,69,58, 68]

# 2 Graphs

plt.figure(figsize = (8,4))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,2,1)
plt.title('Temp')
plt.plot(time, temp, color = 'b', linestyle = '-', linewidth = 1)
plt.subplot(1,2,2)
plt.title('wind')
plt.plot(time, wind, color = 'r', linestyle ='-', linewidth = 1)

# cobine graphs

plt.figure(figsize = (6,6))
plt.subplots_adjust(hspace = .25)
plt.subplot(2,1,1)
plt.title('Humidity')
plt.plot(time, humidity, color = 'b', linestyle = '-', linewidth = 1)
# plt.subplot(2,1,2)
plt.title('Percipitation')
plt.plot(time, percipitation, color = 'r', linestyle ='-', linewidth = 1)

# 4 Graphs

plt.figure(figsize = (9,9))
plt.subplots_adjust(hspace = .3)
plt.subplot(2,2,1)
plt.title('Temp')
plt.plot(time, temp, color = 'b', linestyle = '-', linewidth = 1)
plt.subplot(2,2,2)
plt.title('wind')
plt.plot(time, wind, color = 'r', linestyle ='-', linewidth = 1)
plt.subplot(2,2,3)
plt.title('Humidity')
plt.plot(time, humidity, color = 'g', linestyle = '-', linewidth = 1)
plt.subplot(2,2,4)
plt.title('Percipitation')
plt.plot(time, percipitation, color = 'y', linestyle ='-', linewidth = 1)

# --------------------------------------------------------------------

# Pair-Plot

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df_auto = pd.read_csv("C:\dataset\\auto_mpg.csv")
df_auto.head(5)

def origin(num):
    if num == 1:
        return 'USA'
    elif num == 2:
        return 'Eruope'
    elif num == 3:
        return 'Asia'

df_auto['origin'] = df_auto['origin'].apply(origin)
df_auto.head(30)

sb.pairplot(df_auto[['mpg', 'weight', 'origin']], hue = 'origin', size = 4)

# --------------------------------------------------------------------

#  Pie chart

import matplotlib.pyplot as plt

cause = 'Chronic Diseases', 'Unintentional Injuries', 'Alzheimers', 'Infuenza and Pneuminia', 'Sepsis', 'Others'
percentile = [62, 5, 4, 2, 1, 26]

plt.figure(figsize = (10,10))
explode = (0.05, 0, 0, 0, 0, 0)
plt.pie(percentile, labels = cause, explode = explode, autopct = '%1.1f%%', startangle = 70)
plt.axis('equal')
plt.title('Ohia State - 2012: Leading Cuases of death')
plt.show()

