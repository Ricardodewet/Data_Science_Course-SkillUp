# MATPLOTLIB

import matplotlib.pyplot as plt
#%matplotlib inline

cause = 'chornic diseases', 'unitentional injuries', 'alzheimers', 'infuenza and pneumonia', 'sepsis', 'others'
percentile = [62, 5, 4, 2, 1, 26]

plt.figure(figsize=(10, 10))  # sets chart size
explode = (0.05, 0, 0, 0, 0, 0) # adds space between pie's, largest one exculded from the rest
plt.pie(percentile, labels = cause, explode = explode, autopct = '%1.1f%%', startangle = 70)  # Pie chart propperties
plt.axis('equal')  # draws pie as a circle
plt.title('Cuases of Death')  # sets chart title

plt.show()  # prints chart



# --------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\dataset\kc_house_data.csv")
df.sort_index(inplace = True)

df.head()

plt.plot(df['price'])
plt.boxplot(df['price'])
plt.hist(df['price'])

plt.xlabel('price')
plt.ylabel('sqft_living')
plt.title('new')
plt.scatter(x = df['price'], y = df['sqft_living'])

plt.figure(figsize = (15, 10), dpi = 100)
plt.xlabel('price')
plt.ylabel('sqft_living')
plt.title('new_new')
plt.legend()
plt.plot(df['price'])
#
# SAVING GRAPH
# plt.savefig('Graph_name.png')


# --------------------------------------------------------------------


d = {'a':10, 'b':20, 'c':13}
# plt.bar(x = d.keys(), height = d.values())

plt.pie(x = d.values(), labels = d.keys())



# --------------------------------------------------------------------

# 3D GRAPH

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\dataset\ADANIPORTS.csv", parse_dates = True)
df.head()

df['H-L'] = df.High - df.Low
df['100MA'] = df['Close'].rolling(100).mean()

axis = plt.axes(projection = '3d')
axis.scatter(df.index, df['H-L'], df['100MA'])
axis.set_xlabel('Index')
axis.set_ylabel('H-L')
axis.set_zlabel('100MA')
plt.show()


import numpy as np

z1 = np.linspace(0, 10, 100)
x1 = np.cos(2 + z1)
y1 = np.sin(2 + z1)

sns.set_style('whitegrid')
axis = plt.axes(projection = '3d')
axis.plot3D(x1, y1, z1)
plt.show()


def return_z(x,y):
    return 50-(x**2+y**2)
sns.set_style('whitegrid')
x1,y1 = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
x1,y1 = np.meshgrid(x1,y1)
z1 = return_z(x1, y1)

axis = plt.axes(projection ='3d')
axis.plot_surface(x1,y1,z1)
plt.show()


# --------------------------------------------------------------------

# 3D GRAPH

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\dataset\ADANIPORTS.csv", parse_dates = True)
df.head()

df['H-L'] = df.High - df.Low
df['100MA'] = df['Close'].rolling(100).mean()

axis = plt.axes(projection = '3d')
axis.scatter(df.index, df['H-L'], df['100MA'])
axis.set_xlabel('Index')
axis.set_ylabel('H-L')
axis.set_zlabel('100MA')
plt.show()


import numpy as np

z1 = np.linspace(0, 10, 100)
x1 = np.cos(2 + z1)
y1 = np.sin(2 + z1)

sns.set_style('whitegrid')
axis = plt.axes(projection = '3d')
axis.plot3D(x1, y1, z1)
plt.show()



# --------------------------------------------------------------------

# 3D GRAPH

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\dataset\ADANIPORTS.csv", parse_dates = True)
df.head()

df['H-L'] = df.High - df.Low
df['100MA'] = df['Close'].rolling(100).mean()

axis = plt.axes(projection = '3d')
axis.scatter(df.index, df['H-L'], df['100MA'])
axis.set_xlabel('Index')
axis.set_ylabel('H-L')
axis.set_zlabel('100MA')
plt.show()



# --------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv("C:\dataset\kc_house_data.csv")
df.sort_index(inplace = True)

df.head()

plt.figure(figsize = (7,7), dpi = 100)
plt.title('Box plot')
sns.boxplot(data = df, x = 'price')



# --------------------------------------------------------------------

from scipy import linalg
import numpy as np

# test has 30 Questions and 150 marks
# T/F Questions are 4 marks each - x
# MC Questions are 9 marks each  - y
# x + y = 30
# 4x + 9y = 150

test_Q_var = np.array([[1,1], [4,9]])
test_Q_val = np.array([30, 150])

linalg.solve(test_Q_var, test_Q_val)

# x = 24, y = 6

