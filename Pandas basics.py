import pandas as pd


l = [x for x in range(5)]
s = pd.Series(l)
s[3]

s = pd.Series(l, index = ['a', 'b', 'c', 'd', 'd'])
s['d']

data = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5}
s = pd.Series(data)
s

s = pd.Series(data, index = ['a', 'b'])

# -----------------------------------------------------------------

s = pd.Series([x for x in range(1,11)])
s

s.iloc[0]
s.iat[5]

s[5:9]

s.where(s%2 == 0, 'odd number')
s.where(s%2 == 0, s**2)

s.where(s%2 == 0, inplace = True)
s.fillna('odd')


# -----------------------------------------------------------------

df = pd.DataFrame()
type(df)
df = pd.read_csv('# file_name.csv') # doesn't work, eg.

print(
pd.head(2),
pd.tail(5))

df.iloc[3]


df.values
df = pd.read_csv('file_name.csv', chunksize = 2)
for chunk in df:
    print(chunk)

df = pd.read_csv('file_name.csv')
df = df[df['Age'] > 25]

# --------------------------------------------------------------------

df = pd.read_csv('# file_name.csv', parse_dates = True)


# --------------------------------------------------------------------


df = pd.read_csv('# file_name.csv')
df.head() # Shows rows
df.columns() # Gives an index of Columns

df.plot() # plot a graph
df['column_name'].plot()
df['column_name'].plot(legend = True)
df.plot(x='column_name_1', y='column_name_2', kind = 'scatter / type_of_graph', legend = True)
df.plot(title = 'name_of_graph', figsize = (15,10))
df['column_name'].plot(kind = 'box') # box graph
df['column_name'].plot(kind = 'hist') # histogram graph

