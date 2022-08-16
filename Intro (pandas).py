import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('StudentsPerformance.csv')
for i in data['lunch'].unique():
    print('Дисперсия для группы с ',i,' ланчем равна: ', \
          round(data[data['lunch']==i][['math score','reading score','writing score']].values.std()**2,2), \
          ' , а среднее: ', \
          round(data[data['lunch']==i][['math score','reading score','writing score']].values.mean(),2))
    print(data[data['lunch']==i].describe())

data = pd.read_csv('column_hell.csv')
selected_columns=data.filter(like='-')

data = pd.read_csv('dota_hero_stats.csv')
print(data.groupby(by=['attack_type','primary_attr']).count())

data = pd.read_csv('accountancy.csv')
print(data.groupby(by=['Type','Executor']).mean())

concentrations = pd.read_csv('algae.csv')
mean_concentrations = concentrations.groupby(['genus'])['alanin'].agg(['min', 'mean', 'max'])
print(round(mean_concentrations,2))
print(concentrations.groupby(['group']).describe())


df = pd.read_csv('https://stepik.org/media/attachments/course/4852/income.csv')
df['income'].plot()
df.plot(kind='line')
df.plot()
plt.show()

df = pd.read_clipboard()
f1, f2 = df.columns
df.plot.scatter(f1, f2)
plt.show()

my_data = pd.DataFrame (
    {'type':['A', 'A', 'B', 'B'], 'value': [10, 14 ,12 ,23]}
)

my_stat = pd.read_csv('my_stat.csv')
subset_1 = my_stat.iloc[0:10,[0,2]]
subset_2 = my_stat.iloc[:,[1,3]].drop(my_stat.index[[0,4]])
print(subset_1,
      subset_2
      )

my_stat = pd.read_csv('my_stat.csv')
subset_1 = my_stat[(my_stat['V1']>=0) & (my_stat['V3']=='A')]
subset_2 = my_stat[(my_stat['V2']!=10) | (my_stat['V4']>=1)]
print(subset_1,
      subset_2
      )

my_stat['V5'] = my_stat['V1'] + my_stat['V4']
my_stat['V6'] = np.log(my_stat['V2'])
print(my_stat)

my_stat.rename(index=str, columns={
    'V1':'session_value',
    'V2':'time',
    'V3':'group',
    'V4':'n_users'},inplace=True)

my_stat.session_value=my_stat['session_value'].fillna(0)
my_stat=my_stat.replace({'n_users':my_stat.n_users[my_stat.n_users<0]},{'n_users':my_stat[my_stat.n_users >= 0.0].n_users.median()})
print(my_stat)


mean_session_value_data = my_stat.groupby(by='group',as_index=False).agg({'session_value':'mean'}).rename(columns={'session_value':'mean_session_value'})
print(mean_session_value_data)

data=pd.read_csv('event_data_train.csv')
print(data.groupby(by='user_id').count().sort_values(by='step_id',ascending=True))

dataFrame=pd.read_csv('genome_matrix.csv')
# data = sns.load_dataset('genome_matrix.csv')
map = sns.heatmap(dataFrame.corr(),cmap = 'viridis')
map.xaxis.set_ticks_position('top')
map.xaxis.set_tick_params(rotation=90)
plt.show()

data=pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
sns.distplot([x.count(',')+1 for x in data.roles], bins=15)
data.roles.map(eval).map(len).mode()
plt.show()

data=pd.read_csv('iris.csv')
for column in data:
    sns.kdeplot(data=data, x = column)
    plt.legend(column)
    # sns.distplot(x = column)
    plt.show()

colors = ['g', 'r', 'blue', 'yellow', 'white']
for col, color in zip(data.iloc[:, :-1], colors):
    sns.distplot(data[col], kde_kws = {'color':color, 'lw':1, 'label':col})
plt.show()

sns.violinplot(data = data['petal length'])
plt.show()

sns.pairplot(data=data, hue = 'species')
plt.show()

