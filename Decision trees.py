import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
import math as m

dogs = pd.read_csv("https://stepik.org/media/attachments/course/4852/dogs.csv")
dogs = dogs.drop('Unnamed: 0', axis=1)

dogs_X = dogs.iloc[:, :3]
dogs_y = dogs.iloc[:, 3]

dogs_clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dogs_clf.fit(dogs_X, dogs_y)

print(tree.plot_tree(dogs_clf, feature_names=dogs_X.columns))
plt.show()

# E_sh_sob
i1 =(1/1)*m.log2((1/1)) - 0
# E_sh_kot\
i2 =-(4/9)*m.log2((4/9)) - (5/9)*m.log2((5/9))
# E_gav_sob\
i3 =0 - (5/5)*m.log2((5/5))
# E_gav_kot\
i4 =-(4/5)*m.log2((4/5)) - (1/5)*m.log2((1/5))
# E_laz_sob\
i5 =0 - (6/6)*m.log2((6/6))
# E_laz_kot\
i6 =-(4/4)*m.log2((4/4)) - 0
i7 = - 4/10 * m.log2(4/10) - 6/10 * m.log2(6/10)
print(i1,i2,i3,i4,i5,i6,i7)

train_iris_data = pd.read_csv("train_iris.csv", index_col=0)
test_iris_data = pd.read_csv("test_iris.csv", index_col=0)

scores_data = pd.DataFrame()

max_depth_value = range(1, 100)

X_train, y_train = train_iris_data.drop(columns='species'), train_iris_data.species
X_test, y_test = test_iris_data.drop(columns='species'), test_iris_data.species

rs = np.random.seed(0)

for max_depth in max_depth_value:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=rs)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score]
                                    })
    scores_data = scores_data.append(temp_score_data)

scores_data_long = pd.melt(scores_data,
                           id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score'],
                           var_name='set_type',
                           value_name='score')

plt.figure(figsize=(30, 10))
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
plt.show()

# Считать данные для обучения Дерева
dogs_n_cats = pd.read_csv('dogs_n_cats.csv')
X_train = dogs_n_cats.drop('Вид', axis=1)
y_train = dogs_n_cats['Вид']

# Обучение Дерева
rs = np.random.seed(0)
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=rs, max_depth=5)
clf.fit(X_train, y_train)

# Результат обучения
clf.score(X_train, y_train)

# Считать тестовые данные
X_test = pd.read_json('test_dogs_and_cats.json')

# Предсказать результаты
y_pred = clf.predict(X_test)

# Подсчитать количество собачек
print(pd.Series(y_pred)[y_pred == 'собачка'].count())



train_iris_data = pd.read_csv("songs.csv", index_col=0)
test_iris_data = pd.read_csv("test_iris.csv", index_col=0)

scores_data = pd.DataFrame()

max_depth_value = range(1, 100)

X_train, y_train = train_iris_data.drop(columns='species'), train_iris_data.species
X_test, y_test = test_iris_data.drop(columns='species'), test_iris_data.species

rs = np.random.seed(0)

for max_depth in max_depth_value:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=rs)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score]
                                    })
    scores_data = scores_data.append(temp_score_data)

scores_data_long = pd.melt(scores_data,
                           id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score'],
                           var_name='set_type',
                           value_name='score')

plt.figure(figsize=(30, 10))
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
plt.show()

data_enum = pd.read_csv('enum.csv')
X_train, y_train = data_enum.drop(columns='num'), data_enum.num
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
tree.plot_tree(clf, filled=True)
print(clf.tree_.n_node_samples,
      clf.tree_.impurity)
plt.show()


print(round((0.99587 - (157*0.90271134 + 81*0.8256253) / 238),3))

data = pd.read_csv('submissions_data_train.csv')
print(data[data.submission_status == "wrong"].groupby(['user_id', 'step_id'], as_index=False).agg({'timestamp':'max'}).step_id.value_counts().keys())
print(data.sort_values(['user_id', 'timestamp'], ascending=False).drop_duplicates(['user_id'])\
        .query("submission_status == 'wrong'").groupby('step_id')\
        .count().sort_values('submission_status').tail(1))