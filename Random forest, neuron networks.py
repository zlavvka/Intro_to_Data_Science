import pandas as pd
from time import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble.forest import RandomForestClassifier
import math as m
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Считываем файл c данными о сердечных заболеваниях.
heart_data = pd.read_csv('heart.csv')

# Отбросим колонку, которую будем предсказывать.
X = heart_data.drop(['target'], axis=1)
# Создадим переменную, которую будем предсказывать.
y = heart_data.target

# Разбиваем DataFrame на подмножества test и train в пропорции 0.33-test, а остальное - train.
rs = np.random.seed(0)
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X, y, train_size=0.33, random_state=rs)

# Создаем модель RandomForestClassifier.
rf_heart = RandomForestClassifier()

# Задаем параметры модели.
parametrs = {'n_estimators': [10], 'max_depth': [5]}

# Обучение Random forest моделей GridSearchCV на подмножестве train.
GridSearchCV_heart_clf = GridSearchCV(rf_heart, parametrs, cv=5)
GridSearchCV_heart_clf.fit(X_heart_train, y_heart_train)

# Проведем преобразование, позволяющее определить важность переменных.
best_heart_clf = GridSearchCV_heart_clf.best_estimator_

# Создадим атрибут feature_importances_heart и сохраним его в отдельную переменную.
feature_importances_heart = best_heart_clf.feature_importances_

# Создадим DataFrame с информацией о важности переменных.
feature_importances_heart_df = pd.DataFrame({'feature_importances': feature_importances_heart},
                                            index=X_heart_train.columns)\
    .sort_values(by='feature_importances', ascending=True)\
    .rename(columns={'feature_importances': 'importance'})

# Построение графика важности переменных.
feature_importances_heart_df.plot(kind='barh', figsize=(12, 8))
plt.show()



# Устанавливаем размер области для построения графиков.
sns.set(rc={'figure.figsize': (17, 6)})

# Считываем файл c данными о съедобности грибов.
mush_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/training_mush.csv')

# Отбросим колонку, которую будем предсказывать.
X = mush_data.drop(['class'], axis=1)
# Создадим переменную, которую будем предсказывать.
y = mush_data['class']

# Создаем модель RandomForestClassifier.
rf = RandomForestClassifier(random_state=0)

# Задаем параметры модели.
parameters = {'n_estimators': range(10, 51, 10), 'max_depth': range(1, 13, 2),
              'min_samples_leaf': range(1,8), 'min_samples_split': range(2,10,2)}

# Обучение Random forest моделей GridSearchCV.
GridSearchCV_clf = GridSearchCV(rf, parameters, cv=3, n_jobs=-1)
GridSearchCV_clf.fit(X, y)

# Преобразование, позволяющее определить важность переменных.
best_clf = GridSearchCV_clf.best_estimator_

# Создадим атрибут feature_importances и сохраним его в отдельную переменную.
feature_importances = best_clf.feature_importances_
# и сделаем DataFrame, одна колонка - имена переменных, другая - важность переменных, отсортированные по убыванию.
feature_importances_df = pd.DataFrame({'features': list(X), 'feature_importances': feature_importances})\
    .sort_values(by='feature_importances', ascending=False)

# Построение графика.
f, ax = plt.subplots()
sns.barplot(y=feature_importances_df.features, x=feature_importances_df.feature_importances)
plt.xlabel('Важность атрибутов')
plt.ylabel('Атрибуты')
plt.title('Наиболее важные атрибуты')
plt.show()

before = time()
df = pd.DataFrame(range(10000000))
df.apply(np.mean)
after = time()
print(after - before)

before = time()
df = pd.DataFrame(range(10000000))
df.apply('mean')
after = time()
print(after - before)

before = time()
df = pd.DataFrame(range(10000000))
df.describe().loc['mean']
after = time()
print(after - before)

before = time()
df = pd.DataFrame(range(10000000))
df.mean(axis=0)
after = time()
print(after - before)