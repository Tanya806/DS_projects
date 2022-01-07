#!/usr/bin/env python
# coding: utf-8

# # Рекомендация тарифов

# В вашем распоряжении данные о поведении клиентов, которые уже перешли на эти тарифы (из проекта курса «Статистический анализ данных»). Нужно построить модель для задачи классификации, которая выберет подходящий тариф. Предобработка данных не понадобится — вы её уже сделали.
# 
# Постройте модель с максимально большим значением *accuracy*. Чтобы сдать проект успешно, нужно довести долю правильных ответов по крайней мере до 0.75. Проверьте *accuracy* на тестовой выборке самостоятельно.

# Шаги исследования:
# 1. [Открытие данных](#start)
# 2. [Разбить на выборки](#split)
# 3. [Исследовать модели](#model)
# 4. [Проверить модель на тестовой выборке](#check)
# 5. [Проверить модель на адекватность](#bonus)
# 6. [Общий вывод](#output)

# <a id="start"></a>
# ## Откройте и изучите файл

# In[26]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #NEW
from sklearn.model_selection import RandomizedSearchCV #NEW
from IPython.display import display

users_behavior = pd.read_csv('/datasets/users_behavior.csv', sep=",")
print('--------------')
print('Users behavior')
print('--------------')
display(users_behavior.head(50))
display(users_behavior.describe())
users_behavior.info()


# <a id="split"></a>
# ## Разбейте данные на выборки

# In[27]:


features = users_behavior.drop(['is_ultra'], axis=1)
target = users_behavior['is_ultra']
#Выделим обучающую выборку
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.4, random_state=12345)
#Выделим вылидационную и тестовую выборки
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.5, random_state=12345)
print('Размер обучающей выборки',features_train.count()) #NEW
print('Размер вылидационной выборки',features_valid.count()) #NEW
print('Размер тестовой выборки',features_test.count()) #NEW


# <a id="model"></a>
# ## Исследуйте модели

# In[31]:


#Проверим модель логистической регрессии
model_reg = LogisticRegression(random_state = 12345)#модель логистической регрессии
print(model_reg)
model_reg.fit(features_train, target_train) #модель на тренировочной выборке
result_reg = model_reg.score(features_valid, target_valid) #метрика качества модели на валидационной выборке
print("Accuracy модели логистической регрессии на валидационной выборке:", result_reg)


# In[29]:


#Выберем лучшую модель случайного леса
best_model = None
best_result = 0
for est in range(1, 11):
    model_tree = RandomForestClassifier(random_state=12345, n_estimators= est) #модель с заданным количеством деревьев
    model_tree.fit(features_train, target_train) # модель на тренировочной выборке
    result_tree = model_reg.score(features_valid, target_valid) # качество модели на валидационной выборке
    if result_tree > best_result:
        best_model = model_tree#наилучшая модель
        best_result = result_tree#наилучшее значение метрики accuracy на валидационных данных

print("Accuracy наилучшей модели на валидационной выборке:", best_result)
print(best_model)


# <div class="alert alert-block alert-info">
# 
# <b>Совет:</b> 
# Можно также применить GridSearchCv и RandomizedSearchCV
#     
# Есть хороший пример: https://www.mygreatlearning.com/blog/gridsearchcv/
# </div>

# <a id="check"></a>
# ## Проверьте модель на тестовой выборке

# In[30]:


model_tree_test = RandomForestClassifier(random_state=12345, n_estimators= 10)# инициализирую модель RandomForestRegressor с параметрами random_state=12345, n_estimators=est и max_depth=depth
model_tree_test.fit(features_train,target_train) 
#result = model.score(features_valid, target_valid)
result_model_tree_test = model_tree_test.score(features_test, target_test)
result_reg = model_reg.score(features_test, target_test)
print("Accuracy модели логистической регрессии на тестовой выборке:", result_reg)
print("Accuracy модели случайного леса на тестовой выборке:", result_model_tree_test)
prediction = pd.merge(features_test, target_test, left_index=True, right_index=True)
display(prediction.head(5))


# С учетом проверки на тестовой выборке следует выбрать модель случайного леса с n_estimators = 10

# <a id="bonus"></a>
# ## (бонус) Проверьте модели на адекватность

# In[25]:


#Люди, которые часто разговоривают, шлют сообщения и используют интернет, выберут тариф "Ультра"
#calls	minutes	messages	mb_used
new_features = pd.DataFrame(
    [[20000.0, 2000.0, 2000000.0, 20000.0]],
    columns=features.columns)
predictions = model_tree_test.predict(new_features)
print(predictions)

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(features_train, target_train) # модель на тренировочной выборке
dummy_clf.predict(features_valid)
result_dummy = dummy_clf.score(features_valid, target_valid)
print('Accuracy DummyClassifier на тестовой выборке:', result_dummy)


# <div class="alert alert-block alert-info">
# 
# <b>Вывод:</b> 
# Полученная модель предсказывает лучше, чем DummyClassifier.
# </div>

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x] Jupyter Notebook открыт
# - [x] Весь код исполняется без ошибок
# - [x] Ячейки с кодом расположены в порядке исполнения
# - [x] Выполнено задание 1: данные загружены и изучены
# - [x] Выполнено задание 2: данные разбиты на три выборки
# - [x] Выполнено задание 3: проведено исследование моделей
#     - [x] Рассмотрено больше одной модели
#     - [x] Рассмотрено хотя бы 3 значения гипепараметров для какой-нибудь модели
#     - [x] Написаны выводы по результатам исследования
# - [x] Выполнено задание 3: Проведено тестирование
# - [x] Удалось достичь accuracy не меньше 0.75
# 
