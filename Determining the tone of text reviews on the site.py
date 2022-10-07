#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span><ul class="toc-item"><li><span><a href="#Загрузка-данных-и-исследование-общей-информации" data-toc-modified-id="Загрузка-данных-и-исследование-общей-информации-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Загрузка данных и исследование общей информации</a></span></li><li><span><a href="#Преобразование-текста" data-toc-modified-id="Преобразование-текста-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Преобразование текста</a></span></li><li><span><a href="#Балансировка-классов" data-toc-modified-id="Балансировка-классов-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Балансировка классов</a></span></li><li><span><a href="#Подготовим-признаки" data-toc-modified-id="Подготовим-признаки-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Подготовим признаки</a></span></li><li><span><a href="#Векторизация-текста-с-помощью-CountVectorizer-(Оценка-важности-слов-/-N-грамм)" data-toc-modified-id="Векторизация-текста-с-помощью-CountVectorizer-(Оценка-важности-слов-/-N-грамм)-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Векторизация текста с помощью CountVectorizer (Оценка важности слов / N-грамм)</a></span></li><li><span><a href="#TF-IDF" data-toc-modified-id="TF-IDF-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>TF-IDF</a></span></li></ul></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li><li><span><a href="#Деревья-поиска" data-toc-modified-id="Деревья-поиска-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Деревья поиска</a></span></li></ul></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# # Проект для «Викишоп»

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели. 
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ## Подготовка

# ### Загрузка данных и исследование общей информации

# In[6]:


# !pip uninstall spacy
get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import seaborn as sns
import matplotlib.pyplot as plt

import spacy
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer #для английского текста
#import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
import warnings

nlp =  spacy.load("en_core_web_sm")
#nlp = en_core_web_sm.load() #для русского - ru_core_news_sm
#data_comments = pd.read_csv("/Users/tatiana/Downloads/toxic_comments.csv")


# In[17]:


data_comments = pd.read_csv("/datasets/toxic_comments.csv")


# <div class="alert alert-block alert-success">
# Библиотеки импортированы и данные загружены
# </div>

# In[18]:


display(data_comments.head(5))
display(data_comments.describe())
data_comments.info()


# In[19]:


print(data_comments.isnull().sum())
print("Число полных дубликатов строк в таблице:", data_comments.duplicated().sum())


# <div class="alert alert-block alert-success">
# Дубликаты и нулевые строки отсутствуют, перед нами стоит задача бинарной классификации
# </div>

# ### Преобразование текста

# In[21]:


#1  Удалим лишние символы
import re
def clear_text(text):
    clear_text = re.sub(r'[^a-zA-Z ]', ' ', text) 
    clear_text = clear_text.split()
    clear_text = " ".join(clear_text)
    return clear_text

data_comments['clear_text'] = data_comments['text'].apply(lambda x: clear_text(x))
print("Очищенный текст:")
display(data_comments.head())


# In[23]:


#2 Токенизация
def tokenization(text):
    text = re.split('\W+', text)
    return text

data_comments['tokenized'] = data_comments['clear_text'].apply(lambda x: tokenization(x.lower()))
print("Текст после токенизации:")
display(data_comments.head())


# In[24]:


#Загружаем английские стоп-слова
nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))


# In[25]:


#3 Удаление стоп-слов
def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    text = " ".join(text)
    return text
    
data_comments['nonstop'] = data_comments['tokenized'].apply(lambda x: remove_stopwords(x))
print("Текст после удаления стоп-слов:")
display(data_comments.head())


# In[ ]:


#4 Лемматизация
#lemm = nltk.WordNetLemmatizer()
def lemmatize(text):
    doc = nlp(text)
    lemm_text = " ".join([token.lemma_ if token.lemma_ !='-PRON-' else token.text for token in doc])      
    return lemm_text

data_comments['lemmatized'] = data_comments['nonstop'].apply(lambda x: lemmatize(x))
print("Текст после лемматизации:")
display(data_comments.head())


# <div class="alert alert-block alert-success">
# Интересная статья на тему нормализации текстов:  https://towardsdatascience.com/text-normalization-7ecc8e084e31
# </div>

# ### Балансировка классов

# In[ ]:


sns.countplot(x = 'toxic', data = data_comments)

toxic_numbers = data_comments['toxic'].value_counts()
toxic_rate = toxic_numbers[1] / toxic_numbers[0] 
print('Распределение комментариев на негативные и позитивные: {:.2f}'.format(toxic_rate))
print(toxic_numbers)


# In[ ]:


#Upsampling: cделаем объекты редкого класса не такими редкими
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled


# <div class="alert alert-block alert-success">
# Произвела балансировку классов перед моделированием. 
# Для решения проблемы дисбаланса классов можно посмотреть: https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/. SMOTE очень распространен в реальной жизни и исследованиях.
# </div>

# ### Подготовим признаки

# In[ ]:


features = data_comments.drop(['toxic'], axis = 1)
target = data_comments['toxic']

print("Размер матрицы признаков:", features.shape)
print("Размер целевого признака:", target.shape)


# In[ ]:


# Выделим 80% исходные данных для обучаюшей выборки, и 20% данных для тестовой выборки
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345)

# Проверка размеров полученных выборок
print('Размер полученных выборок:')
print('features_train:', features_train.shape, 'features_test:', features_test.shape)
print('target_train:', target_train.shape, 'target_test:', target_test.shape)  


# In[ ]:


# Используем метод upsampling для улучшения дисбаланса классов в обучаюшей выборке
upsample_repeat = round(1/toxic_rate)
print("upsample_repeat", upsample_repeat)
features_train, target_train = upsample(features_train, target_train, upsample_repeat)

# Проверка размеров получившихся обучающих выборок
print("Размер обучающих выборок после балансировки классов:")
print("features_train:", features_train.shape, "target_train:", target_train.shape)


# ### Векторизация текста с помощью CountVectorizer (Оценка важности слов / N-грамм)

# In[ ]:


count_vect = CountVectorizer()


# In[ ]:


corpus_train = features_train['lemmatized'].values.astype('U')   
features_train_cv = count_vect.fit_transform(corpus_train) 

print("Размер матрицы features_train (CountVectorizer):", features_train_cv.shape)


# In[ ]:


corpus_test = features_test['lemmatized'].values.astype('U')     #  Преобразовываем англ. текст к юникоду
features_test_cv = count_vect.transform(corpus_test)

print("Размер матрицы features_test (CountVectorizer):", features_test_cv.shape)
print(features_test_cv)


# ### TF-IDF

# Переведем тесты из твиттера в векторный формат с помощью TF-IDF меры. В этой модели вес некоторого слова пропорционален частоте употребления этого слова в документе и обратно пропорционален частоте употребления слова во всех документах коллекции

# In[ ]:


count_tf_idf = TfidfVectorizer(stop_words=stopwords)

# Матрица TF-IDF на обучающей выборке
corpus_train = features_train['lemmatized'].values.astype('U')
features_train_tf_idf = count_tf_idf.fit_transform(corpus_train)

print("Размер матрицы features_train (TF-IDF):", features_train.shape)


# In[ ]:


# Матрица TF-IDF на тестовой выборке
corpus_test = features_test['lemmatized'].values.astype('U')
features_test_tf_idf = count_tf_idf.transform(corpus_test)

print("Размер матрицы features_test (TF-IDF):", features_test.shape)


# ## Обучение

# ### Логистическая регрессия

# In[ ]:


param_grid_lr = [{'solver': ['liblinear','newton-cg', 'sag', 'saga', 'lbfgs']}]


# In[ ]:


# Убираем вывод Warnings 
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'reg_model = LogisticRegression()\ngrid_search = GridSearchCV(estimator=reg_model, param_grid=param_grid_lr, scoring=\'f1\', cv=5)\ngrid_search.fit(features_train_cv, target_train)\n\nprint("Подобраны гиперпараметры модели логистической регрессии:")\nprint(grid_search.best_params_) \n#CPU times: user 45min 21s, sys: 10min 32s, total: 55min 53s\n#Wall time: 9min 2s')


# In[ ]:


model = LogisticRegression(solver='newton-cg', random_state=12345)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Обучим модель логистической регрессии для CountVectorizer \nmodel.fit(features_train_cv, target_train)')


# In[ ]:


predicted = model.predict(features_test_cv)


# In[ ]:


f1_lr_cv = f1_score(target_test, predicted)
print("Метрика F1 для CountVectorizer):", f1_lr_cv)


# In[ ]:


# Обучим модель для TfidfVectorizer
model.fit(features_train_tf_idf, target_train)
predicted = model.predict(features_test_tf_idf) 


# In[ ]:


f1_lr_tf_idf = f1_score(target_test, predicted)
print("Метрика F1 для TfidfVectorizer: ", f1_lr_tf_idf)


# ### Деревья поиска

# In[ ]:


get_ipython().run_cell_magic('time', '', '#DecisionTree\nmodel_DecisionTree = DecisionTreeClassifier(random_state=12345)\n\ntree_params = {\'max_depth\': range(11,19,2), \'max_features\': range(6,15,2),\'random_state\': [12345]}\n\ngrid_search = GridSearchCV(model_DecisionTree, param_grid=tree_params, scoring=\'f1\', cv=5)\ngrid_search.fit(features_train_cv, target_train)\nprint("Подобраны гиперпараметры модели дерева решений:")\nprint(grid_search.best_params_)')


# In[ ]:


# Модель дерева решений с подобранными гиперпараметрами
model_tree = DecisionTreeClassifier(max_depth = 17, max_features = 12, random_state=12345)
model_tree.fit(features_train_cv, target_train)

predicted = model_tree.predict(features_test_cv)
f1_tree_cv = f1_score(target_test, predicted)
print("Метрика F1 для CountVectorizer:", f1_tree_cv)


# In[ ]:


# Обучим модель для TfidfVectorizer
model_tree.fit(features_train_tf_idf, target_train)
predicted = model_tree.predict(features_test_tf_idf) 
f1_tree_tf_idf = f1_score(target_test, predicted)
print("Метрика F1 для TfidfVectorizer: ", f1_tree_tf_idf)


# ## Выводы

# Для выбора лучшей модели выявления токсичных комментариев к описаниям товаров интернет-магазина была проделана работа:
# 
#     1. загрузили данные о комментариях покупателей 
#     2. Выполнили обработку данных
#     3. Лемматизировали тексты
#     4. Рассчитали частоту употребления слов/N-грамм в комментариях методами CountVectorizer ("мешка слов") и TF-IDF.
#     5. Обучили модели логистической регрессии и дерева поиска
#     6. Выбрали модель с лучшим значением метрики F1.
#     
# Лучшее качество по метрике F1 дает модель логистической регрессии с методом оценки частоты употребления слов CountVectorizer - 0.75.

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Модели обучены
# - [x]  Значение метрики *F1* не меньше 0.75
# - [x]  Выводы написаны

# <div class="alert alert-block alert-success">
# Полезная статья на будущее: 24 метрики по проверке адекватности модели бинарной классификации: https://neptune.ai/blog/evaluation-metrics-binary-classification. 
# </div>
