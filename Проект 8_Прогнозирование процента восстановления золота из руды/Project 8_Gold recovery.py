#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Загрузим-и-исследуем-первичные-данных" data-toc-modified-id="Загрузим-и-исследуем-первичные-данных-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Загрузим и исследуем первичные данных</a></span></li><li><span><a href="#Устраненим-пропуски" data-toc-modified-id="Устраненим-пропуски-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Устраненим пропуски</a></span></li><li><span><a href="#Проверим-наличие-дубликатов" data-toc-modified-id="Проверим-наличие-дубликатов-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Проверим наличие дубликатов</a></span></li><li><span><a href="#Преобразуем-типы-данных" data-toc-modified-id="Преобразуем-типы-данных-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Преобразуем типы данных</a></span></li><li><span><a href="#Проверим,-каких-признаков-нет-в-тестовой-выборке" data-toc-modified-id="Проверим,-каких-признаков-нет-в-тестовой-выборке-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Проверим, каких признаков нет в тестовой выборке</a></span></li><li><span><a href="#Проверим,-что-эффективность-обогащения-рассчитана-правильно" data-toc-modified-id="Проверим,-что-эффективность-обогащения-рассчитана-правильно-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Проверим, что эффективность обогащения рассчитана правильно</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ данных</a></span><ul class="toc-item"><li><span><a href="#Посмотрим,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки" data-toc-modified-id="Посмотрим,-как-меняется-концентрация-металлов-(Au,-Ag,-Pb)-на-различных-этапах-очистки-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Посмотрим, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки</a></span></li><li><span><a href="#Сравним-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках." data-toc-modified-id="Сравним-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках.-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Сравним распределения размеров гранул сырья на обучающей и тестовой выборках.</a></span></li><li><span><a href="#Исследуем-суммарную-концентрацию-всех-веществ-на-разных-стадиях:-в-сырье,-в-черновом-и-финальном-концентратах." data-toc-modified-id="Исследуем-суммарную-концентрацию-всех-веществ-на-разных-стадиях:-в-сырье,-в-черновом-и-финальном-концентратах.-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Исследуем суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.</a></span></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Модель</a></span><ul class="toc-item"><li><span><a href="#Напишем-функцию-для-вычисления-итоговой-sMAPE." data-toc-modified-id="Напишем-функцию-для-вычисления-итоговой-sMAPE.-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Напишем функцию для вычисления итоговой sMAPE.</a></span></li><li><span><a href="#Обучим-разные-модели-и-оценим-их-качество-кросс-валидацией.-Выберем-лучшую-модель-и-проверим-её-на-тестовой-выборке." data-toc-modified-id="Обучим-разные-модели-и-оценим-их-качество-кросс-валидацией.-Выберем-лучшую-модель-и-проверим-её-на-тестовой-выборке.-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Обучим разные модели и оценим их качество кросс-валидацией. Выберем лучшую модель и проверим её на тестовой выборке.</a></span><ul class="toc-item"><li><span><a href="#Подготовим-данные" data-toc-modified-id="Подготовим-данные-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Подготовим данные</a></span></li><li><span><a href="#Удалим-среднее-и-масштабируем-дисперсию" data-toc-modified-id="Удалим-среднее-и-масштабируем-дисперсию-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Удалим среднее и масштабируем дисперсию</a></span></li><li><span><a href="#Обучим-модель-линейной-регресии" data-toc-modified-id="Обучим-модель-линейной-регресии-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Обучим модель линейной регресии</a></span></li><li><span><a href="#Обучим-модель-дерева-решений" data-toc-modified-id="Обучим-модель-дерева-решений-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>Обучим модель дерева решений</a></span></li><li><span><a href="#Обучим-модель-случайного-леса¶" data-toc-modified-id="Обучим-модель-случайного-леса¶-3.2.5"><span class="toc-item-num">3.2.5&nbsp;&nbsp;</span>Обучим модель случайного леса¶</a></span></li><li><span><a href="#&quot;Обучим&quot;-константную-модель-(dummy)" data-toc-modified-id="&quot;Обучим&quot;-константную-модель-(dummy)-3.2.6"><span class="toc-item-num">3.2.6&nbsp;&nbsp;</span>"Обучим" константную модель (dummy)</a></span></li><li><span><a href="#Сравним-итоговые-SMAPE-всех-моделей" data-toc-modified-id="Сравним-итоговые-SMAPE-всех-моделей-3.2.7"><span class="toc-item-num">3.2.7&nbsp;&nbsp;</span>Сравним итоговые SMAPE всех моделей</a></span></li></ul></li></ul></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Восстановление золота из руды

# Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Вам нужно:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.
# 
# Чтобы выполнить проект, обращайтесь к библиотекам *pandas*, *matplotlib* и *sklearn.* Вам поможет их документация.

# ## Подготовка данных

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.dummy import DummyRegressor

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from tqdm import tqdm


# ### Загрузим и исследуем первичные данных

# In[2]:


data_train = pd.read_csv('/datasets/gold_recovery_train_new.csv') #обучающая выборка
data_test  = pd.read_csv('/datasets/gold_recovery_test_new.csv')  #тестовая выборка
data_full  = pd.read_csv('/datasets/gold_recovery_full_new.csv')  #исходные данные

display(data_train.head(5))
display(data_test.head(5))
display(data_full.head(5))

display(data_train.describe())
display(data_test.describe())
display(data_full.describe())

data_train.info()
data_test.info()
data_full.info()


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# 1. В данных есть пропуски
#     
# 2. Дата сохранена в формате object
# 
# 3. В тестовой выборке меньше признаков, чем в обучающей и исходной (53 vs 87)
# </div>

# ### Устраненим пропуски

# In[3]:


#1 Посчитаем число пропущенных значений в каждом файле
print('-----------------------------')
print('1. Пропуски в обучающей выборке:')
print('-----------------------------')
display(data_train.isnull().head(5))
display(data_train.isnull().sum().sort_values(ascending=False, inplace=False).head(5))

print('---------------------------')
print('2. Пропуски в тестовой выборке:')
print('---------------------------')
display(data_test.isnull().head(5))
display(data_test.isnull().sum().sort_values(ascending=False, inplace=False).head(5))

print('---------------------------')
print('3. Пропуски в исходной выборке:')
print('---------------------------')
display(data_full.isnull().head(5))
display(data_full.isnull().sum().sort_values(ascending=False, inplace=False).head(5))


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# 1. В тестовой выборке почти нет пропусков
# 2. Больше всего пропусков в параметре secondary_cleaner.output.tail_sol в обучающей (1605) и исходной (1748) выборки
# </div>

# In[4]:


#"Данные индексируются датой и временем получения информации (признак date). Соседние по времени параметры часто похожи."
#2 Дозаполним соседними значениями

data_train = data_train.fillna(method="ffill")         # Обучающая выборка
data_test  = data_test.fillna(method="ffill")          # Тестовая выборка
data_full  = data_full.fillna(method="ffill")          # Исходные данные


# In[5]:


print('-----------------------------')
print('1. Пропуски в обучающей выборке после преобразования:')
print('-----------------------------')
display(data_train.isnull().sum().sort_values(ascending=False, inplace=False).head(2))

print('---------------------------')
print('2. Пропуски в тестовой выборке после преобразования:')
print('---------------------------')
display(data_test.isnull().sum().sort_values(ascending=False, inplace=False).head(2))

print('---------------------------')
print('3. Пропуски в исходной выборке после преобразования:')
print('---------------------------')
display(data_full.isnull().sum().sort_values(ascending=False, inplace=False).head(2))


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# все пропуски дозаполнены
# </div>

# ### Проверим наличие дубликатов

# In[6]:


#1 Ищем полные дубликаты
print("Число полных дубликатов строк в обучающей выборке:",data_train.duplicated().sum())
print("Число полных дубликатов строк в тестовой выборке:",  data_test.duplicated().sum())
print("Число полных дубликатов строк в исходной выборке:",  data_full.duplicated().sum())


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> полные дубликаты отсутствуют
# </div>

# ### Преобразуем типы данных

# In[7]:


# Преобразуем дату из типа данных object в datetime
data_train['date'] = pd.to_datetime(data_train['date'], format='%Y-%m-%d %H:%M:%S')
data_test['date']  = pd.to_datetime(data_test['date'], format='%Y-%m-%d %H:%M:%S')
data_full['date']  = pd.to_datetime(data_full['date'], format='%Y-%m-%d %H:%M:%S')


# ### Проверим, каких признаков нет в тестовой выборке

# In[8]:


columns_train = data_train.columns
columns_test  = data_test.columns

#генератор [expr(variable) for variable in iterable if condition(variable)]
columns_not_in_test = [i for i in columns_train if i not in columns_test]
display(columns_not_in_test)


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# В тестовой выборке отстутствуют:
#     
# 1. Параметры продукта по этапам обработки (rougher.output, secondary_cleaner.output, final.output), а именно данные по полученным на каждом этапе обработки концентрату (concentrate) и «отвальным хвостам» (tail). 
# 
# 2. Некоторые расчетные характеристики этапов флотации, финального (rougher.calculation, final.output.recovery).
# </div>

# In[9]:


#Все параметры обучающей выброки
display(columns_train)


# ### Проверим, что эффективность обогащения рассчитана правильно

# In[10]:


#Вычислим эффективность на обучающей выборке для признака      rougher.output.recovery. 
#  Recovery = C*(F-T)/(F*(C-T))*100%,                   
#  где:
#  C — доля золота в концентрате после флотации/очистки;      rougher.output.concentrate_au
#  F — доля золота в сырье/концентрате до флотации/очистки;   rougher.input.feed_au
#  T — доля золота в отвальных хвостах после флотации/очистки. rougher.output.tail_au

C = data_train['rougher.output.concentrate_au']
F = data_train['rougher.input.feed_au']
T = data_train['rougher.output.tail_au']
recovery = data_train['rougher.output.recovery']

recovery_calc = C*(F-T)/(F*(C-T))*100
#Найдем MAE между моим расчётами и значением признака.
print("MAE между расчётов и значением признака: ", 
      mean_absolute_error(recovery, recovery_calc))


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> MAE близка к нулю, т.e. расчет recovery корректен.
# </div>

# ## Анализ данных

# ### Посмотрим, как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки

# In[16]:


#   rougher.input.feed_au                                                       - смесь золотой руды
#   rougher.output.concentrate_au           (rougher — флотация)                 - черновой концентрат                 
#   primary_cleaner.output.concentrate_au   (primary_cleaner — первичная очистка)  - ?после первичной очистки  
#   secondary_cleaner.output.concentrate_au (secondary_cleaner — вторичная очистка) - нет такого
#   final.output.concentrate_au             (final — финальные характеристики)      - финальный концентрат

def concentrate_au_hist(output, col, name):
    plt.hist(output, bins=30, label = name, color= col, alpha = 0.5, density=True)
    plt.ylabel("Доля проверок")
    plt.legend()
#plt.show()

print('----------')
print('1. Золото')
print('----------')
concentrate_au_hist(data_train['rougher.output.concentrate_au'],'r', "Черновой концентрат")
concentrate_au_hist(data_train['primary_cleaner.output.concentrate_au'],'g', "Концентрат после первичной очистки")
concentrate_au_hist(data_train['final.output.concentrate_au'],'b', "Финальный концентрат")
plt.show()
print('----------')
print('2. Серебро')
print('----------')
concentrate_au_hist(data_train['rougher.output.concentrate_ag'],'r', "Черновой концентрат")
concentrate_au_hist(data_train['primary_cleaner.output.concentrate_ag'],'g', "Концентрат после первичной очистки")
concentrate_au_hist(data_train['final.output.concentrate_ag'],'b', "Финальный концентрат")
plt.show()
print('----------')
print('3. Свинец')
print('----------')
concentrate_au_hist(data_train['rougher.output.concentrate_pb'],'r', "Черновой концентрат")
concentrate_au_hist(data_train['primary_cleaner.output.concentrate_pb'],'g', "Концентрат после первичной очистки")
concentrate_au_hist(data_train['final.output.concentrate_pb'],'b', "Финальный концентрат")
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# 1. Золото. Концентрация увеличивается с каждой стадией обработки в ходе технологического процесса.
# 
# 2. Серебро. Концентрация на каждой следующей стадии уменьшается.
# 
# 3. Свинец. Концентрация возрастает после этапа флотации, а после почти не меняется.
# </div>

# ### Сравним распределения размеров гранул сырья на обучающей и тестовой выборках.

# In[17]:


#1) Построим графики распределения гранул сырья в обучающей и тестовой выборок.
##----На этапе подготовки сырья к флотации (rougher.input.feed_size)
print('----------------------------------')
print('1. На этапе подготовки сырья к флотации')
print('----------------------------------')

concentrate_au_hist(data_train['rougher.input.feed_size'],'r', "Гранул сырья в обучающей выборке")
concentrate_au_hist(data_test['rougher.input.feed_size'],'g', "Гранул сырья в тестовой выборке")
plt.show()
#2) Проверим гипотезу о равенстве средних размеров гранул в обучающей и тестовой выборках.
print("Средний размеров гранул сырья в обучающей выборке", data_train['rougher.input.feed_size'].describe()['mean'])
print("Средний размеров гранул сырья в тестовой выборке", data_test['rougher.input.feed_size'].describe()['mean'])
print('Стандартное отклонение в обучающей выборке:', data_train['rougher.input.feed_size'].describe()['std'])
print('Стандартное отклонение в тестовой выборке:', data_test['rougher.input.feed_size'].describe()['std'])

#Пороговое значение alpha - 0.05
alpha = 0.05 #если p-value окажется меньше него - отвергнем гипотезу

results = st.ttest_ind(data_train['rougher.input.feed_size'], data_test['rougher.input.feed_size'], equal_var=False)
print('p-значение:', results.pvalue)

if (results.pvalue < alpha):
    print("Средний размер гранул обучающей и тестовой выборки различаются")
else:
    print("Средний размер гранул обучающей и тестовой выборки одинаковые")


# In[18]:


##----На этапе подготовки сырья к флотации
print('-----------------------------')
print('2. Перед первым этапом очистки')
print('-----------------------------')

concentrate_au_hist(data_train['primary_cleaner.input.feed_size'],'r', "Гранул сырья в обучающей выборке")
concentrate_au_hist(data_test['primary_cleaner.input.feed_size'],'g', "Гранул сырья в тестовой выборке")
plt.show()
#2) Проверим гипотезу о равенстве средних размеров гранул в обучающей и тестовой выборках.
print("Средний размеров гранул сырья в обучающей выборке", data_train['primary_cleaner.input.feed_size'].describe()['mean'])
print("Средний размеров гранул сырья в тестовой выборке", data_test['primary_cleaner.input.feed_size'].describe()['mean'])
print('Стандартное отклонение в обучающей выборке:', data_train['primary_cleaner.input.feed_size'].describe()['std'])
print('Стандартное отклонение в тестовой выборке:', data_test['primary_cleaner.input.feed_size'].describe()['std'])

#Пороговое значение alpha - 0.05
alpha = 0.05 #если p-value окажется меньше него - отвергнем гипотезу

results = st.ttest_ind(data_train['primary_cleaner.input.feed_size'], data_test['primary_cleaner.input.feed_size'], equal_var=False)
print('p-значение:', results.pvalue)

if (results.pvalue < alpha):
    print("Средний размер гранул обучающей и тестовой выборки различаются.")
else:
    print("Средний размер гранул обучающей и тестовой выборки одинаковые.")


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# 1. На этапе подготовки сырья к флотации cредний размер гранул обучающей и тестовой выборки различаются.
# 2. Перед первым этапом очистки cредний размер гранул обучающей и тестовой выборки различаются.
#     
# Распределения с большой долейй вероятности не равны. Это может повлиять на результаты предсказания модели.
# </div>

# ### Исследуем суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.

# In[20]:


steps  =  [ 'rougher.input.feed' ,  'rougher.output.concentrate' 
           ,  'primary_cleaner.output.concentrate',  ' final.output.concentrate ' ]
metals = ['au', 'ag', 'pb']
#-------------------------
rougher_feed_train = data_train['rougher.input.feed_au'] + data_train['rougher.input.feed_ag'] + data_train['rougher.output.concentrate_pb']
rougher_output_train = data_train['rougher.output.concentrate_au'] + data_train['rougher.output.concentrate_ag'] + data_train['rougher.output.concentrate_pb']
primary_cleaner_train = data_train['primary_cleaner.output.concentrate_au'] + data_train['primary_cleaner.output.concentrate_ag'] + data_train['primary_cleaner.output.concentrate_pb'] 
final_train = data_train['final.output.concentrate_au'] + data_train['final.output.concentrate_ag'] + data_train['final.output.concentrate_pb']
#-------------------------
concentrate_au_hist(rougher_feed_train,'y', "Металлов в сырье")
concentrate_au_hist(rougher_output_train,'r', "Черновой концентрат мателлов")
concentrate_au_hist(primary_cleaner_train,'g', "Концентрат мателлов первой очистки")
concentrate_au_hist(final_train,'b', "Финальный концентрат металлов")


# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# Суммарная концентрация металлов возрастает к финальной стадии.
# </div>

# ## Модель

# In[21]:


# Избавимся от выбросов
# Удаляем строки с значением концентрацией металлов, близким к нулю (<3)
# Черновой концентрат rougher.output.concentrate_
data_train_n = data_train[(data_train['rougher.output.concentrate_au'] > 2) 
                          & (data_train['rougher.output.concentrate_ag'] > 2) 
                          & (data_train['rougher.output.concentrate_pb'] > 2)]

data_full_n = data_full[(data_full['rougher.output.concentrate_au'] > 2) 
                        & (data_full['rougher.output.concentrate_ag'] > 2) 
                        & (data_full['rougher.output.concentrate_pb'] > 2)]

    
# Первый этап очистки primary_cleaner.output.concentrate_
data_train_n = data_train_n[(data_train_n['primary_cleaner.output.concentrate_au'] > 2) 
                            & (data_train_n['primary_cleaner.output.concentrate_ag'] > 2) 
                            & (data_train_n['primary_cleaner.output.concentrate_pb'] > 2)]

data_full_n = data_full_n[(data_full_n['primary_cleaner.output.concentrate_au'] > 2) 
                          & (data_full_n['primary_cleaner.output.concentrate_ag'] > 2) 
                          & (data_full_n['primary_cleaner.output.concentrate_pb'] > 2)]

# Финальный концентрат final.output.concentrate_
data_train_n = data_train_n[(data_train_n['final.output.concentrate_au'] > 2) 
                            & (data_train_n['final.output.concentrate_ag'] > 2) 
                            & (data_train_n['final.output.concentrate_pb'] > 2)]

data_full_n = data_full_n[(data_full_n['final.output.concentrate_au'] > 2) 
                            & (data_full_n['final.output.concentrate_ag'] > 2) 
                            & (data_full_n['final.output.concentrate_pb'] > 2)]

data_test_n = data_test.merge(data_full_n[['date']]
                            ,how = 'inner', on = 'date')

#Размер сырья rougher.input.feed_size
data_train_n = data_train_n[data_train_n['rougher.input.feed_size'] < 200]
data_full_n = data_full_n[data_full_n['rougher.input.feed_size'] < 200]

data_test_n = data_test.merge(data_full_n[['date','rougher.output.recovery', 'final.output.recovery']]
                            ,how = 'inner', on = 'date')

display(data_train_n.shape)
display(data_full_n.shape)
display(data_test_n.shape)


# ### Напишем функцию для вычисления итоговой sMAPE.

# In[22]:


print(data_test_n.columns)


# In[23]:


#sMAPE (англ. Symmetric Mean Absolute Percentage Error, «симметричное среднее абсолютное процентное отклонение»).
# Функция для вычисления метрики качества sMAPE
#    Она похожа на MAE, но выражается не в абсолютных величинах, а в относительных
#    Параметр target - значение целевого признака 
#    Параметр predictions - значение предсказания 
def sMAPE(target, predictions):
    return (100*abs(target-predictions)*2/(abs(target)+abs(predictions))).mean()

def final_sMAPE (rougher, final):
    return (0.25*rougher + 0.75*final) 


# ### Обучим разные модели и оценим их качество кросс-валидацией. Выберем лучшую модель и проверим её на тестовой выборке.

# #### Подготовим данные

# In[24]:


#train, Эффективность обогащения - rougher.output.recovery и final.output.recovery
features_train = data_train_n.drop(columns_not_in_test, axis=1) 
features_train_final = features_train.drop(['date'], axis=1) 
features_train_rougher = features_train_final.loc[:, 'rougher.input.feed_ag': 'rougher.state.floatbank10_f_level']

features_test_final = data_test_n.drop(['date', 'rougher.output.recovery', 'final.output.recovery'], axis=1)
features_test_rougher = features_test_final.loc[:, 'rougher.input.feed_ag': 'rougher.state.floatbank10_f_level']

target_train_rougher = data_train_n['rougher.output.recovery']
target_train_final = data_train_n['final.output.recovery']

#target_recovery_test = data_test['rougher.output.recovery']
#target_final_test = data_test['final.output.recovery']
target_test_rougher = data_test_n['rougher.output.recovery']
target_test_final = data_test_n['final.output.recovery']


# #### Удалим среднее и масштабируем дисперсию 

# In[25]:


scaler = StandardScaler()

scaler.fit(features_train_final)
features_train_final_sc  = scaler.transform(features_train_final)
features_test_final_sc = scaler.transform(features_test_final)

scaler.fit(features_train_rougher)
features_train_rougher_sc = scaler.transform(features_train_rougher)
features_test_rougher_sc = scaler.transform(features_test_rougher)


# In[26]:


# Приведем признаки в формат датафрейма
features_train_final_sc   = pd.DataFrame(features_train_final_sc, columns=features_train_final.columns)
features_train_rougher_sc = pd.DataFrame(features_train_rougher_sc, columns=features_train_rougher.columns)
features_test_final_sc    = pd.DataFrame(features_test_final_sc, columns=features_test_final.columns)
features_test_rougher_sc  = pd.DataFrame(features_test_rougher_sc, columns=features_test_rougher.columns)


# #### Обучим модель линейной регресии

# In[27]:


#LinearRegression
model_linear_rougher = LinearRegression()
model_linear_final = LinearRegression()

model_linear_rougher.fit(features_train_rougher_sc, target_train_rougher)
linear_predict_rougher = model_linear_rougher.predict(features_test_rougher_sc)
linear_rougher_smape = sMAPE(target_test_rougher, linear_predict_rougher)

model_linear_final.fit(features_train_final_sc, target_train_final)
linear_predict_final = model_linear_final.predict(features_test_final_sc)
linear_final_smape = sMAPE(target_test_final, linear_predict_final)

linear_sMAPE = final_sMAPE(linear_rougher_smape, linear_final_smape)

print('SMAPE у Линейной регрессии на этапе подготовки сырья к флотации', linear_rougher_smape)
print('SMAPE у Линейной регрессии финальной стадии', linear_final_smape)
print('Итоговый SMAPE у Линейной регрессии {:.2f}'.format(linear_sMAPE))


# #### Обучим модель дерева решений

# In[32]:


#DecisionTree, этап подготовки сырья к флотации 
model_DecisionTree_rougher = DecisionTreeRegressor()

tree_params = {'max_depth': range(1,13,2), 'max_features': range(4,15,2),'random_state': [12345]}
# Тренируем модель дерева решений перебором вариантов
scorer = make_scorer(sMAPE, greater_is_better = False) #NEW

grid_search = GridSearchCV(model_DecisionTree_rougher, param_grid=tree_params, scoring=scorer, cv=5)
# Подбираем гиперпараметры модели дерева решений
grid_search.fit(features_train_rougher_sc, target_train_rougher)
print("Подобраны гиперпараметры модели дерева решений на этапе подготовки сырья к флотации :")
print(grid_search.best_params_)


# In[33]:


# Тестируем модель дерева решений с найденными оптимальными гиперпараметрами
model_DecisionTree_rougher = DecisionTreeRegressor(max_depth=5, max_features = 4, random_state = 12345)
model_DecisionTree_rougher.fit(features_train_rougher_sc, target_train_rougher)
tree_predict_rougher = model_DecisionTree_rougher.predict(features_test_rougher_sc)
tree_rougher_smape = sMAPE(target_test_rougher, tree_predict_rougher)

print('SMAPE у дерева решений на этапе подготовки сырья к флотации', tree_rougher_smape)


# In[36]:


get_ipython().run_cell_magic('time', '', '#DecisionTree, финальный этап\nmodel_DecisionTree_final = DecisionTreeRegressor()\n\ntree_params = {\'max_depth\': range(1,13,2), \'max_features\': range(4,15,2),\'random_state\': [12345]}\n# Тренируем модель дерева решений перебором вариантов\nscorer = make_scorer(sMAPE, greater_is_better = False) #NEW\n\ngrid_search = GridSearchCV(model_DecisionTree_final, param_grid=tree_params, scoring=scorer, cv=5)\n# Подбираем гиперпараметры модели дерева решений\ngrid_search.fit(features_train_final_sc, target_train_final)\nprint("Подобраны гиперпараметры дерева решений на финальном этапе:")\nprint(grid_search.best_params_)')


# In[37]:


# Тестируем модель дерева решений с найденными оптимальными гиперпараметрами
model_DecisionTree_final = DecisionTreeRegressor(max_depth=1, max_features = 14, random_state = 12345)
model_DecisionTree_final.fit(features_train_final_sc, target_train_final)
tree_predict_final = model_DecisionTree_final.predict(features_test_final_sc)
tree_final_smape = sMAPE(target_test_final, tree_predict_final)

print('SMAPE у дерева решений на финальном этапе', tree_final_smape)


# In[38]:


tree_sMAPE = final_sMAPE(tree_rougher_smape, tree_final_smape)
print('Итоговый SMAPE у дерева решений {:.2f}'.format(tree_sMAPE))


# #### Обучим модель случайного леса¶

# In[39]:


get_ipython().run_cell_magic('time', '', '#RandomForestRegressor, этап подготовки сырья к флотации \nmodel_RandomForest_rougher = RandomForestRegressor()\n\nforest_params = {\'max_depth\': range(1,13,4), \'max_features\': range(4,15,4)\n                 , \'n_estimators\': range (4, 34, 10),\'random_state\': [12345]}\n\n# Тренируем модель перебором вариантов\nscorer = make_scorer(sMAPE, greater_is_better = False) #NEW\n\ngrid_search = GridSearchCV(model_RandomForest_rougher, param_grid=forest_params, scoring = scorer, cv=5)\n# Подбираем гиперпараметры\ngrid_search.fit(features_train_rougher_sc, target_train_rougher)\nprint("Подобраны гиперпараметры модели случайного леса на этапе подготовки сырья к флотации:")\nprint(grid_search.best_params_)')


# In[40]:


# Тестируем модель с найденными оптимальными гиперпараметрами
model_RandomForest_rougher = RandomForestRegressor(max_depth=5, max_features = 8
                                                   , n_estimators = 24, random_state = 12345)
model_RandomForest_rougher.fit(features_train_rougher_sc, target_train_rougher)
forest_predict_rougher = model_RandomForest_rougher.predict(features_test_rougher_sc)
forest_rougher_smape = sMAPE(target_test_rougher, forest_predict_rougher)

print('SMAPE у случайного леса на этапе подготовки сырья к флотации', forest_rougher_smape)


# In[41]:


get_ipython().run_cell_magic('time', '', '#RandomForestRegressor, финальный этап\nmodel_RandomForest_final = RandomForestRegressor()\n\nforest_params = {\'max_depth\': range(1,13,4), \'max_features\': range(4,15,4)\n                 , \'n_estimators\': range (4, 34, 10),\'random_state\': [12345]}\n\n# Тренируем модель перебором вариантов\nscorer = make_scorer(sMAPE, greater_is_better = False) #NEW\n\ngrid_search = GridSearchCV(model_RandomForest_final, param_grid=forest_params, scoring = scorer, cv=5)\n# Подбираем гиперпараметры\ngrid_search.fit(features_train_final_sc, target_train_final)\nprint("Подобраны гиперпараметры модели случайного леса на этапе подготовки сырья к флотации:")\nprint(grid_search.best_params_)')


# In[42]:


# Тестируем модель с найденными оптимальными гиперпараметрами
model_RandomForest_final = RandomForestRegressor(max_depth=5, max_features = 12
                                                   , n_estimators = 24, random_state = 12345)
model_RandomForest_final.fit(features_train_final_sc, target_train_final)
forest_predict_final = model_RandomForest_final.predict(features_test_final_sc)
forest_final_smape = sMAPE(target_test_final, forest_predict_final)

print('SMAPE у случайного леса на фианльном этапе', forest_final_smape)


# In[43]:


forest_sMAPE = final_sMAPE(forest_rougher_smape, forest_final_smape)
print('Итоговый SMAPE у случайного леса {:.2f}'.format(forest_sMAPE))


# #### "Обучим" константную модель (dummy)

# In[44]:


#DummyRegressor, этап подготовки сырья к флотации
model_Dummy = DummyRegressor(strategy="median")

model_Dummy.fit(features_train_rougher_sc, target_train_rougher)
dummy_predict_rougher = model_Dummy.predict(features_test_rougher_sc)
dummy_rougher_smape = sMAPE(target_test_rougher, dummy_predict_rougher)

print('SMAPE у константной модели на этапе подготовки сырья к флотации', dummy_rougher_smape)

model_Dummy.fit(features_train_final_sc, target_train_final)
dummy_predict_final = model_Dummy.predict(features_test_final_sc)
dummy_final_smape = sMAPE(target_test_final, dummy_predict_final)

print('SMAPE у константной модели на фианльном этапе', dummy_final_smape)

dummy_sMAPE = final_sMAPE(dummy_rougher_smape, dummy_final_smape)
print('Итоговый SMAPE у константной модели {:.2f}'.format(dummy_sMAPE))


# #### Сравним итоговые SMAPE всех моделей

# In[45]:


print('Итоговый SMAPE у линейной модели {:.2f}'.format(linear_sMAPE))
print('Итоговый SMAPE у дерева поиска {:.2f}'.format(tree_sMAPE))
print('Итоговый SMAPE у случайного леса {:.2f}'.format(forest_sMAPE))
print('Итоговый SMAPE у константной модели {:.2f}'.format(dummy_sMAPE))


# 

# <div class="alert alert-block alert-warning">
# <b>Вывод:</b> 
# 
# Посчитав итоговую функцию SMAPE для моделей
#     
# - линейной регрессии,
#     
# - дерева решений,
#     
# - случайного леса,
#     
# - константной (dummy с strategy="median")
#     
# могу сделать вывод, что лучшее значение получено для модели случайного леса с подобранными с помощью GridSearchCV гиперпараметрами.
# </div>

# ## Чек-лист готовности проекта

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке выполнения
# - [x]  Выполнен шаг 1: данные подготовлены
#     - [x]  Проверена формула вычисления эффективности обогащения
#     - [x]  Проанализированы признаки, недоступные в тестовой выборке
#     - [x]  Проведена предобработка данных
# - [x]  Выполнен шаг 2: данные проанализированы
#     - [x]  Исследовано изменение концентрации элементов на каждом этапе
#     - [x]  Проанализированы распределения размеров гранул на обучающей и тестовой выборках
#     - [x]  Исследованы суммарные концентрации
# - [x]  Выполнен шаг 3: построена модель прогнозирования
#     - [x]  Написана функция для вычисления итогового *sMAPE*
#     - [x]  Обучено и проверено несколько моделей
#     - [x]  Выбрана лучшая модель, её качество проверено на тестовой выборке
