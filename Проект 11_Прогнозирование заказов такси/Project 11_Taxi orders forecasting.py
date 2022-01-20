#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ</a></span><ul class="toc-item"><li><span><a href="#Выполним-его-ресемплирование-и-посчитаем-скользащее-среднее." data-toc-modified-id="Выполним-его-ресемплирование-и-посчитаем-скользащее-среднее.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Выполним его ресемплирование и посчитаем скользащее среднее.</a></span></li><li><span><a href="#Проведем-анализ-трендов-и-сезонности-по-дням-и-по-часам" data-toc-modified-id="Проведем-анализ-трендов-и-сезонности-по-дням-и-по-часам-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Проведем анализ трендов и сезонности по дням и по часам</a></span><ul class="toc-item"><li><span><a href="#Аддетивная-модель-по-дням" data-toc-modified-id="Аддетивная-модель-по-дням-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Аддетивная модель по дням</a></span></li><li><span><a href="#Мультипликативная-модель-по-дням" data-toc-modified-id="Мультипликативная-модель-по-дням-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Мультипликативная модель по дням</a></span></li><li><span><a href="#Аддетивная-модель-по-часам" data-toc-modified-id="Аддетивная-модель-по-часам-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Аддетивная модель по часам</a></span></li></ul></li><li><span><a href="#Исследуем-разности-временного-ряда-(для-ресемплирования-1-день-и-для-ресемлирования-1-час)" data-toc-modified-id="Исследуем-разности-временного-ряда-(для-ресемплирования-1-день-и-для-ресемлирования-1-час)-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Исследуем разности временного ряда (для ресемплирования 1 день и для ресемлирования 1 час)</a></span></li></ul></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#По-дням" data-toc-modified-id="По-дням-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>По дням</a></span></li><li><span><a href="#Создадим-признаки-для-модели" data-toc-modified-id="Создадим-признаки-для-модели-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Создадим признаки для модели</a></span><ul class="toc-item"><li><span><a href="#Разобьем-датасет-на-обучающую-и-тестовую-выборки" data-toc-modified-id="Разобьем-датасет-на-обучающую-и-тестовую-выборки-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Разобьем датасет на обучающую и тестовую выборки</a></span></li><li><span><a href="#Выделим-целевой-признак-и-признаки-для-обучения" data-toc-modified-id="Выделим-целевой-признак-и-признаки-для-обучения-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Выделим целевой признак и признаки для обучения</a></span></li><li><span><a href="#Обучим-модель-линейной-регрессии" data-toc-modified-id="Обучим-модель-линейной-регрессии-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Обучим модель линейной регрессии</a></span></li><li><span><a href="#Расчитаем-MAE-и-RMSE-стационарного-ряда" data-toc-modified-id="Расчитаем-MAE-и-RMSE-стационарного-ряда-3.2.4"><span class="toc-item-num">3.2.4&nbsp;&nbsp;</span>Расчитаем MAE и RMSE стационарного ряда</a></span></li></ul></li><li><span><a href="#По-часам" data-toc-modified-id="По-часам-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>По часам</a></span><ul class="toc-item"><li><span><a href="#Создадим-признаки-для-модели" data-toc-modified-id="Создадим-признаки-для-модели-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>Создадим признаки для модели</a></span></li><li><span><a href="#Разобьем-датасет-на-обучающую-и-тестовую-выборки" data-toc-modified-id="Разобьем-датасет-на-обучающую-и-тестовую-выборки-3.3.2"><span class="toc-item-num">3.3.2&nbsp;&nbsp;</span>Разобьем датасет на обучающую и тестовую выборки</a></span></li><li><span><a href="#Выделим-целевой-признак-и-признаки-для-обучения" data-toc-modified-id="Выделим-целевой-признак-и-признаки-для-обучения-3.3.3"><span class="toc-item-num">3.3.3&nbsp;&nbsp;</span>Выделим целевой признак и признаки для обучения</a></span></li><li><span><a href="#Обучим-модель-линейной-регрессии" data-toc-modified-id="Обучим-модель-линейной-регрессии-3.3.4"><span class="toc-item-num">3.3.4&nbsp;&nbsp;</span>Обучим модель линейной регрессии</a></span></li><li><span><a href="#Расчитаем-MAE-и-RMSE" data-toc-modified-id="Расчитаем-MAE-и-RMSE-3.3.5"><span class="toc-item-num">3.3.5&nbsp;&nbsp;</span>Расчитаем MAE и RMSE</a></span></li><li><span><a href="#Визуализация-модели-линейной-регрессии" data-toc-modified-id="Визуализация-модели-линейной-регрессии-3.3.6"><span class="toc-item-num">3.3.6&nbsp;&nbsp;</span>Визуализация модели линейной регрессии</a></span></li></ul></li></ul></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# #  Прогнозирование заказов такси

# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.
# 
# Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
# 
# Вам нужно:
# 
# 1. Загрузить данные и выполнить их ресемплирование по одному часу.
# 2. Проанализировать данные.
# 3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
# 4. Проверить данные на тестовой выборке и сделать выводы.
# 
# 
# Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders` (от англ. *number of orders*, «число заказов»).

# ## Подготовка

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
import seaborn as sns
import time

from IPython.display import display
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df_taxi = pd.read_csv("/datasets/taxi.csv", index_col=[0], parse_dates=[0])
df_taxi.sort_index(inplace=True)
print(df_taxi.index.is_monotonic)

df_taxi.plot()
plt.show()

display(df_taxi.head(5))
display(df_taxi.describe())
df_taxi.info()
display(df_taxi.isnull().sum())

# Общий процент пропусков в выборке
gaps = df_taxi.isnull().sum().sum()
print('Пропуски в исходной выборке, всего: ', gaps)


# ## Анализ

# ### Выполним его ресемплирование и посчитаем скользащее среднее.

# In[3]:


df_hour = df_taxi.resample('1H').sum()
df_hour['mean'] = df_hour['num_orders'].rolling(7).mean()
df_hour['std'] = df_hour['num_orders'].rolling(7).std()

df_hour.plot()
plt.title('Распределение заказов по часам');
plt.show()

df_hour['2018-04-01':'2018-04-02'].plot()
plt.title('Распределение заказов по часам за день 1 и 2 апреля 2018 года');
plt.show()

df_hour['2018-04-03':'2018-04-04'].plot()
plt.title('Распределение заказов по часам за день 3 и 4 апреля 2018 года');
plt.show()

df_day = df_taxi.resample('1D').sum()
df_day['mean'] = df_day['num_orders'].rolling(7).mean()
df_day['std'] = df_day['num_orders'].rolling(7).std()

df_day.plot()
plt.title('Распределение заказов по дням');
plt.show()

df_day['2018-04-01':'2018-04-30'].plot()
plt.title('Распределение заказов по дням за апрель');
plt.show()

df_week = df_taxi.resample('1W').sum()
df_week['mean'] = df_week['num_orders'].rolling(4).mean()
df_week['std'] = df_week['num_orders'].rolling(4).std()

df_week.plot()
plt.title('Распределение заказов по неделям');
plt.show()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# 1. Временной ряд не является стационарным, так как у него меняется среднее значение.
# 2. Наблюдаем общий тренд увеличения числа заказов со временем за период с марта по сентябрь.
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
#     
# Есть три вида нестационарности: тренд, сезонность, непостоянство дисперсии)<br>
# Стационарный ряд - это когда ничего из перечисленного в ряде не наблюдается (очистили его). Нужно это для одного из методов прогрнозироваия АРИМА, например. Для метода прогнозирования "регрессия" это понятие можно не использовать.
# </div>

# ### Проведем анализ трендов и сезонности по дням и по часам

# #### Аддетивная модель по дням

# In[4]:


decomposed = seasonal_decompose(df_day['2018-03-10':'2018-09'], model ='additive')
print(df_day.index[11], '- день недели ', df_day.index[11].weekday()) # 0 - понедельник
print(df_day.index[18], '- день недели ', df_day.index[11].weekday())
decomposed.seasonal['2018-03-10':'2018-09'].plot()
plt.title('Seasonality по всему периоду')
plt.show()
decomposed.seasonal['2018-03-12':'2018-03-25'].plot()
plt.title('Seasonality за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')
plt.show()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# Видим, что 
# 1. Закономирности циклично повторяются от недели к неделе и от месяца к месяцу. 
# 2. Максимальное цисло заказов наблюдаем по понедельникам и пятницам, манимальное - по вторникам и воскресеньям.
# </div>

# 

# In[5]:


plt.figure(figsize=(6, 8))
plt.subplot(311)
# Чтобы график корректно отобразился, указываем его
# оси ax, равными plt.gca() (англ. get current axis,
# получить текущие оси)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()


# #### Мультипликативная модель по дням

# In[6]:


decomposed = seasonal_decompose(df_day['2018-03-10':'2018-09'], model = 'multiplicative')
print(df_day.index[11], '- день недели ', df_day.index[11].weekday())
print(df_day.index[18], '- день недели ', df_day.index[11].weekday())
decomposed.seasonal['2018-03-10':'2018-09'].plot()
plt.title('Seasonality по всему периоду')
decomposed.seasonal['2018-03-12':'2018-03-25'].plot()
plt.title('Seasonality за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')


# In[7]:


plt.figure(figsize=(6, 8))
plt.subplot(311)
# Чтобы график корректно отобразился, указываем его
# оси ax, равными plt.gca() (англ. get current axis,
# получить текущие оси)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# Видим, что аддетивная и мультипликативная модели приводят к идентичным выводам:
# 
# 1. Наблюдаем общий тренд увеличения числа заказов со временем.
# 2. закономирности циклично повторяются от недели к неделе. 
# 3. Максимальное цисло заказов наблюдаем по понедельникам и пятницам, манимальное - по вторникам и воскресеньям.
# </div>

# #### Аддетивная модель по часам

# In[8]:


decomposed = seasonal_decompose(df_hour['2018-03-02':'2018-09'], model ='additive')
#print(df_hour.index[11], '- день недели ', df_hour.index[11].weekday())
#print(df_hour.index[18], '- день недели ', df_hour.index[11].weekday())
decomposed.seasonal['2018-03-01':'2018-09'].plot()
plt.title('Seasonality по всему периоду')
decomposed.seasonal['2018-03-12':'2018-03-25'].plot()
plt.title('Seasonality за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')
plt.show()
decomposed.seasonal['2018-03-12':'2018-03-18'].plot()
plt.title('Seasonality за 1 неделю за 12 марта 2018 по 18 марта')
plt.show()
decomposed.seasonal['2018-03-18':'2018-03-19'].plot()
plt.title('Seasonality за 2 дня: c 18 марта 2018 (вс) и за 19 марта 2018 (пн)')
plt.show()


# In[9]:


#NEW_20.01.2022
plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend.plot(ax=plt.gca())
plt.title('Trend за весь период')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality за весь период')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals за весь период')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend['2018-03-12':'2018-03-25'].plot(ax=plt.gca())
plt.title('Trend за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')
plt.subplot(312)
decomposed.seasonal['2018-03-12':'2018-03-25'].plot(ax=plt.gca())
plt.title('Seasonality за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')
plt.subplot(313)
decomposed.resid['2018-03-12':'2018-03-25'].plot(ax=plt.gca())
plt.title('Residuals за 2 недели: с 12 марта 2018 (пн) по 25 марта 2018 (вс)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 8))
plt.subplot(311)
decomposed.trend['2018-03-18':'2018-03-19'].plot(ax=plt.gca())
plt.title('Trend за 2 дня: c 18 марта 2018 (вс) и за 19 марта 2018 (пн)')
plt.subplot(312)
decomposed.seasonal['2018-03-18':'2018-03-19'].plot(ax=plt.gca())
plt.title('Seasonality за 2 дня: c 18 марта 2018 (вс) и за 19 марта 2018 (пн)')
plt.subplot(313)
decomposed.resid['2018-03-18':'2018-03-19'].plot(ax=plt.gca())
plt.title('Residuals за 2 дня: c 18 марта 2018 (вс) и за 19 марта 2018 (пн)')
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# 1. Наблюдаем общий тренд увеличения числа заказов со временем.
# 2. закономирности циклично повторяются от недели к неделе, от дня ко дню. 
# 3. Дневная сезонность: максимальное цисло заказов наблюдаем по понедельникам и пятницам, манимальное - по вторникам и воскресеньям.
# 4. Внутрисуточная сезонность: наблюдаем резкое cнижение числа заказов после 6 утра и резкое увеличение числа заказов в полночь.
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
#     
# Инструмент декомпозиции в частности, а графический анализ временных рядов как в принципе - важный этап моделирования рядов. Позволяет увидеть продажи визуально, а это помогает сделать предвариетльные выводы
#     
# В реальности бизнеса есть все три вида сезонности:
# 1) дневная сезонность -  данные собраны по дням, т.е. будни и выходные имеют разные продажи
# 2) внутрисуточная сезонность: когда продажи утром отличаются от продаж вечером
# 3) месячная сезонность: когда продажи лета и осени - разные.
# 
# Если строить прогноз на длительный период времени, то без учёта всех видов сезонности не обойтись.
# Например, если в данных восходящий тренд, то возникает: 
# этот подъём действительно за счёт роста компании (чаще всего тренд - это отражение роста компании) или есть ещё влияние месячной сезнности? (летом - продаж в рост, например, и именно этот момент мы наблюдаем в данных)
# 
#     
# Совет:
#     
# на горизонте прогнозирования несколько часов или дней предположить что у нас тренд.

# ### Исследуем разности временного ряда (для ресемплирования 1 день и для ресемлирования 1 час)

# In[10]:


df_day_sub = df_day - df_day.shift()
df_day_sub['mean'] = df_day_sub['num_orders'].rolling(7).mean()
df_day_sub['std'] = df_day_sub['num_orders'].rolling(7).std()
df_day_sub.plot()
plt.show()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# После преобразования временной ряд стал более стационарным.
# 
# </div>

# In[11]:


df_hour_sub = df_hour - df_hour.shift()
df_hour_sub['mean'] = df_hour_sub['num_orders'].rolling(10).mean()
df_hour_sub['std'] = df_hour_sub['num_orders'].rolling(10).std()

df_hour_sub['2018-03-01':'2018-09'].plot()
plt.title('График разности за весь период')
plt.show()

df_hour_sub['2018-03-12':'2018-03-18'].plot()
plt.title('График разности за 1 неделю: c 12 марта 2018 по 18 марта')
plt.show()


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# После преобразования временной ряд стал более стационарным.
# </div>

# ## Обучение

# ### По дням

# ### Создадим признаки для модели

# In[12]:


def make_features(data, max_lag, rolling_mean_size):
#    data['year'] = data.index.year - исключено 20.01.2022
#    data['month'] = data.index.month - исключено 20.01.2022
#    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()

make_features(df_day, 2, 7)
print(df_day.head())


# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
# 
# 1) Фактор год: НЕТ, не подходит. У нас данных меньше года. Поэтому будет просто константа.<br>
# 2) Фактор месяц: НЕТ, у нас в истории всего один март, май.... Да и горизоно прогноза - часы. Поэтому сезонность времени года в тренде уже заложена.<br>
# 3) Фактор день: НЕТ, также не подходит, будет просто последовательность чисел. Возможно отдельно сделать маркировку начала или конца месяца (например, 1-3 числа месяца заполнить единицами, оставшиеся дни - нулями.<br>
# 4) Фактор день недели: ДА, т.к. у нас есть недельная сезонность<br>
# 5) Фактор час: ДА, т.к. есть часовая (суточная) сезонность.
# 6) Факто lag: ДА, это параметр авторегрессии - учёт влияния вчерашних событий на сегодняшние.
# 7) Фактор rolling: ДА, это аналог тренда. Чем больше окно сглаживания, тем ровнее линия тренда 
#     
# Факторы mean и std можно не испоьзовать. 
#     Первый (mean) почти совпадает со скользящим средним. 
#     Второй - очень сложный для использования потом. Ведь нам модель надо будет использовать для прогнозирования. Т.е. надо будет сначала построить прогноз std на час вперёд, а потом уже сам прогноз продаж. А std скорее всего будет случайной величиной, а значит предскать будет невозможно.
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Рекоммендация</b>
#     
# Учитывать горизонт прогноза: если задача стоит среднесрочного и долгосрочного прогноза, то месяц нам в помощь (при этом данных должно быть не меннее 2-х / 3-х лет). Если задача краткосрочного прогнозирования, то месяц не использовать, он уже в тренде заложен.
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
#     
# Lag: его называют ещё параметром авторегрессии. 
# Допустим lag=1, он учитывает (для удобства будем считать, что продажи собраны по дням) как вчерашние продажи (вчерашние события) поалияли на сегоняшние. Lag=2: учитывает как позавчерашние продажи влияютна сегодняшние и т.д.
#     
# Скользящее среднее:
# Предположим rolling=10 - скользящее среднее. Мы ищем среднее по 10 точкам, потом по следующим 10 точек, и т.д. - это некое подобие тренда будет - средней линии, т.е. сглаживание колебаний/выбросов. 
# Его легко увидеть на пределе: допустим длина ряда 100 дней, тогда rolling(100) - это просто одно число, среднее всего ряда. 
#     
# Если выставить сезонности - hour и день, то дополнительные факторы могут быть такие: лаг =1 (максимум 2), а скользящее среднее = 24 (среднее за сутки). Начинаем с этих переменных, тогда модель будет приемлемая по точности и без большого числа ненужных факторов, а это значит и для бизнес-подразделений она будет понятнее.
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий к коду</b>
# 
# Число переменных (lag, rolling_mean_size) при вызове фукнции выставлены в соотвествии с комментарием выше:
# 
# 1. lag = 2
# 2. rolling_mean_size = 7 для модели по дням, rolling_mean_size = 24 для модели по часам
# </div>

# #### Разобьем датасет на обучающую и тестовую выборки

# In[13]:


train, test = train_test_split(df_day, shuffle=False, test_size=0.2)
train = train.dropna()
print('Обучающая выборка содержит данные за период с ',train.index.min().strftime('%d %b %Y')      , 'по', train.index.max().strftime('%d %B %Y'))
print(train.iloc[9:12])
print()
print('Тестовая выборка содержит данные за период с ',test.index.min().strftime('%d %b %Y')      , 'по', test.index.max().strftime('%d %B %Y'))
print(test.head(3)) 

print('Размеры обучающей и тестовой выборок: ')
print('Train: ', train.shape)
print('Test: ', test.shape) 


# #### Выделим целевой признак и признаки для обучения

# In[14]:


features_train = train.drop(['num_orders','mean','std'], axis = 1) #Изменено 20.01.2022
target_train = train['num_orders']
features_test = test.drop(['num_orders','mean','std'], axis=1) #Изменено 20.01.2022
target_test = test['num_orders']


# In[15]:


train


# In[16]:


features_train


# In[17]:


test.head(5)


# In[18]:


features_test.head(5)


# #### Обучим модель линейной регрессии

# In[19]:


model = LinearRegression() 
model.fit(features_train, target_train)
prediction_train = model.predict(features_train)
prediction_test = model.predict(features_test)


# #### Расчитаем MAE и RMSE стационарного ряда

# In[20]:


MAE_train = mean_absolute_error(train['num_orders'], prediction_train)
MAE_test = mean_absolute_error(test['num_orders'], prediction_test)
print("MAE обучающей выборки:", MAE_train)
print("MAE тестовой выборки: ", MAE_test)

#расчет RMSE
def rmse(validation, predictions):
    return mean_squared_error(validation, predictions) ** 0.5

rmse_train = rmse(train['num_orders'], prediction_train)
rmse_test = rmse(test['num_orders'], prediction_test)
print("RMSE обучающей выборки:", rmse_train)
print("RMSE тестовой выборки: ", rmse_test)


# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# Получили MAE = 210 и RMSE = 278. Ниже пробуем модель для часа.
# </div>

# ### По часам

# #### Создадим признаки для модели

# In[21]:


df_hour['hour'] = df_hour.index.hour #дополнительный признак по часам #Изменено 20.01.2022
make_features(df_hour, 2, 24) #Изменено 20.01.2022
print(df_hour.head())


# #### Разобьем датасет на обучающую и тестовую выборки

# In[22]:


train, test = train_test_split(df_hour, shuffle=False, test_size=0.2)
train = train.dropna()
print('Обучающая выборка содержит данные за период с ',train.index.min().strftime('%d %b %Y %H')      , 'по', train.index.max().strftime('%d %B %Y %H'))
print(train.iloc[9:12])
print()
print('Тестовая выборка содержит данные за период с ',test.index.min().strftime('%d %b %Y %H')      , 'по', test.index.max().strftime('%d %B %Y %H'))
print(test.head(3)) 

print('Размеры обучающей и тестовой выборок: ')
print('Train: ', train.shape)
print('Test: ', test.shape) 


# #### Выделим целевой признак и признаки для обучения

# In[23]:


features_train = train.drop(['num_orders','mean','std'], axis = 1)
target_train = train['num_orders']
features_test = test.drop(['num_orders','mean','std'], axis=1)
target_test = test['num_orders']


# In[24]:


train


# In[25]:


features_train.head(5)


# In[26]:


test


# In[27]:


features_test.head(5)


# #### Обучим модель линейной регрессии

# In[28]:


model = LinearRegression() 
model.fit(features_train, target_train)
prediction_train = model.predict(features_train)
prediction_test = model.predict(features_test)


# #### Расчитаем MAE и RMSE

# In[29]:


MAE_train = mean_absolute_error(train['num_orders'], prediction_train)
MAE_test = mean_absolute_error(test['num_orders'], prediction_test)
print("MAE обучающей выборки:", MAE_train)
print("MAE тестовой выборки: ", MAE_test)

#расчет RMSE
def rmse(validation, predictions):
    return mean_squared_error(validation, predictions) ** 0.5

rmse_train = rmse(train['num_orders'], prediction_train)
rmse_test = rmse(test['num_orders'], prediction_test)
print("RMSE обучающей выборки:", rmse_train)
print("RMSE тестовой выборки: ", rmse_test)


# 

# <div class="alert alert-block alert-info">
# <b>Вывод</b>
# 
# Получили MAE = 29,3 и RMSE = 46,7. Модель предсказыват будущие заказы по часам с RMSE < 48.
# 
# Нам удалось найти модель c RMSE < 48.
# </div>

# 

# #### Визуализация модели линейной регрессии

# In[30]:


## train['num_orders'], prediction_train
pred_test = pd.Series(prediction_test)
pred_test = pred_test.to_frame(name = 'num_orders')
pred_test = pred_test.set_index(target_test.index)

#pred_test['2018-08-03':'2018-08-05'].plot(figsize=(15,6), label = 'Предсказание')
#target_test['2018-08-03':'2018-08-05']['num_orders'].plot(figsize=(15,6), label = 'Факт')
#plt.show()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(pred_test['2018-08-03':'2018-08-05'], color = 'r', label = 'Предсказание')
ax.plot(target_test['2018-08-03':'2018-08-05'], color = 'g', label = 'Факт');
plt.title('C 3 по 5 августа 2018') 
plt.legend()
ax.grid(color = 'blue'        ,linewidth = 0.2        ,linestyle = '--')
ax.set_xlabel('время')
ax.set_ylabel('число заказов (за час)')
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Комментарий:</b>
#     
# У предсказанной модели меньше разброс значений.
# Глядя на график, а также добавляя отдельный анализ остатков, можно дорабатывать и улучшать модель дальше.
# </div>

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Данные проанализированы
# - [x]  Модель обучена, гиперпараметры подобраны
# - [x]  Качество моделей проверено, выводы сделаны
# - [x]  Значение *RMSE* на тестовой выборке не больше 48

# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
# 
# Леса, бустинги, регрессии.... могут подобрать хорошие модели на старых данных. Но иногда, как прогноз эти, модели могут оказаться пустышками - не интересными бизнесу. <br> 
# 1) Модели нужно подсказать причины, по которым происходит колебания продаж (временного ряда): понимать причины колебаний треда, понимать аномалии, учитывать акционные продажи, видеть действия конкурентов , закладывать их в модель и т.д. 
# 
# 2) Важны три группы факторов: 
# 1)внешние факторы (например выручка такси-компании зависит от количества машин на линии),
# 2)есть факторы внутри самих данных (Вы использовали как раз поняти лаг, сезонности), 
# 3)есть параметры самих моделей (то, что обычно используется при моделировании леса или деревьев).
# 
# 3)  важно иметь ввиду горизонт прогноза: на кокой период строить прогноз нужно  - на час вперёд, на неделю, на месяц.<br> Это накладывает ограничения на подбор моделей
# Удобно составить матрицу  в разрезе:
# а)есть тренд/нет тренда; 
# б)еть сезоннотсь/нет сезонности;
# в)какой горизонт прогноза требуется;
# г) есть ли у нас дополнительные факторы для модей
# 
# На пересечении знаний о моделях легко будет подобрать адкватную модель.
# Не все задачи прогнозирования временных рядов можно хорошо решить лесом и регрессией на основе знаний только факторов ряда. Они хорошо подойдут для коротких горизонтов прогнозирования, а если горизонт чуть больше: недели, месяцы, то тут надо подключать понимание фаткоров продаж самого бизнеса.
# </div>

# <div class="alert alert-block alert-info">
# <b>Комментарий</b>
# 
# Поэтой теме можно посмотреть практические материалы здесь:
# https://ibf.org/knowledge - заморский институт бизнес-прогнозирования.
# 
# 
# Есть классический труд (он НЕ ML, а исключительно на эксель). Но для понимания сути временных рядов - то, что надо:
#     УИЧЕРН "Бизнес-прогнозирование"
# 
# Идеологически по системе прогнозирования, интересно почитать
# ТОМАС УОЛЛАС, Р. СТАЛЬ "планирование продаж и операций" SO&P   
# </div>
