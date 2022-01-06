#!/usr/bin/env python
# coding: utf-8

# Шаги исследования:
# 1. [Открытие данных](#start)
# 2. [Предобработка данных](#preprocessing)
# 3. [Подготовка данных о пользователе](#user)
# 4. [Анализ данных пользователя](#analisys)
# 5. [Проверка гипотез](#hypothesis)
# 6. [Общий вывод](#output)

# <div style="border:solid blue 2px; padding: 20px"> 
# <a id="start">Шаг 1. Открываем файл с данными и изучаем общую информациюх</a>
# </div>

# In[1]:


#Шаг 1. Открываемфайл с данными и изучим общую информацию
import math
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import scipy.stats as st

d_calls    = pd.read_csv("/datasets/calls.csv", sep=",")
d_internet = pd.read_csv("/datasets/internet.csv", sep=",")
d_messages = pd.read_csv("/datasets/messages.csv", sep=",")
d_tariffs  = pd.read_csv("/datasets/tariffs.csv", sep=",")
d_users    = pd.read_csv("/datasets/users.csv", sep=",")
#1. Calls
print('1. Calls')
display(d_calls.head(5))
display(d_calls.describe())
d_calls.info()
#2. Internet
print('2. Internet')
display(d_internet.head(5))
display(d_internet.describe())
d_internet.info()
#3. Messages
print('3. Messages')
display(d_messages.head(5))
display(d_messages.describe())
d_messages.info()
#4. Tariffs
print('4. Tariffs')
display(d_tariffs.head(5))
display(d_tariffs.describe())
d_tariffs.info()
#5. Users
print('5. Users')
display(d_users.head(5))
display(d_users.describe())
d_users.info()


# <div class="alert alert-info">
# <b>Вывод: </b> 
#     
# Можем 
#     
# 1. уточнить форматы дат в таблицах
# 
# 2. округлить мегабайты трафика интернет
# </div>

# <div style="border:solid blue 2px; padding: 20px"> 
# <a id="preprocessing">
#     Шаг 2. Предобработка данных
# </a>
#         
#     Приводим данные к нужным типам;
#     Исправляем ошибки в данных.
# </div>

# In[2]:


#Шаг 2. Предобработка данных
#1). Преобразование дат, добавление месяца
###приводим в формат datetime
d_calls['call_date']       = pd.to_datetime(d_calls['call_date']
                                            , format='%Y-%m-%d')
d_internet['session_date'] = pd.to_datetime(d_internet['session_date']
                                            , format='%Y-%m-%d')
d_messages['message_date'] = pd.to_datetime(d_messages['message_date']
                                            , format='%Y-%m-%d')
d_users['reg_date']        = pd.to_datetime(d_users['reg_date']
                                            , format='%Y-%m-%d')
d_users['churn_date']      = pd.to_datetime(d_users['churn_date']
                                            , format='%Y-%m-%d')
###добавляем месяц
d_calls['month'] = pd.to_datetime(d_calls['call_date']).dt.month
###или вот так  d_calls['month_number'] = pd.DatetimeIndex(d_calls['call_date']).month
d_internet['month'] = pd.to_datetime(d_internet['session_date']).dt.month
d_messages['month'] = pd.to_datetime(d_messages['message_date']).dt.month
d_users['reg_month']    = pd.to_datetime(d_users['reg_date']).dt.month
d_users['churn_month']    = pd.to_datetime(d_users['churn_date']).dt.month
#2). Округляем мегабайты трафика интернет
#d_calls['duration'] = math.ceil(d_calls['duration']).astype('int')
def ceil_value(value):
    return math.ceil(value)
d_calls['duration'] = d_calls['duration'].apply(ceil_value).astype('int')
d_internet['mb_used'] = d_internet['mb_used'].apply(ceil_value).astype('int')
#1. Calls
print('1. Calls')
display(d_calls.head(5))
d_calls.info()
#2. Internet
print('2. Internet')
display(d_internet.head(5))
d_internet.info()
#3. Messages
print('3. Messages')
display(d_messages.head(5))
d_messages.info()
#4. Tariffs
print('4. Tariffs')
display(d_tariffs.head(5))
d_tariffs.info()
#5. Users
print('5. Users')
display(d_users.head(5))
d_users.info()


# <div style="border:solid blue 2px; padding: 20px"> 
# <a id="user">
#     Шаг 3. Подготовка данных о пользователе
# </a>
# </div>

# In[21]:


#Посчитайте для каждого пользователя:
#1. количество сделанных звонков и израсходованных минут разговора по месяцам;
calls_by_month = d_calls.pivot_table(index=['user_id', 'month']
                                     , values='duration'
                                     , aggfunc=['sum', 'count']
                                    )
calls_by_month.sort_values(by=['user_id'], ascending=False).reset_index()
calls_by_month.columns = ['calls_duration', 'calls_amount']
display(calls_by_month.head(5))
###Заменить сводную табицу на обычную
calls_by_month = calls_by_month.reset_index()
display(calls_by_month.head(5))
#calls_by_month.columns = calls_by_month.columns.droplevel(0)
#сalls_by_month = calls_by_month.reset_index().rename_axis(None, axis=1)
#2. количество отправленных сообщений по месяцам;
messages_by_month = d_messages.pivot_table(index=['user_id', 'month']
                                           , values='id'
                                           , aggfunc=['count']
                                          )
messages_by_month.sort_values(by=['user_id'], ascending=False).reset_index()
messages_by_month.columns = ['messages']
messages_by_month['messages'] = messages_by_month['messages'].astype('int')
messages_by_month = messages_by_month.reset_index()
#3. объем израсходованного интернет-трафика по месяцам;
internet_by_month = d_internet.pivot_table(index=['user_id', 'month']
                                           , values='mb_used'
                                           , aggfunc=['sum']
                                          )
internet_by_month.sort_values(by=['user_id'], ascending=False).reset_index()
internet_by_month.columns = ['mb_used']
internet_by_month = internet_by_month.reset_index()
#4. помесячную выручку с каждого пользователя
#--smart-----------------------------------
#--messages_included = 50,--mb_per_month_included = 15360--minutes_included = 500
#--rub_monthly_fee = 550,--rub_per_gb = 200,--rub_per_message =3,--rub_per_minute = 3
#for row in all_by_month

all_by_month = calls_by_month.merge(messages_by_month, on=['user_id', 'month'], how='left')
all_by_month.info()
all_by_month = all_by_month.merge(internet_by_month, on=['user_id', 'month'], how='left')
all_by_month = all_by_month.merge(d_users, on=['user_id'], how='left')
all_by_month.info()
all_by_month = all_by_month.merge(d_tariffs, left_on=['tariff'], right_on=['tariff_name'], how='left')
display(all_by_month.head(5))
display('all_by_month',all_by_month.count())
display('internet_by_month',internet_by_month.count())
display('messages_by_month',messages_by_month.count())


# <div style="border:solid blue 2px; padding: 20px"> 
# <a id="analisys">
#     Шаг 4. Анализ данных пользователя
# </a>
# </div>

# In[4]:


#Посчитаем выручку от сообщений/интернета/звонков
###вычтем бесплатный лимит из суммарного количества звонков, сообщений и интернет-трафика; 
###остаток умножьте на значение из тарифного плана; 
###прибавьте абонентскую плату, соответствующую тарифному плану.

def users_by_month(row):   
    if row['messages'] > row['messages_included']:
        revenue_message = (row['messages']-row['messages_included'])*row['rub_per_message']
    else:
        revenue_message = 0
    if row['mb_used'] > row['mb_per_month_included']:
        revenue_internet = math.ceil((row['mb_used']-row['mb_per_month_included'])/1024)*row['rub_per_gb']
    else:
        revenue_internet = 0
    if row['calls_duration'] > row['minutes_included']:
        revenue_calls = (row['calls_duration']-row['minutes_included'])*row['rub_per_minute']
    else:
        revenue_calls = 0
    total_revenue = revenue_message + revenue_internet + revenue_calls + row['rub_monthly_fee']
    return total_revenue

all_by_month['total_revenue'] = all_by_month.apply(users_by_month, axis=1)
display(all_by_month.head(20))
all_by_month.info()


# In[5]:


#Шаг 3. Проанализируем данные #Опишем поведение клиентов оператора, исходя из выборки. 
#Сколько минут разговора, сколько сообщений и какой объём интернет-трафика 
#требуется пользователям каждого тарифа в месяц? 
#Добавим информацию о сообщениях, интернете и звонках, используемых сверх лимита
def over_limit(row, column1, column2):   
    if row[column1] > row[column2]:
        over_limit_ = row[column1]-row[column2]
    else:
        over_limit_ = 0
    return over_limit_
all_by_month['over_limit_messages'] = all_by_month.apply(over_limit
                                                         , axis=1
                                                         , column1 = 'messages'
                                                         , column2 ='messages_included')
all_by_month['over_limit_internet'] = all_by_month.apply(over_limit
                                                         , axis=1
                                                         , column1 = 'mb_used'
                                                         , column2 ='mb_per_month_included')
all_by_month['over_limit_calls'] = all_by_month.apply(over_limit
                                                         , axis=1
                                                         , column1 = 'calls_duration'
                                                         , column2 ='minutes_included')
display(all_by_month.head(20))


# In[6]:


#Посчитаем среднее количество, дисперсию и стандартное отклонение. 
smart = all_by_month.query('tariff == "smart"')
print('-----------------------')
print('По тарифу SMART, в среднем за месяц:')
print('-----------------------')
print('Минут разговора:', round(smart['calls_duration'].describe()['mean']))
print('Сообщений:', round(smart['messages'].describe()['mean']))
print('Мегабайт интернета:', round(smart['mb_used'].describe()['mean']))
print('')
print('Стандартное отклонение минут разговора:', round(smart['calls_duration'].describe()['std']))
print('Стандартное отклонение сообщений:', round(smart['messages'].describe().describe()['std']))
print('Стандартное отклонение мегабайт интернета:', round(smart['mb_used'].describe()['std']))
print('')
print('Дисперсия минут разговора:', round(smart['calls_duration'].describe()['std']** 2))
print('Дисперсия сообщений:', round(smart['messages'].describe().describe()['std']** 2))
print('Дисперсия мегабайт интернета:', round(smart['mb_used'].describe()['std']**2))
print('')
print('Среднее превышение лимита разговора:', round(smart['over_limit_calls'].describe()['mean']))
print('Среднее превышение лимита сообщений:', round(smart['over_limit_messages'].describe()['mean']))
print('Среднее превышение лимита интернета (в мб):', round(smart['over_limit_internet'].describe()['mean']))
print('')


# In[7]:


ultra = all_by_month.query('tariff == "ultra"')
print('-----------------------')
print('По тарифу ULTRA, в среднем за месяц:')
print('-----------------------')
print('Минут разговора:', round(ultra['calls_duration'].describe()['mean']))
print('Сообщений:', round(ultra['messages'].describe()['mean']))
print('Мегабайт интернета:', round(ultra['mb_used'].describe()['mean']))
print('')
print('Стандартное отклонение минут разговора:', round(ultra['calls_duration'].describe()['std']))
print('Стандартное отклонение сообщений:', round(ultra['messages'].describe().describe()['std']))
print('Стандартное отклонение мегабайт интернета:', round(ultra['mb_used'].describe()['std']))
print('')
print('Дисперсия минут разговора:', round(ultra['calls_duration'].describe()['std']** 2))
print('Дисперсия сообщений:', round(ultra['messages'].describe().describe()['std']** 2))
print('Дисперсия мегабайт интернета:', round(ultra['mb_used'].describe()['std']**2))
print('')
print('Среднее превышение лимита разговора:', round(ultra['over_limit_calls'].describe()['mean']))
print('Среднее превышение лимита сообщений:', round(ultra['over_limit_messages'].describe()['mean']))
print('Среднее превышение лимита интернета (в мб):', round(ultra['over_limit_internet'].describe()['mean']))
print('')

#В тарифе ULTRA люди не превышают лимит разговора и сообщений, 
#превышение лимита интернета в среднем ниже, чем в SMART.


# <div style="border:solid blue 2px; padding: 20px"> 
# Вывод:
#     
# В тарифе ULTRA люди не превышают лимит разговора и сообщений, превышение лимита интернета в среднем ниже, чем в SMART.
# </div>   

# In[12]:


#Посмотим на распределение числа пользователей по тарифам.
print('-----------')
print('Тариф SMART')
print('-----------')
smart.groupby('month')['user_id'].count().plot(x='month',y='user_id',kind='bar') 
plt.title('Распределение числа пользователей тарифом по месяцам')
plt.show()
    
print('-----------')
print('Тариф ULTRA')
print('-----------')
ultra.groupby('month')['user_id'].count().plot(x='month',y='user_id',kind='bar') 
plt.title('Распределение числа пользователей тарифом по месяцам')
plt.show()

#число пользователей по тарифам к концу года возрастает


# <div style="border:solid blue 2px; padding: 20px"> 
# Вывод:
#     
# Число пользователей по тарифам к концу года возрастает
# </div>

# In[13]:


#Постройте гистограммы. Опишем распределения.

columns = ['calls_duration'
           , 'messages'
           , 'mb_used'
           ,'over_limit_internet'
           ,'total_revenue'
          ]
print('-----------')
print('Тариф SMART')
print('-----------')
#smart.groupby('month')['calls_duration'].mean().plot(x='month',y='calls_duration',kind='bar')
for i in columns:
    smart.groupby('month')[i].mean().plot(x='month',y=i,kind='bar') 
    if i == 'calls_duration':
        plt.title('Распределение средней длительности звонков по месяцам')
    if i == 'messages':
        plt.title('Распределение среднего числа сообщений по месяцам')
    if i == 'mb_used':
        plt.title('Распределение среднего использования интернета в мб по месяцам') 
    if i == 'over_limit_internet':
        plt.title('Распределение интернета сверх лимита в мб в среднем по месяцам') 
    if i == 'total_revenue':
        plt.title('Распределение доходности в среднем на одного пользователя') 
    plt.show()
    
print('-----------')
print('Тариф ULTRA')
print('-----------')
#smart.groupby('month')['calls_duration'].mean().plot(x='month',y='calls_duration',kind='bar')
for i in columns:
    ultra.groupby('month')[i].mean().plot(x='month',y=i,kind='bar') 
    if i == 'calls_duration':
        plt.title('Распределение средней длительности звонков по месяцам')
    if i == 'messages':
        plt.title('Распределение среднего числа сообщений по месяцам')
    if i == 'mb_used':
        plt.title('Распределение среднего использования интернета в мб по месяцам')      
    if i == 'over_limit_internet':
        plt.title('Распределение интернета сверх лимита в мб в среднем по месяцам') 
    if i == 'total_revenue':
        plt.title('Распределение доходности в среднем на одного пользователя') 
    plt.show()
#Клиенты наименее активно пользуются тарифом "SMART" в январе,
#а тарифом "ULTRA" в феврале.
#К декабрю активность использования обох тарифов возрастает.
#Средняя доходность по тарифу "SMART" возрастает к концу года,
#в то время как по тарифу "ULTRA" в среднем на пользователя в течение года не меняется.


# <div style="border:solid blue 2px; padding: 20px"> 
# Вывод:
#     
# Клиенты наименее активно пользуются тарифом "SMART" в январе, а тарифом "ULTRA" в феврале.
# К декабрю активность использования обох тарифов возрастает.
# Средняя доходность по тарифу "SMART" возрастает к концу года, в то время как по тарифу "ULTRA" в среднем на пользователя в течение года не меняется.
# </div>

# In[19]:


#Посмотрим на распределение среднего дохода на одного пользователя по тарифам
print('-----------')
print('Тариф SMART')
print('-----------')
smart_user = smart.groupby('user_id')['total_revenue'].sum()
smart_user = smart_user.reset_index()
print('Минимальная доходность -', round(smart_user['total_revenue'].min()))
print('Максимальная доходность -', round(smart_user['total_revenue'].max()))
smart_user['total_revenue'].hist(bins=20,range=(550,28000), label = 'Доходность пользователя SMART')
plt.title('Гистограмма доходности пользователя тарифа SMART')
plt.show()
print('-----------')
print('Тариф ULTRA')
print('-----------')
ultra_user = ultra.groupby('user_id')['total_revenue'].sum()
ultra_user = ultra_user.reset_index()
print('Минимальная доходность -', round(ultra_user['total_revenue'].min()))
print('Максимальная доходность -', round(ultra_user['total_revenue'].max()))
ultra_user['total_revenue'].hist(bins=20,range=(550,28000), label = 'Доходность пользователя ULTRA')
plt.title('Гистограмма доходности пользователя тарифа ULTRA')
plt.show()


# <div style="border:solid blue 2px; padding: 20px"> 
# <a id = "hypothesis">
#     Шаг 5: Проверка гипотез.
# </a>
# </div>

# In[15]:


#1) Проверим гипотезу "Распределения обучающей и тестовой выборки равны"

#Пороговое значение alpha - 0.05
alpha = 0.05 #если p-value окажется меньше него - отвергнем гипотезу
#Нулевая гипотеза - средняя выручка пользователей тарифов «Ультра» и «Смарт» равны. 
#Альтернативной гипотезой - cредняя выручка пользователей тарифов «Ультра» и «Смарт» различается.
#Использую метод scipy.stats.ttest_ind (array1, array2, equal_var = False)
tariff_results = st.ttest_ind(smart_user['total_revenue'], ultra_user['total_revenue'],equal_var = False)
print('p-значение:', tariff_results.pvalue)

#p-значение очень мало, таким образом средняя выручка пользователей тарифов «Ультра» и «Смарт» различается


# <div style="border:solid blue 2px; padding: 20px"> 
# Вывод:
#     
# p-значение очень мало, таким образом средняя выручка пользователей тарифов «Ультра» и «Смарт» различается
# </div>    

# In[16]:


#2) Проверим гипотезу "Cредняя выручка пользователи из Москвы отличается 
#от выручки пользователей из других регионов."
moscow = all_by_month.query('city == "Москва"')
region = all_by_month.query('city != "Москва"')
moscow_user = moscow.groupby('user_id')['total_revenue'].sum()
moscow_user = moscow_user.reset_index()
region_user = region.groupby('user_id')['total_revenue'].sum()
region_user = region_user.reset_index()

moscow_results = st.ttest_ind(moscow_user['total_revenue'], region_user['total_revenue'],equal_var = False)
print('p-значение:', moscow_results.pvalue)
#p-значение велико, таким образом средняя выручка пользователи из Москвы не отличается от выручки пользователей из других регионов.
#Можно также проверять нормальность распределения,например, тестом Шапиро-Уилка https://www.machinelearningmastery.ru/a-gentle-introduction-to-normality-tests-in-python/


# <div style="border:solid blue 2px; padding: 20px"> 
# Вывод:
#     
# p-значение велико, таким образом средняя выручка пользователи из Москвы не отличается от выручки пользователей из других регионов.
# </div>

# <div class="alert alert-info">
# <a id = "output">
# <b>Общий вывод: </b> 
# </a>
#     
# 1. Cредняя выручка от пользователей тарифов «Ультра» выше выручки от пользователей тарифа «Смарт».
# 
# 2. Пользователи тарифа «Ультра» не превышают лимит разговора и сообщений в отличие от пользователей тарифа «Смарт». Превышение лимита интернета у пользователей тарифа «Ультра» также в среднем ниже, чем у пользователей тарифа «Смарт».
# 
# 3. Средняя доходность на пользователя тарифа «Ультра» расномерно распределена в течение года,
#     в то время как средняя доходность на пользователя тарифа «Смарт» значительно ниже в начале года и возрастает к декабрю.
#     
# 4. Cредняя выручка пользователи из Москвы не отличается от выручки пользователей из других регионов.
# </div>
