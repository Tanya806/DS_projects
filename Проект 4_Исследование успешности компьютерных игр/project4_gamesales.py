#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Short description:</b>
# 
# Historical data on game sales, user and expert ratings, genres and platforms (for example, Xbox or PlayStation) are available from open sources. 
# I identified the patterns that determine the success of the game. This will allow you to bet on a potentially popular product and plan your advertising campaigns.
# </div>

# Шаги исследования:
# 1. [Открытие данных](#start)
# 2. [Предобработка данных](#preprocessing)
# 3. [Проведение исследовательского анализа данных](#analisys)
# 4. [Составление портрета пользователя каждого региона](#user)
# 5. [Проверка гипотез](#hypothesis)
# 6. [Общий вывод](#output)

# <div style="border:solid blue 2px; padding: 20px"> 
#     
# <a id="start">**Шаг 1. Открытие файла с данными и изучаем общую информацию**</a>
# 
# Путь к файлу: /datasets/games.csv. Скачать датасет
# </div>

# In[2]:


import math
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import scipy.stats as st
import plotly.express as px

d_games = pd.read_csv("/datasets/games.csv", sep=",")
display(d_games.head(5))
display(d_games.describe())
d_games.info()
display(d_games.isnull().sum())


# <div class="alert alert-info">
# <b>Вывод: </b> 
# 
# Следующие поля содержат пропуски:
#     
# - Name                  2
# 
# - Year_of_Release     269
#     
# - Genre                 2
#     
# - Critic_Score       8578
#     
# - User_Score         6701
#     
# - Rating             6766
# 
# Пустые значения свидетельствуют, что для некоторых игр исторические данные о дате выпуска, оценке критиков, оценке пользователей и экспертов, рейтинге от организации ESRB недоступны.
# Причины могут быть разные: скажем, данные игры не оценивались. Возможны также проблемы с записью данных.
# </div>

# <div style="border:solid blue 2px; padding: 20px"> 
# 
# <a id="preprocessing">**Шаг 2. Подготовьте данные**</a>
#     
# Заменим названия столбцов (приведем к нижнему регистру);    
# </div>

# In[3]:


d_games.columns = [x.lower() for x in d_games.columns]
display(d_games.head(5))
display(d_games[d_games['name'].isna()]) 


# <div style="border:solid blue 2px; padding: 20px"> 
# 
#     Преобразуем данные в нужные типы.    
#     Опишем, в каких столбцах заменили тип данных и почему.
#     
#     Обработаем пропуски при необходимости:
#     Объясним, почему заполнили пропуски определённым образом или почему не стали это делать;
#     Опишем причины, которые могли привести к пропускам;
#     Обратим внимание на аббревиатуру 'tbd' в столбцах с рейтингом. 
#     Отдельно разберем это значение и опишем, как его обработать.
# </div>

# In[4]:


#0. Поиск полных дубликатов
print("Число полных дубликатов строк в таблице:", d_games.duplicated().sum())
#Полные дубликаты отсутствуют
#1. Проверим, есть ли проблемы с регистрами
#print(d_games['genre'].unique()) 
print(d_games['genre'].value_counts()) 
#print(d_games['platform'].unique()) 
print(d_games['platform'].value_counts()) 
#2. Name и genre, обнаружено два пропуска. 
#Предполагаю, что продажи даже таких игр должны быть учтены.
d_games['name'] = d_games['name'].fillna('noname')
d_games['name'] = d_games['name'].str.lower() #переведем названия в нижний регистр
d_games['genre'] = d_games['genre'].fillna('nogenre')
#display(d_games[d_games['platform']== 'PS2']) 
#3. Year_of_Release, 269 пропуска
display(d_games[d_games['year_of_release'].isna()]) 
print('Минимальный год выпуска игр', d_games['year_of_release'].min())
#пусть 1900 - значение, если не заполнен год. Вводим значение п умолчаю, если не заполнен
d_games['year_of_release'] = d_games['year_of_release'].fillna(1900.0)
d_games['year_of_release'] = pd.to_numeric(d_games['year_of_release'],errors = 'coerce') 
                                           #.astype('int')
#4. Critic_score - 8578
#d_games['critic_score'] = d_games['critic_score'].fillna(0.0)
#display(d_games[d_games['critic_score'] == 0.0]) 
#d_games['critic_score'] = d_games['critic_score'].astype('float')
d_games['critic_score'] = pd.to_numeric(d_games['critic_score'],errors = 'coerce') 
#5. User_score 6701
print('Оценка пользователей:' , d_games['user_score'].unique())
d_games['user_score'] = d_games['user_score'].replace('tbd', np.nan)#Заменим tbd на Nan (преднамеренно не заполнен скор)
d_games['user_score'] = pd.to_numeric(d_games['user_score'],errors = 'coerce') 
##https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
#6. Rating - 6766
print('Рейтинг от организации ESRB:', d_games['rating'].unique())
d_games['rating'] = d_games['rating'].fillna('undefined')


# <div style="border:solid blue 2px; padding: 20px"> 
# Посчитаем суммарные продажи во всех регионах и запишем их в отдельный столбец.
# </div>

# In[5]:


d_games['total_sales'] = d_games['na_sales'] + d_games['eu_sales'] + d_games['jp_sales'] + d_games['other_sales']
print(d_games['total_sales'].head(4))


# <div style="border:solid blue 2px; padding: 20px"> 
# 
# <a id="analysis">**Шаг 3. Проведение исследовательского анализа данных**</a>
# 
# Посмотрим, сколько игр выпускалось в разные годы. Важны ли данные за все периоды?
# 
# Посмотрим, как менялись продажи по платформам. Выберем платформы с наибольшими суммарными продажами и постройте распределение по годам. За какой характерный срок появляются новые и исчезают старые платформы?
# </div>

# In[6]:


games_by_year = d_games.pivot_table(index='year_of_release', values='name', aggfunc='count')
games_by_year = games_by_year.reset_index()
print(games_by_year.head(4))
#Как менялись платформы
#Распределение по годам


# <div style="border:solid blue 2px; padding: 20px"> 
# Возьмем данные за соответствующий актуальный период. Актуальный период определим самостоятельно в результате исследования предыдущих вопросов. Основной фактор — эти данные помогут построить прогноз на 2017 год.
# 
# 
# Не будем учитывать в работе данные за предыдущие годы.
# Какие платформы лидируют по продажам, растут или падают? Выберем несколько потенциально прибыльных платформ.
# </div>

# In[7]:


#Актуальный период -  с 1980 по 2016
games_by_year[games_by_year['year_of_release']>1900].boxplot(column='year_of_release')
plt.show()
games_by_year[games_by_year['year_of_release']>1900].plot(x='year_of_release',y='name',kind='bar')
plt.title("Количество игр выпускаемые в разные года")
plt.xlabel("Год выпуска")
plt.ylabel("Количество выпущенных игр")
plt.show()


# <div style="border:solid blue 2px; padding: 20px"> 
# Построим график «ящик с усами» по глобальным продажам игр в разбивке по платформам. Опишем результат.
# </div>

# In[8]:


#--Смотрим на актуальность платформ (перенесла код ревьюера)
import plotly.express as px

fig = px.line(d_games.query('year_of_release > 2000').pivot_table(index = ['year_of_release', 'platform'], values = 'total_sales', aggfunc = 'sum').reset_index(),
             x = 'year_of_release', y = 'total_sales', color = 'platform', title = 'Изменение суммарных продаж по платформам', template = 'plotly_dark')
fig.show()
#--Добавляем новый  dataframe с отсечением по дате
d_games_2016 = d_games[d_games['year_of_release']>=2016]
platform_sales = d_games_2016.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)
platform_sales = platform_sales.reset_index()
plt.figure(figsize=(13,6))
platform_sales.plot(x='platform',y='total_sales', kind = 'bar')
plt.title("Продажи по платформам")
plt.xlabel("Название платформы")
plt.ylabel("Число продаж")
plt.show()
top_platform = platform_sales.loc[platform_sales['total_sales']>5,'platform']
plt.figure(figsize=(12,6))
print(top_platform)
#print(d_games[d_games['platform'].isin(top_platform)])
d_games_top_platform = d_games_2016.loc[d_games_2016['platform'].isin(top_platform)]
#d_games_top_platform['total_sales'].describe()
print(d_games_top_platform.loc[d_games_top_platform['total_sales']>0.6, 'name'].count())
#print(d_games_top_platform.loc[d_games_top_platform['total_sales']<0.02, 'name'].count()) 
#Уберем выбросы
d_games_top_platform = d_games_top_platform[(d_games_top_platform['total_sales']<0.6)]
#Нарисуем график с усами
d_games_top_platform[d_games_top_platform['platform'].isin(top_platform)].boxplot(column='total_sales', by ='platform')
plt.xlabel('Платформа')
plt.ylabel('Глобальные продажи')
plt.title('Ящик с усами')
plt.show()


# In[9]:


#Датафрейм для дальнейшего анализа - с 2013 года
d_games_2013 = d_games.query('year_of_release >= 2013')


# In[10]:


import plotly.express as px

fig = px.line(d_games_2013.pivot_table(index = ['year_of_release', 'platform'], values = 'total_sales', aggfunc = 'sum').reset_index(),
             x = 'year_of_release', y = 'total_sales', color = 'platform', title = 'Изменение суммарных продаж по платформам', template = 'plotly_dark')
fig.show()


# <div style="border:solid blue 2px; padding: 20px"> 
# Посмотрим, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Постройте диаграмму рассеяния и посчитайте корреляцию между отзывами и продажами. Сформулируйте выводы.
# Соотнесем выводы с продажами игр на других платформах.
# </div>

# In[11]:


#PS4 
#Корреляция между оценками пользователей и продажами 
ps4 = d_games_2013[d_games_2013['platform']=='PS4']
print('--------------------')
print('Игры на платформе PS4')
print('--------------------')
print('1. Корреляция между оценками пользователей и продажами:',ps4['user_score'].corr(ps4['total_sales']))
print('2. Корреляция между оценками критиков и продажами:',ps4['critic_score'].corr(ps4['total_sales']))
ps4.plot(x='user_score', y = 'total_sales', kind = 'scatter',grid=True)
plt.show()


# In[12]:


#XOne
#Корреляция между оценками пользователей и продажами 
XOne = d_games_2013[d_games_2013['platform']=='XOne']
print('--------------------')
print('Игры на платформе XOne')
print('--------------------')
print('1. Корреляция между оценками пользователей и продажами:',XOne['user_score'].corr(XOne['total_sales']))
print('2. Корреляция между оценками критиков и продажами:',XOne['critic_score'].corr(XOne['total_sales']))
XOne.plot(x='user_score', y = 'total_sales', kind = 'scatter',grid=True)
plt.show()


# In[13]:


#3DS
#Корреляция между оценками пользователей и продажами 
DS = d_games_2013[d_games_2013['platform']=='3DS']
print('--------------------')
print('Игры на платформе 3DS')
print('--------------------')
print('1. Корреляция между оценками пользователей и продажами:',DS['user_score'].corr(DS['total_sales']))
print('2. Корреляция между оценками критиков и продажами:',DS['critic_score'].corr(DS['total_sales']))
DS.plot(x='user_score', y = 'total_sales', kind = 'scatter',grid=True)
plt.show()


# In[14]:


#PC
pc = d_games_2013[d_games_2013['platform']=='PC']
print('--------------------')
print('Игры на платформе PC')
print('--------------------')
print('1. Корреляция между оценками пользователей и продажами:', pc['user_score'].corr(pc['total_sales']))
print('2. Корреляция между оценками критиков и продажами:', pc['critic_score'].corr(pc['total_sales']))
pc.plot(x='user_score', y = 'total_sales', kind = 'scatter',grid=True)
plt.show()


# <div class="alert alert-info">
# <b>Вывод: </b> 
# 
# - Наблюдается слабая корреляция между оценками критиков и продажами.
# 
# - Корреляции между оценками пользователей и продажами нет.
# 
# - Большинство дорогих игр получают более высокие оценки критиков.
# </div>

# <div style="border:solid blue 2px; padding: 20px"> 
# Посмотрим на общее распределение игр по жанрам. 
#     
# Что можем сказать о самых прибыльных жанрах? 
# Выделяются ли жанры с высокими и низкими продажами?
# </div>

# In[15]:


display(d_games_2013['genre'].value_counts())
genre_sales = d_games_2013.pivot_table(index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)
genre_sales = genre_sales.reset_index()
plt.figure(figsize=(12,6))
genre_sales.plot(x='genre',y='total_sales', kind = 'bar')
plt.title('Распределение игр по жанрам')
plt.xlabel('Жанры')
plt.ylabel('Продажи')
plt.show()


# In[16]:


display(d_games_2013['genre'].value_counts())
genre_sales = d_games_2013.pivot_table(index='genre', values='total_sales', aggfunc='mean').sort_values(by='total_sales', ascending=False)
genre_sales = genre_sales.reset_index()
plt.figure(figsize=(12,6))
genre_sales.plot(x='genre',y='total_sales', kind = 'bar')
plt.title('Распределение игр по жанрам')
plt.xlabel('Жанры')
plt.ylabel('Продажи')
plt.show()


# <div class="alert alert-info">
# <b>Вывод: </b> 
#     
# В среднем самые доходные жанры - это Shooter, Sports, Platform и Role-playing. Также высок суммарный доход жанра Action. Наименьший доход принесли Puzzle и Strategy игры.
# </div>

# <div style="border:solid blue 2px; padding: 20px"> 
# 
# <a id="user">**Шаг 4. Составление портрета пользователя каждого региона**</a>
# 
# Определим для пользователя каждого региона (NA, EU, JP):
# Самые популярные платформы (топ-5). Опишем различия в долях продаж.
# Самые популярные жанры (топ-5). Поясним разницу.
# Влияет ли рейтинг ESRB на продажи в отдельном регионе?
# </div>

# In[17]:


#Определим для пользователя каждого региона (NA, EU, JP): Самые популярные платформы (топ-5).
columns_to_compare = ['platform','genre','rating']

#EU
print('---')
print('EU')
print('---')
for i in columns_to_compare:
    top_eu = d_games_2013.pivot_table(index=i
                          , values=['eu_sales','total_sales']
                          , aggfunc='sum').sort_values(by='eu_sales'
                                                       , ascending=False).head(5).reset_index()
    top_eu['sales_share']= top_eu['eu_sales']/top_eu['total_sales']
    display(top_eu)
    top_eu.plot(x= i,y = 'sales_share', kind = 'bar')
    plt.show()
#    if i == 'platform':
#        top_platforms_eu = top_eu
#    elif i =='genre':
#        top_genre_eu = top_eu
#    elif i =='rating':
#        top_rating_eu = top_eu
 
#platform_sales.plot(x='platform',y='total_sales', kind = 'bar')
#plt.title("Продажи по платформам")
#plt.xlabel("Название платформы")
#plt.ylabel("Число продаж")
#plt.show()   
    


# In[18]:


#NA
print('---')
print('NA')
print('---')
for i in columns_to_compare:
    top_na = d_games_2013.pivot_table(index=i
                          , values=['na_sales','total_sales']
                          , aggfunc='sum').sort_values(by='na_sales'
                                                       , ascending=False).head(5).reset_index()
    top_na['sales_share']= top_na['na_sales']/top_na['total_sales']
    display(top_na)
    top_na.plot(x= i,y = 'sales_share', kind = 'bar')
    plt.show()


# In[19]:


#JP
print('---')
print('JP')
print('---')
for i in columns_to_compare:
    top_jp = d_games_2013.pivot_table(index=i
                                 , values=['jp_sales','total_sales']
                                 , aggfunc='sum').sort_values(by='jp_sales'
                                                              , ascending=False).head(5).reset_index()
    top_jp['sales_share']= top_jp['jp_sales']/top_jp['total_sales']
    display(top_jp)
    top_jp.plot(x= i,y = 'sales_share', kind = 'bar')
    plt.show()


# <div class="alert alert-block alert-info">
# <b>Рейтинг ESRB (из википедии):</b>
#     
# - «EC» («Early childhood») — «Для детей младшего возраста»
#         
# - «E» («Everyone») — «Для всех»
#         
# - «E10+» («Everyone 10 and older») — «Для всех от 10 лет и старше»
#         
# - «T» («Teen») — «Подросткам»
#         
# - «M» («Mature») — «Для взрослых»
#         
# - «AO» («Adults Only 18+») — «Только для взрослых»
#         
# - «RP» («Rating Pending») — «Рейтинг ожидается»
# </div>

# <div class="alert alert-info">
# <b>Вывод: </b> 
# 
# Европа:
#     
#     топ платформ: PS4, PS3, XOne
#     топ жанров: Shooter, Sports, Action
#     топ рейтингов ESRB: «Для всех» , «Для всех от 10 лет и старше», «Для взрослых»
#     
# Северная Америка:
#     
#     топ платформ: PS4, XOne, X360
#     топ жанров: Shooter, Action, Sports
#     топ рейтингов ESRB: «Для всех» , «Для всех от 10 лет и старше», «Для взрослых»
# 
# Япония:
#     
#     топ платформ: 3DS, PS3, PSV
#     топ жанров: Role-playing, Action 
#     топ рейтингов ESRB: «Подросткам», «Для всех», «Для взрослых»
# </div>

# <div class="alert alert-info">
# <b>Выводы по итогам анализа данных: </b> 
#     
# 1. Значительный рост числа игр наблю дается с 1993 года до 2008 года. С 2008 года общее число игр начинает сокращаться. Вероятно это связано с высокой популярностью мобильных игр и приложений.
#     
# 2. С течением времени ранее популярные платформы становятся неактуальными
#     
# 3. Самые популярные игровые платформы 2016 года: PS4, XOne, 3DS, PC.
#     
# 4. У всех платформ наблюдается слабая корреляция между продажами и оценками критиков. Игры, которые принесли наибольший доход, получили более высокие оценки критиков.
#     
# 5. В среднем самые доходные жанры - это Shooter, Sports, Platform и Role-playing. Также высок суммарный доход жанра Action. Наименьший доход принесли Puzzle и Strategy игры.
#     
# 6. Европа с Америкой предпочитают динамичные игры, высокий возрастной порог и стационарные консоли, тогда как Япония больше склонна к спокойным играм в жанре ролевых и портативным консолям.
# </div>

# <div style="border:solid blue 2px; padding: 20px"> 
# 
# <a id="hypothesis"> **Шаг 5. Проверка гипотез**</a>
#     
# - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые;
# 
# - Средние пользовательские рейтинги жанров Action (англ. «действие», экшен-игры) и Sports (англ. «спортивные соревнования») разные.
# </div>

# In[20]:


d_games_2013.platform.unique()


# In[21]:


#1. Средние пользовательские рейтинги платформ Xbox One и PC одинаковые
#Нулевая гипотеза: Средние пользовательские рейтинги платформ Xbox One и PC одинаковые;
#Альтернативная гипотеза: Средние пользовательские рейтинги платформ Xbox One и PC различаются

xone = d_games_2013[(d_games_2013['platform']=='XOne')]['user_score']
#print(x360)
#pc_platform = ['PC','PC2','PC3']
pc = d_games_2013[(d_games_2013['platform'] == 'PC')]['user_score']
#print(pc)
#print(x360.mean(),pc.mean())

#Пороговое значение alpha - 0.05
alpha = 0.05 #если p-value окажется меньше него - отвергнем гипотезу

results = st.ttest_ind(xone.dropna(), pc.dropna(), equal_var=False)
print('p-значение:', results.pvalue)


if (results.pvalue < alpha):
    print("Средние пользовательские рейтинги платформ Xbox One и PC различаются")
else:
    print("Средние пользовательские рейтинги платформ Xbox One и PC одинаковые")


# In[22]:


#2. Средние пользовательские рейтинги жанров Action (англ. «действие», экшен-игры) 
#и Sports (англ. «спортивные соревнования») разные.
#Нулевая гипотеза: Средние пользовательские рейтинги жанров Action и Sports одинаковые
#Альтернативная гипотеза: Средние пользовательские рейтинги жанров Action и Sports различаются
#print(d_games['genre'].unique())
action = d_games_2013[(d_games_2013['genre']=='Action')]['user_score']
sports= d_games_2013[(d_games_2013['genre']=='Sports')]['user_score']

#Пороговое значение alpha - 0.05
#Проверяем гипотезу
alpha = 0.05 #если p-value окажется меньше него - отвергнем гипотезу
results2 = st.ttest_ind(action.dropna(), sports.dropna(), equal_var=False)
print('p-значение:', results2.pvalue)


if (results2.pvalue < alpha):
    print("Средние пользовательские рейтинги жанров Action и Sports различаются")
else:
    print("Средние пользовательские рейтинги жанров Action и Sports одинаковые")


# <div class="alert alert-info">
# <b>Вывод: </b> 
#     
# - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
# 
# - Средние пользовательские рейтинги жанров Action и Sports различаются.
# </div>

# <div class="alert alert-info">
# <a id="output"> <b>Общий вывод: </b></a>
# 
# 
# 1. Значительный рост числа игр наблю дается с 1993 года до 2008 года. С 2008 года общее число игр начинает сокращаться. Вероятно это связано с высокой популярностью мобильных игр и приложений.
#     
# 2. С течением времени ранее популярные платформы становятся неактуальными
#     
# 3. Самые популярные игровые платформы 2016 года: PS4, XOne, 3DS, PC.
#     
# 4. У всех платформ наблюдается слабая корреляция между продажами и оценками критиков. Игры, которые принесли наибольший доход, получили более высокие оценки критиков.
#     
# 5. В среднем самые доходные жанры - это Shooter, Sports, Platform и Role-playing. Также высок суммарный доход жанра Action. Наименьший доход принесли Puzzle и Strategy игры.
#     
# 6. Европа с Америкой предпочитают динамичные игры, высокий возрастной порог и стационарные консоли, тогда как Япония больше склонна к спокойным играм в жанре ролевых и портативным консолям.
#     
# 7. Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
#     
# 8. Средние пользовательские рейтинги жанров Action и Sports различаются.
# </div>
