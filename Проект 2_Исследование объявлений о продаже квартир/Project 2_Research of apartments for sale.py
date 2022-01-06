#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Изучение-данных-из-файла" data-toc-modified-id="Изучение-данных-из-файла-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Изучение данных из файла</a></span><ul class="toc-item"><li><span><a href="#Вывод" data-toc-modified-id="Вывод-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Вывод</a></span></li></ul></li><li><span><a href="#Предобработка-данных" data-toc-modified-id="Предобработка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Предобработка данных</a></span></li><li><span><a href="#Расчёты-и-добавление-результатов-в-таблицу" data-toc-modified-id="Расчёты-и-добавление-результатов-в-таблицу-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Расчёты и добавление результатов в таблицу</a></span></li><li><span><a href="#Исследовательский-анализ-данных" data-toc-modified-id="Исследовательский-анализ-данных-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Исследовательский анализ данных</a></span></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Общий вывод</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Исследование объявлений о продаже квартир
# 
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктах за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости. Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. 
# 
# По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма. 

# <div class="alert alert-success">
# <b>Комментарий ревьюера:</b>
# 
# Здорово, когда есть вступление в сам проект, каждый сможет быстрее разобраться в концепции / деталях работы. 
# 
# </div>

# ## Изучение данных из файла

# In[120]:


#23.08.2021 - добавила метод display
import pandas as pd
import matplotlib.pyplot as plt #NEW
from IPython.display import display #NEW

data = pd.read_csv("/datasets/real_estate_data.csv", sep="\t")
display(data.head(10)) #NEW
data.info()
#print(data.describe())
display(data.describe()) #NEW


# ### Вывод

# 1) При выгрузке csv файла в Data Frame данные склеились в одну строку "\t". Я разделила их, указав на сепаратор "\t"
# 
# 2) В данных есть пропущенные значения в столбцах.

# ## Предобработка данных

# In[121]:


#1. определите и изучите пропущенные значения:
#---для некоторых пропущенных значений можно предположить логичную замену. Например, если человек не указал число балконов — скорее всего, их нет. Такие пропуски правильно заменить на 0. Для других типов данных нет подходящего значения на замену. В этом случае правильно оставить эти значения пустыми. Отсутствие значения — тоже важный сигнал, который не нужно прятать;
#---заполните пропуски, где это уместно. Опишите, почему вы решили заполнить пропуски именно в этих столбцах и как выбрали значения;
#---укажите причины, которые могли привести к пропускам в данных.
display(data.isnull().sum())

#Пропуски:
###########################
#1) ceiling_height - 9195
#Проверим возможные значения
print('Минимальная высота потолков -', data['ceiling_height'].min())
print('Максимальная высота потолков -', data['ceiling_height'].max())
#Проблемы при сборе данных
#Заполним пропущенные значения медианой
data['ceiling_height'] = data['ceiling_height'].fillna(data['ceiling_height'].median())
###########################
#2) floors_total (86) -  немного, можем исключить
#data.dropna(subset = ['floors_total'], inplace = True)
data = data.dropna(subset = ['floors_total']) #NEW
###########################
#3) living_area (1903) -  немного + #5) kitchen_area (2231) - немного
print('Число симметричных значений living_area и kitchen_area:',
      data.loc[(data['living_area'].isna())
               &(data['kitchen_area'].isna()),'total_area'].count())
print(data['kitchen_area'].median())
#Заполним пропущенные значения кухни нулями (возможно не была заполнена для студий)
#data['kitchen_area'].fillna(0, inplace=True)
data['kitchen_area'] = data['kitchen_area'].fillna(0) #NEW
#Если кухня не заполнялась не для студий (площать > 40), то заполним медианой
data.loc[(data['total_area'] > 40) 
         & (data['kitchen_area']==0), 'kitchen_area'] = data['kitchen_area'].median()
#Заполняем living_area, зная total_area и kitchen_area
data.loc[(data['living_area'].isna()), 'living_area'] = data['total_area']-data['kitchen_area']
print(data.loc[(data['living_area'].isna())])
###########################
#4) is_apartment
print('Значения, принимаемые is_apartment')
print(data['is_apartment'].unique())
# Предпологаю, что значения не заполнялись, когда False
#data['is_apartment'].fillna(False, inplace = True)
data['is_apartment'] = data['is_apartment'].fillna(False) #NEW
###########################
#6) balcony
#Предполагаю, что не заполнены там, где нет балконов
#data['balcony'].fillna(0, inplace = True)
#NEW - заполнение нулем добавлено позже в цикле (над пунктом 14)
###########################
#7) locality_name - немного
#data.dropna(subset = ['locality_name'], inplace = True)
data = data.dropna(subset = ['locality_name']) #NEW
###########################
#8) airports_nearest 
#Проблемы при сборе данных
print('Значения медианы airports_nearest', data['airports_nearest'].median())
#Заполним пропущенные значения медианой
#data['airports_nearest'].fillna(data['airports_nearest'].median(), inplace = True)
data['airports_nearest'] = data['airports_nearest'].fillna(data['airports_nearest'].median()) #NEW
###########################
#9) cityCenters_nearest - 5519
print('Значения медианы cityCenters_nearest', data['cityCenters_nearest'].median())
#Заполним пропущенные значения медианой
#data['cityCenters_nearest'].fillna(data['cityCenters_nearest'].median(), inplace = True)
data['cityCenters_nearest'] = data['cityCenters_nearest'].fillna(data['cityCenters_nearest'].median()) #NEW
###########################
#10) parks_around3000 и #11) parks_nearest
#Не заполнено, когда нет парков близко?
print('Число незаполненных одновременно parks_around3000 и parks_nearest'
      ,data.loc[(data['parks_around3000'].isna())
                &(data['parks_nearest'].isna())
                ,'parks_nearest'].count())
#data['parks_around3000'].fillna(0, inplace = True)
#data['parks_nearest'].fillna(0, inplace = True)
###########################
#12) ponds_around3000 и #13) ponds_nearest
#Не заполнено, когда нет озер близко?
print('Число незаполненных одновременно ponds_around3000 и ponds_nearest'
      ,data.loc[(data['ponds_around3000'].isna())
                &(data['ponds_nearest'].isna())
                ,'ponds_nearest'].count())
#data['ponds_around3000'].fillna(0, inplace = True)
#data['ponds_nearest'].fillna(0, inplace = True)

columns = ['balcony','parks_around3000', 'parks_nearest', 'ponds_around3000','ponds_nearest'] #NEW
for i in columns: #NEW
    data[i] = data[i].fillna(0) #NEW
###########################
#14) days_exposition - немного
#Пропуски days_exposition вероятно появились при загрузке данных
#data['days_exposition'].fillna(data['days_exposition'].median(), inplace = True)
data['days_exposition'] = data['days_exposition'].fillna(data['days_exposition'].median()) #NEW
#2. приведите данные к нужным типам:
#---поясните, в каких столбцах нужно изменить тип данных и почему.
#Заменим тип данных на целочисленный
#data['...']  = data['...'].astype('int')
data['days_exposition'] = data['days_exposition'].astype('int64')

columns = ['airports_nearest'
           , 'parks_nearest','parks_around3000'
           ,'ponds_nearest', 'ponds_around3000'
           ,'floor','floors_total'] 

for i in columns: 
    data[i] = data[i].astype('int') 
data.info() 


# ```python
# 
# columns = [col1, col2, col3]
# 
# for i in columns:
#     df[i] = df[i].fillna(0)
# 
# ```

# ## Расчёты и добавление результатов в таблицу

# In[122]:


#1. цену квадратного метра;
data['price_sqm']=data['last_price']/data['total_area']
print('Средняя цена кв. метра - ',data['price_sqm'].median(), 'руб.')
#2. день недели, месяц и год публикации объявления;
import datetime
data['first_day_exposition'] = (
    pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')
)
data['date_hour'] = data['first_day_exposition'].dt.round('1H')
#data['week_day'] = data['date_hour'].get_weekday()
data['week_number'] = pd.DatetimeIndex(data['date_hour']).weekday
data['month_number'] = pd.DatetimeIndex(data['date_hour']).month
data['year_number'] = pd.DatetimeIndex(data['date_hour']).year

def dayNameFromWeekday(weekday):
    if weekday == 0:
        return "1 Monday"
    if weekday == 1:
        return "2 Tuesday"
    if weekday == 2:
        return "3 Wednesday"
    if weekday == 3:
        return "4 Thursday"
    if weekday == 4:
        return "5 Friday"
    if weekday == 5:
        return "6 Saturday"
    if weekday == 6:
        return "7 Sunday"
    
def MonthNameFromNumber(month_number):
    if month_number == 1:
        return "1 January"
    if month_number == 2:
        return "2 February"
    if month_number == 3:
        return "3 March"
    if month_number == 4:
        return "4 April"
    if month_number == 5:
        return "5 May"
    if month_number == 6:
        return "6 June"
    if month_number == 7:
        return "7 July"
    if month_number == 8:
        return "8 August"
    if month_number == 9:
        return "9 September"
    if month_number == 10:
        return "_10 October"
    if month_number == 11:
        return "_11 November"
    if month_number == 12:
        return "_12 December"
    
data['week_name'] = data['week_number'].apply(dayNameFromWeekday)
data['month_name'] = data['month_number'].apply(MonthNameFromNumber)
print(data[['date_hour','week_name','month_name','year_number']])
####################################  
#3. этаж квартиры; варианты — первый, последний, другой;
def FloorName(floor, lastfloor):   
    if floor == 1:
        return "первый"
    elif floor == lastfloor:
        return "последний"
    else:
        return "другой"
#-------
def FloorName_N(floor, lastfloor):   
    if floor == 1:
        return 1
    elif floor == lastfloor:
        return 2
    else:
        return 3
data['floor_name']  = data.apply(lambda x: FloorName(x.floor, x.floors_total), axis=1)
print(data[['floor_name','floor', 'floors_total']].head(3))
data['floor_name_N'] = data.apply(lambda x: FloorName_N(x.floor, x.floors_total), axis=1)
###################################
#4. соотношение жилой и общей площади, а также отношение площади кухни к общей.
data['living2total'] = data['living_area']/data['total_area']
data['kitchen2total'] = data['kitchen_area']/data['total_area']
print("Среднее соотношение жилой и общей площади - {0:.2f}%".format(data['living2total'].median()*100))
print("Среднее соотношение кухни и общей площади - {0:.2f}%".format(data['kitchen2total'].median()*100))


# ## Исследовательский анализ данных

# In[123]:


#import matplotlib.pyplot as plt
#1. Изучите следующие параметры: площадь, цена, число комнат, высота потолков.
#2. Постройте гистограммы для каждого параметра.
print('Минимальная площадь -', data['total_area'].min())
print('Максимальная площадь -', data['total_area'].max())
data['total_area'].hist(bins=30,range=(10,150), label= 'Площадь')
plt.title('Гистограмма площади квартир') #NEW
plt.show() #NEW
#Больше всего продается квартир площадью 45-50 м.кв


# In[124]:


print('Минимальная цена -', data['last_price'].min())
print('Максимальная цена -', data['last_price'].max())
data['last_price'].hist(bins=20,range=(0,10000000), label = 'Цена')
plt.title('Гистограмма цен квартир') #NEW
plt.show() #NEW
#Больше всего квартир стоимостью 3-5 млн руб.


# In[125]:


print('Минимальное число комнат -', data['rooms'].min())
print('Максимальное число комнат -', data['rooms'].max())
data['rooms'].hist(bins=11,range=(1,10), label = 'Число комнат')
plt.title('Гистограмма числа комнат') #NEW
plt.show() #NEW
#Преобладают 1-е и 2-е квартиры
#проверим, что квартиры с числом комнат более 8 имеют площадь больше 200 кв м
display(data[['rooms','total_area']].loc[(data['rooms']> 8)]) #NEW - display вместо print


# In[126]:


print('Минимальная высота потолков -', data['ceiling_height'].min())
print('Максимальная высота потолков -', data['ceiling_height'].max())
data['ceiling_height'].hist(bins=20,range=(2,4), label = 'Высота потолка')
plt.title('Гистограмма высоты потолка') #NEW
plt.show() #NEW
#Больше всего продаж квартир с высотой потока 2.6 м


# In[127]:


#3. Изучите время продажи квартиры. 
#4. Постройте гистограмму. 
plt.figure(figsize=(10,7))
data['days_exposition'].hist(bins=30,range=(0,150))
plt.title('Гистограмма времени продажи квартир')
plt.show()
data['days_exposition'].describe()
print('Арифм. среднее продажи квартиры',data['days_exposition'].mean(),'дней')
print('Медиана продажи квартиры',data['days_exposition'].median(),'дней')
#5. Посчитайте среднее и медиану. 
#---Опишите, сколько обычно занимает продажа. 
#Продажа обычно в среднем занимает от 90 до 100 дней
#---Когда можно считать, что продажи прошли очень быстро, а когда необычно долго?
#Можем считать, что продажа прошла быстро, если квартира была продана в течение 45 дней
#Можем считать, что продажа проходит долго, если квартира продается дольше 198 дней


# In[108]:


#6. Уберите редкие и выбивающиеся значения.
data = data[data['days_exposition'] < 500]
#Убираем квадтиры, продающиеся дольше 500 дней
data = data[data['last_price'] < 100000000]
#Убираем квартиры стоимостью более 100 млн руб
data = data[data['total_area'] < 1000]
#Убираем квартиры площадью более 1000 кв м
#---Опишите, какие особенности обнаружили.
#---Какие факторы больше всего влияют на стоимость квартиры? 
#7. Изучите, зависит ли цена от площади, числа комнат, удалённости от центра. 
print('Стоимость коррелирует с площадью с коэффициентом'
      ,data['last_price'].corr(data['total_area']))
print('Стоимость коррелирует с числом комнат с коэффициентом'
      , data['last_price'].corr(data['rooms']))
print('Стоимость не коррелирует с этажом, коэффициент  корреляции - '
      ,data['last_price'].corr(data['floor']))
print('Стоимость не коррелирует с удаленностью от центра, коэффициент  корреляции - '
      ,data['last_price'].corr(data['cityCenters_nearest']))
#8. Изучите зависимость цены от того, на каком этаже расположена квартира: первом, последнем или другом.
print('Стоимость не коррелирует с этажом, коэффициент  корреляции - '
      ,data['last_price'].corr(data['floor_name_N']))

data.corr()['last_price']


# In[109]:


#Корреляцию можно посчитать сразу для всех столбцов:
data.corr()['last_price']


# In[110]:


data.groupby('floor_name')['last_price'].median().plot(x='floor_name',y='last_price',kind='bar')
plt.title('Зависимость цены квартиры от этажа')       
plt.show()
#На первом этаже квартиры в среднем дешевле


# In[111]:


#Также изучите зависимость от даты размещения: дня недели, месяца, года
#data.groupby('week_name')['last_price'].count().plot(x='week_name',y='last_price',kind='bar')
#Применение for к гистограммам зависимости цены от дня недели, месяца, года

columns = ['week_name'
           , 'month_name'
           , 'year_number'
          ] 

for i in columns: 
    data.groupby(i)['last_price'].median().plot(x=i,y='last_price',kind='bar') 
    if i == 'week_name':
        plt.title('Зависимость цены квартиры от дня недели публикации')
    if i == 'month_name':
        plt.title('Зависимость цены квартиры  от месяца публикации')
    if i == 'year_number':
        plt.title('Зависимость цены квартиры  от года публикации')       
    plt.show() 
#В 2014 году было опубликовано больше дорогих квартир


# In[112]:


#Также изучите зависимость от даты размещения: дня недели, месяца, года
#data.groupby('week_name')['last_price'].count().plot(x='week_name',y='last_price',kind='bar')
#Применение for к гистограммам зависимости числа объявлений от дня недели, месяца, года
columns = ['week_name'
           , 'month_name'
           , 'year_number'
          ] 

for i in columns: 
    data.groupby(i)['last_price'].count().plot(x=i,y='last_price',kind='bar')
    if i == 'week_name':
        plt.title('Зависимость числа объявлений от дня недели публикации')
    if i == 'month_name':
        plt.title('Зависимость  числа объявлений от месяца публикации')
    if i == 'year_number':
        plt.title('Зависимость числа объявлений от года публикации')       
    plt.show()
#Чаще всего объявления публиковались во вторник, четверг и пятницу. В выходные меньше всего.
#Меньше всего объявлений публикуется в декабре, январе, мае.


# In[113]:


#9. Выберите 10 населённых пунктов с наибольшим числом объявлений.
#10. Посчитайте среднюю цену квадратного метра в этих населённых пунктах. 
#Выделите среди них населённые пункты с самой высокой и низкой стоимостью жилья. 
#Эти данные можно найти по имени в столбце 'locality_name'.
data_agg_loc = data.pivot_table(index='locality_name',values='price_sqm',aggfunc=('count','median'))
data_agg_loc.sort_values(by='count',ascending=False).head(10)

#В Санкт-Петербурге больше всего объявлений и самая большая стоимость 1 кв м
#В Выборге меньше всего объявлений и самая низкая стоимость 1 кв м среди городо Топ-10 по числу объявлений


# In[114]:


#11. Изучите предложения квартир: 
#для каждой квартиры есть информация о расстоянии до центра. 
print(data['cityCenters_nearest'].head(4))
#Выделите квартиры в Санкт-Петербурге ('locality_name'). 
#Ваша задача — выяснить, какая область входит в центр. 
#Создайте столбец с расстоянием до центра в километрах: округлите до целых значений.
data['center_km'] = round(data['cityCenters_nearest'] / 1000, 0)
spb_city = data.loc[(data['locality_name'] == 'Санкт-Петербург')]
spb_city = spb_city.dropna(subset = ['cityCenters_nearest'], inplace = False) 
spb_city['center_km'] = round(spb_city['cityCenters_nearest'] / 1000, 0)
#После этого посчитайте среднюю цену для каждого километра.
spb_segment = spb_city.groupby(by = ['center_km'], as_index=False).mean()
#spb_center_pivot = spb_city.pivot_table(index='center_km', values=['price_sqm', 'last_price'], aggfunc='median')

#Постройте график: он должен показывать, как цена зависит от удалённости от центра. 
spb_segment.plot(x='center_km', y = 'last_price', kind = 'scatter',grid=True)
#Определите границу, где график сильно меняется — это и будет центральная зона.
# --  8  км - граница центральной зоны


# In[115]:


#12. Выделите сегмент квартир в центре. 
spb_center = spb_city[spb_city['center_km'] <= 8]
#13. Проанализируйте эту территорию и изучите следующие параметры: 
#площадь, цена, число комнат, высота потолков. - 'total_area','last_price','rooms','ceiling_height'
display(spb_center[['total_area','last_price','rooms','ceiling_height']])
#Также выделите факторы, которые влияют на стоимость квартиры
#(число комнат, этаж, удалённость от центра, дата размещения объявления). 
#'rooms'
spb_rooms = spb_center.groupby(by = ['rooms'], as_index=False).mean()
spb_rooms.plot(x='rooms', y = 'last_price', kind = 'scatter',grid=True)
#Цена зависит от числа комнат (скорее всего потому, что число комнат коррелирует с площадью)


# In[116]:


#floor_name_N'
spb_floor = spb_center.groupby(by = ['floor_name'], as_index=False).mean()
spb_floor.plot(x='floor_name', y = 'last_price', kind = 'bar',grid=True)
#На первом этаже в среднем квартиры дешевле


# In[119]:


#высота потолков 'ceiling_height'
spb_center = spb_center.loc[(spb_center['ceiling_height']<20)] #NEW, исключила 5 строк, где высота потолка больше 20 метров
spb_center['ceiling_new'] = spb_center['ceiling_height'].astype(int)
print(spb_center['ceiling_new'].unique())
spb_ceiling = spb_center.groupby(by = ['ceiling_new'], as_index=False).mean()
spb_ceiling.plot(x='ceiling_new', y = 'last_price', kind = 'bar',grid=True)
#plt.title('Зависимость цены квартиры в С-П от высоты потолка')       
#    plt.show() #NEW
#Чем выше потолок, тем дороже квартира


# ## Общий вывод

# Мы определили, что 
# 
# 1 стоимость объектов недвижимости зависит от следующих факторов:
#     общая площадь квартиры,
#     расстояние до центра города,
#     число комнат (вероятно потому, что чило комнат зависит от общей площади квартиры),
#     этаж квартиры.
# 
# 2 центр С-П находится в радиусе 8 км
# 
# 3 стоимость квартир в центре С-П также зависит от факторов:
#     площадь, число комнат,
#     высота потолков.

# <div class="alert alert-success">
# <b>Комментарий:</b>
# 
# Для стилизации проекта можно использовать:
# https://sqlbak.com/blog/jupyter-notebook-markdown-cheatsheet
#     
# </div>

# ## Чек-лист готовности проекта
# 
# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  открыт файл
# - [x]  файлы изучены (выведены первые строки, метод info())
# - [x]  определены пропущенные значения
# - [x]  заполнены пропущенные значения
# - [x]  есть пояснение, какие пропущенные значения обнаружены
# - [x]  изменены типы данных
# - [ ]  есть пояснение, в каких столбцах изменены типы и почему
# - [x]  посчитано и добавлено в таблицу: цена квадратного метра
# - [x]  посчитано и добавлено в таблицу: день недели, месяц и год публикации объявления
# - [x]  посчитано и добавлено в таблицу: этаж квартиры; варианты — первый, последний, другой
# - [x]  посчитано и добавлено в таблицу: соотношение жилой и общей площади, а также отношение площади кухни к общей
# - [x]  изучены следующие параметры: площадь, цена, число комнат, высота потолков
# - [x]  построены гистограммы для каждого параметра
# - [x]  выполнено задание: "Изучите время продажи квартиры. Постройте гистограмму. Посчитайте среднее и медиану. Опишите, сколько обычно занимает продажа. Когда можно считать, что продажи прошли очень быстро, а когда необычно долго?"
# - [x]  выполнено задание: "Уберите редкие и выбивающиеся значения. Опишите, какие особенности обнаружили."
# - [x]  выполнено задание: "Какие факторы больше всего влияют на стоимость квартиры? Изучите, зависит ли цена от квадратного метра, числа комнат, этажа (первого или последнего), удалённости от центра. Также изучите зависимость от даты размещения: дня недели, месяца и года. "Выберите 10 населённых пунктов с наибольшим числом объявлений. Посчитайте среднюю цену квадратного метра в этих населённых пунктах. Выделите населённые пункты с самой высокой и низкой стоимостью жилья. Эти данные можно найти по имени в столбце '*locality_name'*. "
# - [x]  выполнено задание: "Изучите предложения квартир: для каждой квартиры есть информация о расстоянии до центра. Выделите квартиры в Санкт-Петербурге (*'locality_name'*). Ваша задача — выяснить, какая область входит в центр. Создайте столбец с расстоянием до центра в километрах: округлите до целых значений. После этого посчитайте среднюю цену для каждого километра. Постройте график: он должен показывать, как цена зависит от удалённости от центра. Определите границу, где график сильно меняется — это и будет центральная зона. "
# - [x]  выполнено задание: "Выделите сегмент квартир в центре. Проанализируйте эту территорию и изучите следующие параметры: площадь, цена, число комнат, высота потолков. Также выделите факторы, которые влияют на стоимость квартиры (число комнат, этаж, удалённость от центра, дата размещения объявления). Сделайте выводы. Отличаются ли они от общих выводов по всему городу?"
# - [x]  в каждом этапе есть выводы
# - [x]  есть общий вывод
