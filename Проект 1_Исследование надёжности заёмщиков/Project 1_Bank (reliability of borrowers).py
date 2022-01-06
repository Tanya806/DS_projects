#!/usr/bin/env python
# coding: utf-8

# # Исследование надёжности заёмщиков
# 
# Заказчик — кредитный отдел банка. Нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. Входные данные от банка — статистика о платёжеспособности клиентов.
# 
# Результаты исследования будут учтены при построении модели **кредитного скоринга** — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.

# ## Шаг 1. Откройте файл с данными и изучите общую информацию

# In[3]:


import pandas as pd
data = pd.read_csv("/datasets/data.csv")


# In[4]:


print(data.head(10))
data.info()
print(data.describe()) #-NEW-
#Рассмотрим уникальные значения данных для столбцов.
#---new_begin---
print(data.describe())
#---new_end-----
for row in data: 
    print(data[row].value_counts())


# **Вывод**
# Каждая строка таблицы содержит информацию о заемщиках с определенным семейным положеним и количеством детей, которые пользователи взяли кредит под определенные цели. В таблице содержится информация о наличии просрочки.
# Проблемы, которые нужно решать: 
#     #1 некачественные данные (отрицательные days_employed)
#     #2 проблема с регистрами: Education в разных регистрах,     
#     #3 разные значения Purpose предполагают одну и ту же цель кредита (необходима категоризация),    
#     #4 количество значений в столбцах различается. Это говорит о том, что в данных есть days_employed и total_income есть пропущенные значения  
#     #5 столбец "children" содержит ошибочные значения 20 и -1,    
#     #6 столбец "gender" содержит одно ошибочное значение XNA,   
#     #7 у 101 заемщика указан возраст 0.
# 
# Для проверки рабочих гипотез особенно ценны столбцы debt и purpose.

# 

# ## Шаг 2. Предобработка данных

# ### Обработка пропусков

# In[5]:


#Таблица не требует переименования столбцов: названия интуитивно понятны, 
#в названиях столбцов нет пробелов, которые могут затруднять доступ к данным.
#Проверим данные на наличие пропусков
print(data.isnull().sum())
#---new_begin---
print('Число симметричных значений:',data.loc[(data['total_income'].isna())&(data['days_employed'].isna()),'children'].count())
#---new_end-----


# **Вывод**

# days_employed и total_income содержат 2174 пропуска.
# Пустые значения свидетельствуют, что для некоторых заемщиков доступна не вся информация. 
# Причины могут быть разные: скажем, не применимо к определенным категориям заемщиков или поле не было обязательным при заполнении анкеты. Возможны также проблемы с записью данных.
# 
# #---new_begin---
# Пропуски в двух столбцах симметричны.
# #---new_end-----

# ### Замена типа данных

# In[6]:


#4 количество значений в столбцах total_income и days_employed различается. Это говорит о том,
#что в данных есть days_employed и total_income есть пропущенные значения
    #total_income
data['total_income']  = pd.to_numeric(data['total_income'],errors='coerce')
#---new_begin---
data['total_income'] = data['total_income'].fillna(0)
medians_income = data.groupby('income_type')['total_income'].median()
print('Средняя зарплата:', medians_income)
data.loc[(data['total_income']== 0) & (data['income_type'] == 'безработный'), 'total_income'] = medians_income[0]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'в декрете'), 'total_income'] = medians_income[1]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'госслужащий'), 'total_income'] = medians_income[2]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'компаньон'), 'total_income'] = medians_income[3]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'пенсионер'), 'total_income'] = medians_income[4]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'предприниматель'), 'total_income'] = medians_income[5]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'сотрудник'), 'total_income'] = medians_income[6]
data.loc[(data['total_income']== 0) & (data['income_type'] == 'студент'), 'total_income'] = medians_income[7]
#---new_end-----
#data['total_income']  = data['total_income'].fillna(data.loc[(data['total_income'] > 0),'total_income'].median())
data['total_income']  = data['total_income'].astype('int')
print('Число пропущеннных значений total_income:',len(data[data['total_income'].isna()])) 

    #days_employed
data['days_employed'] = pd.to_numeric(data['days_employed'],errors='coerce')
#---new_begin---
#берем медиану только по значениям от 0 до 15000 раб. дней
data['days_employed'] = data['days_employed'].fillna(data.loc[(data['days_employed'] > 0),'days_employed'].median())
data['days_employed'] = data['days_employed'].astype('int')
#---new_end-----
print('Число пропущеннных значений days_employed:',len(data[data['days_employed'].isna()])) 

#кроме того устраним выявленные проблемы*
    #1 некачественные данные (отрицательные days_employed)
data['days_employed'] = abs(data['days_employed'])
print('Минимальный стаж в днях -', data['days_employed'].min())
print('Максимальный стаж в днях -', data['days_employed'].max())
    #5 столбец "children" содержит ошибочные значения 20 и -1,
data['children'] = data['children'].replace(20, 2) #предполагаем, что ошибочно записали лишний 0
data['children'] = data['children'].replace(-1, 1) #предполагаем, что ошибочно  указали минус
print('Число детей заемщиков - ',data['children'].unique())   
    #6 столбец "gender" содержит одно ошибочное значение XNA,
    # Строка с пропущенным значением одна, удалим из статистики. 
data = data[data['gender'] != 'XNA']
print('Пол заемщиков', data['gender'].unique())
    #7 у 101 заемщика указан возраст 0.
print('Средний возраст заемщика mean - ', data.loc[(data['dob_years'] > 0), 'dob_years'].mean())
print('Средний возраст заемщика median - ', data.loc[(data['dob_years'] > 0), 'dob_years'].median())
data.loc[(data['dob_years'] == 0),'dob_years'] = data.loc[(data['dob_years'] > 0), 'dob_years'].median()
print('Возраст заемщиков:')
print(' - минимальный -', data['dob_years'].min())
print(' - максимальный -', data['dob_years'].max())


# **Вывод**

# Исключены дубликаты, скорректированы следующие ошибки в данных:
#     #4 количество значений в столбцах различается. Пропущенные значение days_employed и total_income заменены на средние (mean)
#     #1 Отрицательные days_employed заменены на положительные (по модулю)      
#     #5 В столбце "children" скорректированы ошибочные значения 20 и -1,    
#     #6 Удалены строка с ошибкой в столбце "gender" (содержало одно ошибочное значение XNA)   
#     #7 у 101 заемщика с возрастом 0 скорректированы значение (заменены на средние)
#     #---new_begin---
#      - Нулевая зарплата заменена на среднее для каждого income_type
#      - В столбце со стажем содержались некорректные значения(слишком большие, отрицательные). Об этом нужно сообщить разработчикам.
#     #---new_end-----

# ### Обработка дубликатов

# In[7]:


#2 Устранение проблемы с регистрами: Education в разных регистрах
data['education'] = data['education'].str.lower() 
print('Уровни образования заемщиков - ',data['education'].unique())

#Удаление полных дубликатов
#print(data.duplicated().sum())
data = data.drop_duplicates().reset_index(drop=True)
print('Число дубликатов -',data.duplicated().sum())

#print(data.head(10))


# **Вывод**

# Дубликаты могли появиться вследствие сбоя в записи данных.
# Устранена проблем с регистрами Education. Все полные дубликаты данных были удалены с новой индексацией.

# ### Лемматизация

# In[10]:


from pymystem3 import Mystem
from nltk.stem import SnowballStemmer 
from collections import Counter
m = Mystem()

#3 разные значения Purpose предполагают одну и ту же цель кредита (необходима лемматизация и категоризация)   
def lemma_purpose(purpose):
    lemma = ' '.join(m.lemmatize(purpose))
    return lemma

data['purpose_lemma'] = data['purpose'].apply(lemma_purpose)
print(data.head(3))
#print(data['purpose_lemma'].unique()) 


# **Вывод**

# Проведена лемматизация поля Purpose, которая требуется для категоризации.

# ### Категоризация данных

# In[7]:


data['category'] = 'неизвестная категория'
def purpose_cat(p_lemma):
    if "жилье" in p_lemma:
        return "жилье"
    if 'автомобиль' in p_lemma:
        return "автомобиль"
    if "свадьба" in p_lemma:
        return "свадьба"
    if "образование" in p_lemma:
        return "образование"
    if "недвижимость" in p_lemma:
        return "недвижимость"
    if "строительство" in p_lemma:
        return "строительство"

data['category'] = data['purpose_lemma'].apply(purpose_cat)
print(data['category'].head(10))
print(data.head(10))
#print(data.loc[data['category'] == 'неизвестная категория'])


# **Вывод**

# Данные разбиты по категориям по целям по результатам лемматизации.
# После группировки цели кредита разделены следующие группы:
# - жилье, 
# - автомобиль, 
# - свадьба, 
# - образование, 
# - недвижимость, 
# - строительство.

# ## Шаг 3. Ответьте на вопросы

# - Есть ли зависимость между наличием детей и возвратом кредита в срок?

# In[21]:


#index — столбец или столбцы, по которым группируют данные (название товара)
#columns — столбец, по значениям которого происходит группировка (даты)
#values — значения, по которым мы хотим увидеть сводную таблицу (количество проданного товара)
#aggfunc — функция, применяемая к значениям
data_pivot1 = data.pivot_table(index=['debt'], columns='children', values = 'total_income', aggfunc='count')
print(data_pivot1)
#ошибки в данных - количество детей = -1 и 20
#---new_begin---
ch0 = data_pivot1[0][1] / (data_pivot1[0][0]+data_pivot1[0][1])
ch1 = data_pivot1[1][1] / (data_pivot1[1][0]+data_pivot1[1][1])
ch2 = data_pivot1[2][1] / (data_pivot1[2][0]+data_pivot1[2][1])
ch3 = data_pivot1[3][1] / (data_pivot1[3][0]+data_pivot1[3][1])
ch4 = data_pivot1[4][1] / (data_pivot1[4][0]+data_pivot1[4][1])
#---new_end-----
print("Нет детей - {0:.2f}%".format(ch0*100))
print("Один ребенок - {0:.2f}%".format(ch1*100))
print("Два ребенка - {0:.2f}%".format(ch2*100))
print("Три ребенка - {0:.2f}%".format(ch3*100))
print("Четыре ребенка - {0:.2f}%".format(ch4*100))

#нет ребенка - 8% просрочки, есть ребенок - более 10% просрочки. Есть зависимость.


# **Вывод**
# Есть слабая зависимость просрочки от наличия детей: при отсутствии ребенка - 
# #---new_begin---
# 7,5% просрочки, у взрослых с деньми более 9% просрочки.
# #---new_end-----

# 

# - Есть ли зависимость между семейным положением и возвратом кредита в срок?

# In[ ]:


#index — столбец или столбцы, по которым группируют данные (название товара)
#columns — столбец, по значениям которого происходит группировка (даты)
#values — значения, по которым мы хотим увидеть сводную таблицу (количество проданного товара)
#aggfunc — функция, применяемая к значениям
data_pivot2 = data.pivot_table(index=['debt'], columns='family_status', values = 'family_status_id', aggfunc='count')
print(data_pivot2)

#---new_begin---
fd0 = data_pivot2['Не женат / не замужем'][1] / (data_pivot2['Не женат / не замужем'][0]+data_pivot2['Не женат / не замужем'][1])
fd1 = data_pivot2['в разводе'][1] / (data_pivot2['в разводе'][0] + data_pivot2['в разводе'][1])
fd2 = data_pivot2['вдовец / вдова'][1] / (data_pivot2['вдовец / вдова'][0]+data_pivot2['вдовец / вдова'][1])
fd3 = data_pivot2['гражданский брак'][1] / (data_pivot2['гражданский брак'][0]+data_pivot2['гражданский брак'][1])
fd4 = data_pivot2['женат / замужем'][1] / (data_pivot2['женат / замужем'][0]+data_pivot2['женат / замужем'][1])
#---new_end-----
#print(fd0, fd1,fd2, fd3, fd4)

print("Не женат / не замужем - {0:.2f}%".format(fd0*100))
print("В разводе - {0:.2f}%".format(fd1*100))
print("Вдовец / вдова - {0:.2f}%".format(fd2*100))
print("Гражданский брак - {0:.2f}%".format(fd3*100))
print("Женат / замужем - {0:.2f}%".format(fd4*100))

#Не женатые/ Не замужние или люди в гражданском браке реже выплачивают кредит в срок (7-8% vs 10%)


# **Вывод**

# Не женатые/ Не замужние и люди в гражданском браке реже выплачивают кредит в срок 
# #---new_begin---
# (>9,7%)
# #---new_end-----

# - Есть ли зависимость между уровнем дохода и возвратом кредита в срок?

# In[ ]:


#data['total_income'].quantile(.2, axis = 0)
#Преобразуем данные столбца total_income в квантили с помощью quantile()
total_income_quantile = data.total_income.quantile([.2, .4, .6, .8])
print('Данные столбца total_income в квантили по квантилям:')
print(total_income_quantile)

#Используем округленные значения квантилей для группировки
def income_group(row):
    income = row['total_income']

    if income <= 100000:
        return 'Доход до 100'
    
    if income <= 130000 and income > 100000:
        return 'Доход 100-130'
    
    if income <= 160000 and income > 130000:
        return 'Доход 130-160'
    
    if income <= 215000 and income > 160000:
        return 'Доход 160-215'
    
    if income > 215000:
        return 'Доход больше 215'

data['income_gr'] = data.apply(income_group, axis=1)

print (data['income_gr'])

data_pivot3 = data.pivot_table(index = 'debt', columns = 'income_gr', values = 'family_status_id', aggfunc = 'count')
print(data_pivot3)
#---new_begin---
income0 = data_pivot3['Доход до 100'][1] / (data_pivot3['Доход до 100'][0]+data_pivot3['Доход до 100'][1])
income1 = data_pivot3['Доход 100-130'][1] / (data_pivot3['Доход 100-130'][0]+data_pivot3['Доход 100-130'][1])
income2 = data_pivot3['Доход 130-160'][1] / (data_pivot3['Доход 130-160'][0]+data_pivot3['Доход 130-160'][1])
income3 = data_pivot3['Доход 160-215'][1] / (data_pivot3['Доход 160-215'][0]+data_pivot3['Доход 160-215'][1])
income4 = data_pivot3['Доход больше 215'][1] / (data_pivot3['Доход больше 215'][0]+data_pivot3['Доход больше 215'][1])
#---new_end-----
#print(income0, income1,fd2, fd3, fd4)

print("Доход менее 100 тыс. - {0:.2f}%".format(income0*100))
print("Доход от 100 до 130 тыс. - {0:.2f}%".format(income1*100))
print("Доход от 130 до 160 тыс. - {0:.2f}%".format(income2*100))
print("Доход от 160 до 215 тыс. - {0:.2f}%".format(income3*100))
print("Доход более 215 тыс. - {0:.2f}%".format(income4*100))


# **Вывод**

# Люди с маленьким доходом (менее 100 тыс.)
# #---new_begin---
# и с очень высоким доходом (более 215 тыс.) наиболее вероятно погасят свой кредит в срок.
# #---new_end-----

# - Как разные цели кредита влияют на его возврат в срок?

# In[ ]:


print(data['category'])

data_pivot4 = data.pivot_table(index = 'debt', columns = 'category', values = 'family_status_id', aggfunc = 'count')
print(data_pivot4)

#автомобиль  жилье  недвижимость  образование  свадьба
#---new_begin---
cd0 = data_pivot4['автомобиль'][1] / (data_pivot4['автомобиль'][0]+data_pivot4['автомобиль'][1])
cd1 = data_pivot4['жилье'][1] / (data_pivot4['жилье'][0]+data_pivot4['жилье'][1])
cd2 = data_pivot4['недвижимость'][1] / (data_pivot4['недвижимость'][0]+data_pivot4['недвижимость'][1])
cd3 = data_pivot4['образование'][1] / (data_pivot4['образование'][0]+data_pivot4['образование'][1])
cd4 = data_pivot4['свадьба'][1] / (data_pivot4['свадьба'][0]+data_pivot4['свадьба'][1])
#---new_end-----
#print(cd0, cd1, cd2, cd3, cd4)

print("Автомобиль - {0:.2f}%".format(cd0*100))
print("Жилье - {0:.2f}%".format(cd1*100))
print("Недвижимость - {0:.2f}%".format(cd2*100))
print("Образование - {0:.2f}%".format(cd3*100))
print("Свадьба - {0:.2f}%".format(cd4*100))

#Больше всего просрочек у кредитов на образование и на автомобиль, меньше всего - на жилье


# **Вывод**
# Больше всего просрочек у кредитов на образование и на автомобиль, меньше всего у кредитов на жилье.

# 

# ## Шаг 4. Общий вывод

# 1 Файл подготовлен для анализа:
# - были скорректированы типы данных,
# - удалены дубли,
# - цели кредита разбиты на категории. 
# 
# 2 Проведен анализ зависимости числа просрочек от различных характеристик заемщика. По итогам анализа были сделаны следующие выводы:
# - число просрочек чувствительно с семейному статусу. Не женатые/ Не замужние, люди в гражданском браке реже выплачивают кредит в срок,
# - люди с меньшим доходом наиболее вероятно погасят свой кредит в срок, 
# - наибольшее число просрочек наблюдается у кредитов на образование и на автомобиль, меньше всего у кредитов на жилье.

# ## Чек-лист готовности проекта
# 
# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  открыт файл;
# - [x]  файл изучен;
# - [x]  определены пропущенные значения;
# - [x]  заполнены пропущенные значения;
# - [x]  есть пояснение, какие пропущенные значения обнаружены;
# - [x]  описаны возможные причины появления пропусков в данных;
# - [x]  объяснено, по какому принципу заполнены пропуски;
# - [x]  заменен вещественный тип данных на целочисленный;
# - [ ]  есть пояснение, какой метод используется для изменения типа данных и почему;
# - [x]  удалены дубликаты;
# - [ ]  есть пояснение, какой метод используется для поиска и удаления дубликатов;
# - [x]  описаны возможные причины появления дубликатов в данных;
# - [x]  выделены леммы в значениях столбца с целями получения кредита;
# - [x]  описан процесс лемматизации;
# - [x]  данные категоризированы;
# - [x]  есть объяснение принципа категоризации данных;
# - [x]  есть ответ на вопрос: "Есть ли зависимость между наличием детей и возвратом кредита в срок?";
# - [x]  есть ответ на вопрос: "Есть ли зависимость между семейным положением и возвратом кредита в срок?";
# - [x]  есть ответ на вопрос: "Есть ли зависимость между уровнем дохода и возвратом кредита в срок?";
# - [x]  есть ответ на вопрос: "Как разные цели кредита влияют на его возврат в срок?";
# - [x]  в каждом этапе есть выводы;
# - [x]  есть общий вывод.
