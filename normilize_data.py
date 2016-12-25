# coding=utf-8
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_title_from_family(name): # нормализуем параметр фамилии, а именно выделяем из каждого пассажира его титул,
    # леди и мисс имеют больший приоритет, после этого замужние женщины, потом идут молодые люди и юноши, члены команды
    # и мужчины имеют меньший приоритет
    if pd.isnull(name):
        return 0
    if re.search(r'(Miss|Ms|Lady)\.', name):
        return 0.66
    elif re.search(r'(Mrs|Dona)\.', name):
        return 1
    elif re.search(r'(Master|Sir|)\.', name):
        return 0.33
    elif re.search(r'(Rev|Major|Mr|Don|Capt)\.', name):
        return 0
    elif re.search(r'(Mme|Mlle|Jonkheer|Countess|Dr|Col)\.', name):
        return 0.5


def length(number): # функция для подсчета длины числа
    summ = 0
    while number > 0:
        summ += 1
        number /= 10
    return summ


def digit_sum(number): # функция для подсчета суммы цифр числа
    summ = 0
    while number > 0:
        summ += number % 10
        number /= 10
    return summ


def normilize_ticket_number(ticket): # нормализуем данные о билете, для этого выделяем последние три цифры из номера
    # билета
    result = re.search(r'\d{3,}', ticket)
    if result:
        number = int(result.group(0))
        return number % 1000
    return 0


def normilize_ticket_identifier(ticket): # функция подсчета серии билета, если таковый имеется,  А=1, S=1,C=2,W=3,
    # F=4,P=5, остальные 0
    simbol = str(ticket)[0]
    if simbol == 'A':
        return 1
    elif simbol == 'S':
        return 1
    elif simbol == 'C':
        return 2
    elif simbol == 'W':
        return 3
    elif simbol == 'F':
        return 4
    elif simbol == 'P':
        return 5
    return 0


def is_young(age): #проверка на то молодой ли пассажир, для этого присваиваем наибольший приоритет детям младше 12,
    # детям младше 17 чуть меньший приоритет
    if (age > 35) & (age < 45):
        return 0.2
    if (age < 17) & (age > 11):
        return 0.7
    if age < 12:
        return 1
    return 0


def normilize_cabins_num(cabin): #выделение номера кабины из первичных данных
    if cabin:
        result = re.search(r'\d+', str(cabin))
        return int(result.group(0)) % 2 if result else -1
    return -1


def is_have_cabin_number(cabin): # выясняем указан ли номер кабины или нет
    if cabin:
        result = re.search(r'([a-z]|[A-Z])+', str(cabin))
        return result.group(0) if result else 0
    return 0


def normilize_fares(fare): #изменяем разброс цен билетов, для этого все билеты дороже 52 считаем равными 1, от 35 до 52
    # равными 0,48, от 26 до 35 равными 0,47, от 22 до 26 равными 0,45, от 18 до 22 равными 0,42, от 15 до 18 равными
    # 0,39, от 10 до 15 равными 0,32, если дешевле, считаем равным нулю
    if fare > 52:
        return 1
    elif fare > 35:
        return 0.48
    elif fare > 26:
        return 0.47
    elif fare > 22:
        return 0.46
    elif fare > 18:
        return 0.42
    elif fare > 15:
        return 0.39
    elif fare > 10:
        return 0.32
    else:
        return 0


def normilize_family_size(family): #изменяем данные о размере семьи, считаем количество родственников, и вводим
    # коэфиценты свыше 7 = 0,1, от 5 до 7 0,3, от 3 до 5 0,7, меньше 3 тогда 1, если ноль то 0
    if family > 7:
        return 0.1
    if family > 5:
        return 0.3
    if family > 3:
        return 0.7
    if family >= 1:
        return 1
    else:
        return 0


def normalize_ticket(number): #вырезаем последние три цыфры
    return round(number / 1000, 1)


def normilize_data(data): #нормализуем каждое поле, а именно от каждого элемента столбца отнимаем нименьшее значение и
    # делим на максимальное
    for r in data:
        data[r] = data[r].astype(float)
        data[r] = data[r] - min(data[r])
        data[r] = data[r] / max(data[r])


def normilize(data): #основной процесс нормализации
    average_age_mr = data["Age"].mean() #считаем средний возраст
    std_age_mr = data["Age"].std() #считаем среднеквадратичное отклонение
    count_nan_age_mr = data["Age"].isnull().sum() #считаем количество пустых ячеек
    rand_1 = np.random.randint(average_age_mr - std_age_mr, average_age_mr + std_age_mr,
                               size=count_nan_age_mr) #заполняем случайными значениями из диапазона среднеквадратичного
    # отклонения от среднего значения
    data["Age"][data["Age"].isnull()] = rand_1
    data['Family'] = data['Parch'] + data['SibSp'] #содаем новое поле Family, которое сосотоит из суммы Parch и SibSp
    data['Family'] = data['Family'].astype(int) #делаем это поле типа float
    data = data.drop(['Parch', 'SibSp'], axis=1) #удаляем Parch и SibSp и нормализуем поле Family
    data.Family = data.Family.apply(normilize_family_size)
    data["Is_cabin"] = data.Cabin.apply(normilize_cabins_num) #создаем поле, заполняем его значениями есть ли кабина
    data.Fare = data.Fare.apply(normilize_fares) / data.Pclass #нормализуем поле цен
    data.Cabin = data.Cabin.apply(is_have_cabin_number) #нормализуем данные о номере кабины
    data["Young"] = data.Age.apply(is_young) #создаем колонку, отвечающую за молодость пассажира
    data['Number'] = data.Ticket # создаем и нормализуем поле номера билета
    data['Number'] = data.Number.apply(normilize_ticket_identifier)
    data.Ticket = data.Ticket.apply(normilize_ticket_number) # выделяем из номера билета серию билета
    data.Name = data.Name.apply(get_title_from_family) #оставляем из имени только титул
    data.Ticket = data.Ticket.astype(float)
    data.Ticket = data.Ticket.apply(normalize_ticket)
    data = data.drop(['PassengerId'], axis=1) #удаляем данные о номере пассажира

    data['Embarked'] = data.Embarked.fillna("S") # делим поле Embarked на три новых бинарных поля, по трем его значениям
    # и удаляем одно из них из-за свойства мультипликативности, удаляем поле Embarked
    embark_dummies_titanic = pd.get_dummies(data['Embarked'])
    data = pd.concat([data, embark_dummies_titanic], axis=1)
    data.drop(['S'], axis=1, inplace=True)
    data.drop(['Embarked'], axis=1, inplace=True)
    pclass_dummies = pd.get_dummies(data.Pclass) # делим поле Pclass на три поля, в зависимости от класса, удаляем поле
    # Embarked и один из только что созданных полей по свойству мультипликативности
    pclass_dummies.columns = ['Class_1', 'Class_2', 'Class_3']
    data.drop(['Pclass'], axis=1, inplace=True)
    data = pd.concat([data, pclass_dummies], axis=1)
    data.drop(['Class_3'], axis=1, inplace=True)

    label = LabelEncoder() #делаем замену в каждом нечисловом поле на числа, создав словарь соответсвия числа, ячейке
    dicts = {}
    label.fit(data.Sex.drop_duplicates()) # задаем список значений для кодирования
    dicts['Sex'] = list(label.classes_)
    data.Sex = label.transform(data.Sex) # заменяем значения из списка кодами закодированных элементов

    label.fit(data.Number.drop_duplicates())
    dicts['Number'] = list(label.classes_)
    data.Number = label.transform(data.Number)
    label.fit(data.Cabin.drop_duplicates())
    dicts['Cabin'] = list(label.classes_)
    data.Cabin = label.transform(data.Cabin)
    data.Is_cabin = data.Is_cabin.fillna(0)
    data = data.drop(['Cabin'], axis=1) #удаляем поле кабины
    data = data.drop(['Age'], axis=1) # кдаляем поле возраста
    normilize_data(data)
    return data
