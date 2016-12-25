# coding=utf-8

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import best_classifier
from my_network import MyNetwork
import normilize_data

train_data = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, ) #считываем тренировочные данные в таблицу
test_data = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, ) #считываем тестовые данные в таблицу
train_data = normilize_data.normilize(train_data) # производим нормализацию тренировочной таблицы
test_data = normilize_data.normilize(test_data) # производим нормализацию тестовой таблицы


def corrcoef(data): # считаем корреляцию каждого признака попарно, чтобы увидеть общую картину
    f = True
    df = None
    for r in data: # проходимся по всем параметрам
        l = list() # создаем лист, в котором будет хранится корреляция для данного объекта
        for p in data: # проходимся по всем парам
            l.append(round(np.corrcoef(data[r], data[p])[0, 1], 2)) # высчитываем линейную корреляцию Пирсона, считаем
            # ее как  средние значения деленные на среднеквадратичное отклонение
        d = list()
        d.append(l) #добавляем это значение в лист
        if f: #после этого создаем таблицу, куда записываем все попарные соотношения корреляции(на пересечении
            # строки и столбца)
            df = pd.DataFrame(d, columns=data.columns)
        else:
            df2 = pd.DataFrame(d, columns=data.columns)
            df = df.append(df2, ignore_index=True)
        f = False
    print df


corrcoef(train_data)
model_rfc = RandomForestClassifier(n_estimators=90, min_samples_split=8)  # создаем случайный лес, в котором не меньше 8
# объектов в каждой вершине и 90 деревьев
model_knc = KNeighborsClassifier(n_neighbors=12)  # создаем классификатор к-ближних соседей, в параметре передаем кол-во
#  соседей, равное 12
model_lr = LogisticRegression(penalty='l1', tol=0.01) #создаем логистическую регрессию
model_svc = svm.SVC()  # создаем классификатор
net = MyNetwork() #создаем нейронную сеть
model_svc.probability = True
data = train_data
model = best_classifier.calculate_best_classifier(model_rfc, model_knc, model_lr, model_svc, net, data) # выяснем
# лучший классификатор для задачи
data = train_data.drop(['Survived'], axis=1) #отделяем данные о результатах и выборку
model.fit(data, train_data['Survived']) #обучаемся на лучшем классификаторе
result = model.predict(X=test_data) #получаем результат обучения на тестовых данных
test_data2 = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, ) #записываем результат в файл
submission = pd.DataFrame({
    "PassengerId": test_data2["PassengerId"],
    "Survived": result
})

submission.Survived = submission.Survived.astype(int)
submission.to_csv("titanic-submission.csv", index=False)
