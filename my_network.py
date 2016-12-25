# coding=utf-8
import random

from pybrain import SoftmaxLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import pandas as pd
from pybrain.utilities import percentError


class MyNetwork(object):
    def __init__(self):
        self.ds_train = {}
        self.ds_test = {}
        self.net = None
        random.seed(0)  # Зафиксируем seed для получния воспроизводимого результата

    def network_normilize(self, data, target=None):  # нормализуем входные данные для нейронной сети
        if target is None:
            target = pd.DataFrame({"id": [1 for i in range(len(data))]})["id"]  # если входным параметром не подан
            # результат то заполняем его единичками
        ds_data = ClassificationDataSet(np.shape(data.values)[1],
                                        nb_classes=2)  # Первый аргумент - количество признаков np.shape(X)[1],
        # второй аргумент -- количество меток классов len(np.unique(y_train)))
        ds_data.setField('input', data.values)  # Инициализация объектов
        ds_data.setField('target',
                         (target.values)[:, np.newaxis])  # Инициализация ответов; np.newaxis создает вектор-столбец
        ds_data._convertToOneOfMany()  # Бинаризация вектора ответов
        return ds_data

    def fit(self, train, target): # производим обучение на входных данных
        self.ds_train = self.network_normilize(train, target) # нормализуем входные данные
        self.net = buildNetwork(self.ds_train.indim, 13, self.ds_train.outdim, outclass=SoftmaxLayer) # создаем
        # нейронную сеть, состоящую из 13 скрытых узлов, 13 входных элементов и 1 выходного, в качестве функции
        # активвации нужно задать функцию сигмоиды f(x)= 1/ (1+ e^x), она будет приближать значения к более точным
        init_params = np.random.random((len(self.net.params)))
        # Инициализируем веса сети для получения воспроизводимого результата
        self.net._setParameters(init_params)
        trainer = BackpropTrainer(self.net, dataset=self.ds_train)  # Инициализируем модуль оптимизации
        err_train, err_val = trainer.trainUntilConvergence(maxEpochs=300) #проводим тренировку нейронной сети 300 раз

    def predict(self, X=None, Y=None): # производим предсказание
        if X is None: #если входных данных нет, то заполняем их тренировочными данными, если есть, то нормализуем
            X = self.ds_train
            Y = self.ds_train
        if Y is None:
            self.ds_test = self.network_normilize(X)
        else:
            self.ds_test = self.network_normilize(X, Y)
        return self.net.activateOnDataset(self.ds_test).argmax(axis=1) #производим активацию нейронной сети (подаем
        # нейронной тестовые данные, на которых и будт выдан результат)

    def true_percent(self, train, target): # вычисление точности нейронной сети
        self.predict(X=train, Y=target) #запускаем предсказание на данных
        return str(100.0 - float(percentError(self.predict(X=train), target.values)))  # сравниваем количество
        # правильных и ошибочных ответов, вычисляем процент правильных
