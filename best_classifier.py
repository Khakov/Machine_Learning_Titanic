# coding=utf-8
import pylab as pl
from sklearn import cross_validation
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def calculate_best_classifier(model_rfc, model_knc, model_lr, model_svc, net, data):  # считаем лучший показатель
    target = data.Survived  # отделяем ответ от выборки
    train = data.drop(['Survived'], axis=1)
    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)
    # производим кросс-валидационную проверку, поделив выборку на 2 части, тренировочную и тесовую, на тестовую 25 %
    itog_val = {}
    best_model = model_svc
    best_result = 0
    best_name = 'SVC'

    def calculate_score(type, name, best_name, best_model, best_result):  # функция для подсчета среднего значения
        # классификатора
        cv = 3
        scores = cross_validation.cross_val_score(type, train, target, cv=cv) # считаем данные кросс-валидации
        itog_val[name] = scores.mean() #подсчитываем среднее значение кросс-валидации
        if scores.mean() > best_result: # сравниваем с лучшим значением, если результат лучше, то делаем даыный
            # класификатор лучшим, иначе оставляем все как есть
            best_model = type
            best_name = name
            best_result = itog_val[name]
        return best_name, best_model, best_result

    net.fit(ROCtrainTRN, ROCtrainTRG) #прежде чем тестить нейроннцую сеть, ее нужно обучить
    # проверяем каждый классификатор на качество и выясняем лучший классификатор, на основе кросс-валидационных данных
    best_name, best_model, best_result = calculate_score(model_rfc, 'RandomForestClassifier', best_name, best_model,
                                                         best_result)
    best_name, best_model, best_result = calculate_score(model_knc, 'KNeighborsClassifier', best_name, best_model,
                                                         best_result)
    best_name, best_model, best_result = calculate_score(model_lr, 'LogisticRegression', best_name, best_model,
                                                         best_result)
    best_name, best_model, best_result = calculate_score(model_svc, 'SVC', best_name, best_model, best_result)
    itog_val['My_network'] = float(net.true_percent(ROCtestTRN, ROCtestTRG)) / 100.0
    if itog_val['My_network'] > best_result:
        best_model = net
        best_name = 'Network'
        best_result = itog_val['My_network']
    # рисуем ROC-кривую
    draw_ROC(model_rfc, model_knc, model_lr, model_svc, ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG)
    print 'best result:' + best_name + ' with ' + str(best_result)
    return best_model #возвращаем лучший классификатор


def draw_ROC(model_1, model_2, model_3, model_4, ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG): #рисуем ROC-кривую
    def draw_oneROC(model, name): # прорисовка одного из классификаторов
        probas = model.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN) # учим модель на тренировочных данных
        fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1]) #проверяем результат с помощью построения ROC-кривой
        # ROC кривая откладывает по оси Ox True Positive Rate (мы предполагаем правду, она оказывается правдой),
        # по Oy False Negative Rate(мы предполагаем ложь, она является ложью), на основе этих даных строим
        # график данной классификации.
        roc_auc = auc(fpr, tpr)
        pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, roc_auc))

    pl.clf()
    #строим график для каждой классификации
    draw_oneROC(model_1, 'SVC')
    draw_oneROC(model_2, 'RandonForest')
    draw_oneROC(model_3, 'KNeighborsClassifier')
    draw_oneROC(model_4, 'LogisticRegression')
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc=0, fontsize='small')
    pl.show()