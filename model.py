import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# ========= Data pre-processing =========

dataset = pd.read_csv('./heart_disease_data.csv')
X = dataset.drop('target',axis=1)
Y = dataset['target']

# if we don't set Stratify, all data of the train may crossponds to Y=1 or Y=0
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)


# ========= Model training =========

model = LinearRegression()
model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)
X_test_prediction_binary = np.round(X_test_prediction)

test_data_accuracy = accuracy_score(X_test_prediction_binary, Y_test)


def test_input(age, sex, cp, tresp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.asarray([
        float(age), float(sex), float(cp), float(tresp), float(chol), float(fbs),
        float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
        float(ca), float(thal)
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    prediction = np.round(prediction)
    print(f"\n\n\n\n\n\nModel Accuracy: {test_data_accuracy}\n\n\n\n\n\n")

    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'
