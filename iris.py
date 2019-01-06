import pandas as pd
import numpy as np
from sklearn import svm



a = pd.read_csv('Iris.csv')

x = a.drop(['Class'],axis=1)
y = a['Class']

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split

xtr, xts, ytr, yts = train_test_split(x,y,test_size=0.2)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


knn = svm.SVC()
knn.fit(xtr, ytr)

print(knn.score(xts, yts))
predictions = knn.predict(xts)
print(accuracy_score(yts, predictions))
print(confusion_matrix(yts, predictions))
print(classification_report(yts, predictions))
