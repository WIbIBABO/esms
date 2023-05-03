import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import warnings;warnings.filterwarnings('ignore')
# dataset = load_breast_cancer()
# data = dataset.data
# target = dataset.target
# X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2)
# rf = RandomForestClassifier(n_estimators=5)
# rf.fit(X_train,y_train)
# pred = rf.predict_proba(X_test)[:,1]
y_test = np.loadtxt('./1label.csv')
pred = np.loadtxt('./1pre.csv')
sum = 0
for i in range(len(y_test)):
    if abs(y_test[i] - pred[i])<0.5:
        sum += 1
acc = sum / len(y_test)
ma = max(pred)
mi = min(pred)
pred = (pred-mi)/(ma - mi)
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %1f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
