"""
@author: Grisolia Giuseppe

Questo modulo analizza un dataset riguardante gliomi correlati a mutazioni molecolari
utilizzando classificatore K-NN. Viene effettuato il caricamento dei dati, la loro elaborazione,
l'addestramento del modello, e l'analisi delle sue prestazioni attraverso metriche e grafici.
"""

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Caricamento del dataset dal file .csv
dataset = pd.read_csv('../2.Ontologia/TCGA_InfoWithGrade.csv')

# esplorazione del dataset
print(dataset.info())

# divisione dati in features di input e feature target
y = dataset['Grade']
X = dataset.drop('Grade', axis=1)  # 'Grade' è la colonna target

# divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# calcolo del numero di vicini ottimale da utilizzare per il knn, sulla base di chi da il minor valore di mean error
error = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# grafico che mostra i cambiamenti dell'mean error, al cambiare del numero di vicini
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

# addestramento del modello con il numero di vicini ottimale trovato (7)
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)

# effettua previsioni sul test set
prediction = neigh.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f"accuracy_score: {accuracy:.2f}")

# stampa del classification report e della matrice di confusione
print('\nClassification report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

# creazione grafica della matrice di confusione
confusion_matrix = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()

# k-fold cross-validation (con k=5)
cv_scores = cross_val_score(neigh, X, y, cv=5)

# stampa statistiche dell k-fold cv
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))


# calcolo delle probabilità e dell'AUC score per la curva di ROC
probs = neigh.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print('\nAUC: %.3f' % auc)

# calcolo della curva di ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)

# tracciando graficamente la curva di ROC
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()

# # Calcolo dell'average precision e visualizzazione della curva precision-recall
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

# In matplotlib < 1.5, plt.fill_betwee non dispone dell'argomento 'step'
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Calcolo dell'F1-score
f1 = f1_score(y_test, prediction)
print('\nf1 score: ', f1)

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()
