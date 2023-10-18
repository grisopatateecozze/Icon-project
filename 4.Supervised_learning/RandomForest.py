"""
@author: Grisolia Giuseppe

Questo modulo analizza un dataset riguardante gliomi correlati a mutazioni molecolari
utilizzando una RANDOM FOREST. Viene effettuato il caricamento dei dati, la loro elaborazione,
l'addestramento del modello, e l'analisi delle sue prestazioni attraverso metriche e grafici.
"""

import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from inspect import signature

# Caricamento del dataset da un file CSV
dataset = pd.read_csv("../Ontologia/TCGA_InfoWithGrade.csv")

# esplorazione del dataset
print(dataset.info())

# divido i dati in features di input e feature target
y = dataset['Grade']
X = dataset.drop('Grade', axis=1)

# divido il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# ------------------------------------------------------------------------------------------------------------
# Definire un intervallo di valori da testare per max_depth
max_depth_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Definire un intervallo di valori per verificare lo random_state
random_state_values = [0, 4, 16, 64, 256, 1024, 4096]

# Memorizza i punteggi medi di cross-validation per ogni combinazione di max_depth e random_state
scores = []
for max_depth in max_depth_values:
    for random_state in random_state_values:
        RFC = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
        cv_scores = cross_val_score(RFC, X, y, cv=5)
        scores.append((max_depth, random_state, cv_scores.mean()))

# Converti l'elenco dei punteggi (scores list) in un array numpy
scores = np.array(scores)

# Ottieni l'indice del punteggio massimo
best_index = np.argmax(scores[:, 2])

# Ottieni i valori di max_depth e random_state
best_max_depth = scores[best_index, 0]
best_random_state = scores[best_index, 1]

# Print the best max_depth and random_state values
# Stampa i valori di max_depth e random_state
print("Valore migliore max_depth: {}".format(best_max_depth))
print("Valore migliore random_state: {}".format(best_random_state))

clf = RandomForestClassifier(max_depth=int(best_max_depth), random_state=int(best_random_state), n_estimators=20)
# ----------------------------------------------------------------------------------------------------------------------

# Addestramento del classificatore Random Forest prova con parametri casuali
# clf1 = RandomForestClassifier(max_depth=30, random_state=0, n_estimators=20)
clf.fit(X_train, y_train)

# Effettua previsioni sul test set
prediction = clf.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print('\naccuracy_score:', accuracy)

# Stampa del rapporto di classificazione e della matrice di confusione
print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfussion matrix:\n', confusion_matrix(y_test, prediction))

# Creazione e visualizzazione della matrice di confusione come heatmap
confusion_matrix = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
pyplot.show()

# Valutazione del modello attraverso cross-validation (di 5)
cv_scores = cross_val_score(clf, X, y, cv=5)

# Stampa delle statistiche ottenute dalla cross-validation
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))
print('\n')

# Calcolo delle probabilità e dell'AUC per la curva ROC
probs = clf.predict_proba(X_test)
# Conserva solo le probabilità per l'outcome positivo
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# Calcola la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
# Disegna la curva ROC per il modello
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()

# Calcolo dell'average precision e visualizzazione della curva precision-recall
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

# In Matplotlib < 1.5, plt.fill_between non ha l'argomento 'step'
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

# Calcolo del punteggio F1
f1 = f1_score(y_test, prediction)
print('\nf1 score: ', f1)
# Creazione di un grafico a barre per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()
