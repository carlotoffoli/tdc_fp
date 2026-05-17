# Split stratificato 80/20 (stratify=y conserva la proporzione delle classi, importante con solo 297 campioni)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC



seed = 2141118
y = Y.ravel()   # sklearn vuole un vettore 1D
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

# Confronto modelli con 5-fold CV solo sul training set (test set intoccato fino alla valutazione finale)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = seed)
candidates = {
    "Perceptron":          Perceptron(random_state=seed),
    "SVM (Linear)":       SVC(kernel="linear", random_state=seed),
    "SVM (Gaussian)":           SVC(kernel="rbf",    random_state=seed),
}
for name, m in candidates.items():
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy")
    print(name, "-> Mean: ", round(s.mean(), 3), " Std Deviation: ", round(s.std(), 3))


# Il GridSearchCV senza uno scoring specifico lavora usando l'accuracy(massimizzo l'accuracy = minimizzo il generalization error)
# C e gamma sono gli iperparametri del SVM, gamma mi dice quanto i dati di un paziente debbano essere simili a quelli di uno precedentamente diagnosticato per ricevere la stessa diagnosi
# quindi gamma grande significa rischio overfitting. C decide l'errore tollerabile sulla classificazione, C grande significa rischio overfitting
grid = GridSearchCV(
    SVC(kernel="rbf", random_state=seed),
    param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1, 10]},
    cv=cv).fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_svm = grid.best_estimator_

# Valutazione sul test set
y_pred  = best_svm.predict(X_test)
y_score = best_svm.decision_function(X_test)


# - Precision: di quelli che ho pedetto malati, quanti lo sono davvero
# - Recall: dei malati veri, quanti ne ho beccati
# - F1: media di precision e recall
print("Test set results: ",
    "\nAccurancy: ", round(accuracy_score(y_test, y_pred), 3),
    "\nPrecision: ", round(precision_score(y_test, y_pred), 3),
    "\nRecall: ", round(recall_score(y_test, y_pred), 3))
