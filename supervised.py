# Split stratificato 80/20 (stratify=y conserva la proporzione delle classi, importante con solo 297 campioni)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

def algo_selection(X_train, X_test, y_train, y_test, seed):
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

    return y_pred, y_score