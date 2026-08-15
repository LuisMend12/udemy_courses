"""Generic precomputed-kernel model selection: k-fold CV over a hyperparameter
grid, refit best config on the full train set, evaluate once on held-out test.
Kept manual (rather than sklearn's GridSearchCV pairwise-tag path) so the same
protocol is guaranteed identical across the quantum and classical kernels."""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


def cv_select_and_test(K_full, y, train_idx, test_idx, C_grid, n_folds=5, seed=0):
    """K_full: (N,N) Gram matrix over ALL samples (already includes kernel
    hyperparameters baked in, e.g. a fixed RBF gamma). Selects C by CV
    accuracy on train_idx, refits on all of train_idx, reports test accuracy.
    """
    y_train = y[train_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    best_C, best_cv_acc = None, -np.inf
    for C in C_grid:
        accs = []
        for fold_tr, fold_val in skf.split(train_idx, y_train):
            tr_idx = train_idx[fold_tr]
            val_idx = train_idx[fold_val]
            K_tr = K_full[np.ix_(tr_idx, tr_idx)]
            K_val = K_full[np.ix_(val_idx, tr_idx)]
            clf = SVC(kernel="precomputed", C=C, max_iter=200_000)
            clf.fit(K_tr, y[tr_idx])
            accs.append(clf.score(K_val, y[val_idx]))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_cv_acc:
            best_cv_acc, best_C = mean_acc, C

    K_train_full = K_full[np.ix_(train_idx, train_idx)]
    K_test = K_full[np.ix_(test_idx, train_idx)]
    clf = SVC(kernel="precomputed", C=best_C, max_iter=200_000)
    clf.fit(K_train_full, y_train)
    test_acc = clf.score(K_test, y[test_idx])

    return {"best_C": best_C, "cv_acc": best_cv_acc, "test_acc": float(test_acc)}
