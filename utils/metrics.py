import numpy as np
from scipy.stats import gamma
import time
from sklearn.metrics import roc_auc_score
import logging
def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, axis=1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, axis=1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2 * np.dot(pattern1, pattern2.T)
    H = np.exp(-H / (2 * deg**2))

    return H

def hsic_gam(X, Y, alph=0.5):
    n = X.shape[0]

    # ----- width of X -----
    Xmed = X
    G = np.sum(Xmed * Xmed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))
    # ----- -----

    # ----- width of Y -----
    Ymed = Y
    G = np.sum(Ymed * Ymed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))
    # ----- -----

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2
    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)
    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC * n / mHSIC

    thresh = gamma.ppf(1 - alph, al, scale=bet)[0]

    return testStat, thresh


def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        unique_labels = np.unique(label[index_ui])
        if len(unique_labels) < 2:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction or only one class users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    logging.info(f"computed user: {auc_for_user.shape[0]} can not users: {only_one_class}")
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user