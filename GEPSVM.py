import numpy as np
import pandas as pd
from scipy.linalg import eigh, norm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
def GEP(A,B):
    b,w = eigh(A,B,subset_by_index=[0,0])
    return w
def Statlog_GEPSVM():
    statlog_data = fetch_ucirepo(id=145)
    X1= statlog_data.data.features
    X1_DF= pd.DataFrame(X1)
    Y1= statlog_data.data.targets
    Y1_DF = pd.DataFrame(Y1*2-3)
    delta = 35
    skf_df1 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    #pl1_pos_results = []
    #pl1_neg_results = []
    #pv1_pos_results = []
    #pv1_neg_results = []
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df1.split(X1_DF,Y1_DF)) :
        X1_train,Y1_train = X1_DF[X1_DF.index.isin(train_index)],Y1_DF[Y1_DF.index.isin(train_index)]
        X1_test,Y1_test = X1_DF[X1_DF.index.isin(test_index)],Y1_DF[Y1_DF.index.isin(test_index)]
        W1_train = X1_train.join(Y1_train)
        W1_train_pos = W1_train[W1_train['heart-disease']==1]
        W1_train_neg = W1_train[W1_train['heart-disease']==-1]
        W1_train_pos = W1_train_pos.drop(W1_train_pos.columns[[13]],axis=1)
        W1_train_neg = W1_train_neg.drop(W1_train_neg.columns[[13]],axis=1)
        bias_term_pos = [1]*W1_train_pos.shape[0]
        bias_term_neg = [1]*W1_train_neg.shape[0]
        W1_train_pos_bias = W1_train_pos.assign(bias = bias_term_pos)
        W1_train_neg_bias = W1_train_neg.assign(bias = bias_term_neg)
        W1_mtx_pos = W1_train_pos_bias.to_numpy()
        W1_mtx_neg = W1_train_neg_bias.to_numpy()
        G1_pos = W1_mtx_pos.T@W1_mtx_pos+np.identity(W1_mtx_pos.shape[1])*delta
        G1_neg = W1_mtx_neg.T@W1_mtx_neg+np.identity(W1_mtx_neg.shape[1])*delta
        H1_pos = W1_mtx_pos.T@W1_mtx_pos
        H1_neg = W1_mtx_neg.T@W1_mtx_neg
        b1_pos = GEP(G1_pos,H1_neg)
        b1_neg = GEP(G1_neg,H1_pos)
        b1_pos_list = b1_pos.tolist()
        b1_neg_list = b1_neg.tolist()
        pv1_pos, pv1_neg = b1_pos_list.pop(), b1_neg_list.pop()
        pl1_pos, pl1_neg = b1_pos_list, b1_neg_list
        #pv1_pos_results.append(pv1_pos[0])
        #pv1_neg_results.append(pv1_neg[0])
        #pl1_pos_results.append(np.array(pl1_pos))
        #pl1_neg_results.append(np.array(pl1_neg))
        X1_test_array, Y1_test_list = X1_test.to_numpy(), Y1_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X1_test.shape[0]) :
            dist1 = (abs(np.dot(X1_test_array[j],pl1_pos)+pv1_pos)/norm(pl1_pos))[0]
            dist2 = (abs(np.dot(X1_test_array[j],pl1_neg)+pv1_neg)/norm(pl1_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y1_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Statlog dataset ({X1_DF.shape[0]},{X1_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")
    #DF1 = X1_DF.join(Y1_DF)
    #DF1_pos = DF1[DF1["heart-disease"]==1]
    #DF1_neg = DF1[DF1["heart-disease"]==-1]
    #x1_pos = [i for i in range(0,1000)]
    #y1_pos = [-pv1_pos_results[0]/pl1_pos_results[0][7]-pl1_pos_results[0][0]/pl1_pos_results[0][7]*t for t in range(1000)]
    #x1_neg = [i for i in range(0,1000)]
    #y1_neg = [-pv1_neg_results[0]/pl1_neg_results[0][7]-pl1_neg_results[0][0]/pl1_pos_results[0][7]*t for t in range(1000)]
    #plt.scatter(x=DF1_pos["age"],y=DF1_pos["max-heart-rate"], marker="o", c="red")
    #plt.scatter(x=DF1_neg["age"],y=DF1_neg["max-heart-rate"], marker="x", c="blue")
    #plt.plot(x1_pos,y1_pos,c="green")
    #plt.plot(x1_neg,y1_neg,c="yellow")
    #plt.show()

def Heart_c_GEPSVM():
    Heart_c=fetch_ucirepo(id=45)
    X2_DF = pd.DataFrame(Heart_c.data.features)
    Y2_DF = pd.DataFrame(Heart_c.data.targets)
    DF2 = X2_DF.join(Y2_DF)
    DF2_clean = DF2.dropna()
    id = [i+1 for i in range(DF2_clean.shape[0])]
    X2_DF_clean = DF2_clean[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']]
    Y2_DF_clean = DF2_clean[['num']]
    X2_DF_clean_id = X2_DF_clean.assign(id=id)
    Y2_DF_clean_id = Y2_DF_clean.assign(id=id)
    skf_dF2 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    delta = 100
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_dF2.split(X2_DF_clean,Y2_DF_clean)) :
        X2_train,Y2_train = X2_DF_clean_id[X2_DF_clean_id["id"].isin(train_index)],Y2_DF_clean_id[Y2_DF_clean_id["id"].isin(train_index)]
        X2_test,Y2_test = X2_DF_clean_id[X2_DF_clean_id["id"].isin(test_index)],Y2_DF_clean_id[Y2_DF_clean_id["id"].isin(test_index)]
        X2_train = X2_train.drop(X2_train.columns[[13]],axis=1)
        Y2_train = Y2_train.drop(Y2_train.columns[[1]],axis=1)
        X2_test = X2_test.drop(X2_test.columns[[13]],axis=1)
        Y2_test = Y2_test.drop(Y2_test.columns[[1]],axis=1)
        W2_train = X2_train.join(Y2_train)
        W2_train_target_is_0 = W2_train[W2_train["num"]==0]
        W2_train_target_is_1 = W2_train[W2_train["num"]==1]
        W2_train_target_is_2 = W2_train[W2_train["num"]==2]
        W2_train_target_is_3 = W2_train[W2_train["num"]==3]
        W2_train_target_is_4 = W2_train[W2_train["num"]==4]
        W2_train_target_is_0_pos =  W2_train_target_is_0.drop(W2_train_target_is_0.columns[[13]],axis = 1)
        W2_train_target_is_1_pos =  W2_train_target_is_1.drop(W2_train_target_is_1.columns[[13]],axis = 1)
        W2_train_target_is_2_pos =  W2_train_target_is_2.drop(W2_train_target_is_2.columns[[13]],axis = 1)
        W2_train_target_is_3_pos =  W2_train_target_is_3.drop(W2_train_target_is_3.columns[[13]],axis = 1)
        W2_train_target_is_4_pos =  W2_train_target_is_4.drop(W2_train_target_is_4.columns[[13]],axis = 1)
        W2_train_target_is_0_neg = pd.concat([W2_train_target_is_1_pos,W2_train_target_is_2_pos,W2_train_target_is_3_pos,W2_train_target_is_4_pos])
        W2_train_target_is_1_neg = pd.concat([W2_train_target_is_0_pos,W2_train_target_is_2_pos,W2_train_target_is_3_pos,W2_train_target_is_4_pos])
        W2_train_target_is_2_neg = pd.concat([W2_train_target_is_0_pos,W2_train_target_is_1_pos,W2_train_target_is_3_pos,W2_train_target_is_4_pos])
        W2_train_target_is_3_neg = pd.concat([W2_train_target_is_0_pos,W2_train_target_is_1_pos,W2_train_target_is_2_pos,W2_train_target_is_4_pos])
        W2_train_target_is_4_neg = pd.concat([W2_train_target_is_0_pos,W2_train_target_is_1_pos,W2_train_target_is_2_pos,W2_train_target_is_3_pos])
        bias_term_target_is_0_pos = [1]*W2_train_target_is_0_pos.shape[0]
        bias_term_target_is_1_pos = [1]*W2_train_target_is_1_pos.shape[0]
        bias_term_target_is_2_pos = [1]*W2_train_target_is_2_pos.shape[0]
        bias_term_target_is_3_pos = [1]*W2_train_target_is_3_pos.shape[0]
        bias_term_target_is_4_pos = [1]*W2_train_target_is_4_pos.shape[0]
        bias_term_target_is_0_neg = [1]*W2_train_target_is_0_neg.shape[0]
        bias_term_target_is_1_neg = [1]*W2_train_target_is_1_neg.shape[0]
        bias_term_target_is_2_neg = [1]*W2_train_target_is_2_neg.shape[0]
        bias_term_target_is_3_neg = [1]*W2_train_target_is_3_neg.shape[0]
        bias_term_target_is_4_neg = [1]*W2_train_target_is_4_neg.shape[0]
        W2_train_target_is_0_pos_bias = W2_train_target_is_0_pos.assign(bias = bias_term_target_is_0_pos)
        W2_train_target_is_1_pos_bias = W2_train_target_is_1_pos.assign(bias = bias_term_target_is_1_pos)
        W2_train_target_is_2_pos_bias = W2_train_target_is_2_pos.assign(bias = bias_term_target_is_2_pos)
        W2_train_target_is_3_pos_bias = W2_train_target_is_3_pos.assign(bias = bias_term_target_is_3_pos)
        W2_train_target_is_4_pos_bias = W2_train_target_is_4_pos.assign(bias = bias_term_target_is_4_pos)
        W2_train_target_is_0_neg_bias = W2_train_target_is_0_neg.assign(bias = bias_term_target_is_0_neg)
        W2_train_target_is_1_neg_bias = W2_train_target_is_1_neg.assign(bias = bias_term_target_is_1_neg)
        W2_train_target_is_2_neg_bias = W2_train_target_is_2_neg.assign(bias = bias_term_target_is_2_neg)
        W2_train_target_is_3_neg_bias = W2_train_target_is_3_neg.assign(bias = bias_term_target_is_3_neg)
        W2_train_target_is_4_neg_bias = W2_train_target_is_4_neg.assign(bias = bias_term_target_is_4_neg)
        W2_mtx_target_is_0_pos = W2_train_target_is_0_pos_bias.to_numpy()
        W2_mtx_target_is_1_pos = W2_train_target_is_1_pos_bias.to_numpy()
        W2_mtx_target_is_2_pos = W2_train_target_is_2_pos_bias.to_numpy()
        W2_mtx_target_is_3_pos = W2_train_target_is_3_pos_bias.to_numpy()
        W2_mtx_target_is_4_pos = W2_train_target_is_4_pos_bias.to_numpy()
        W2_mtx_target_is_0_neg = W2_train_target_is_0_neg_bias.to_numpy()
        W2_mtx_target_is_1_neg = W2_train_target_is_1_neg_bias.to_numpy()
        W2_mtx_target_is_2_neg = W2_train_target_is_2_neg_bias.to_numpy()
        W2_mtx_target_is_3_neg = W2_train_target_is_3_neg_bias.to_numpy()
        W2_mtx_target_is_4_neg = W2_train_target_is_4_neg_bias.to_numpy()
        G2_target_is_0 = W2_mtx_target_is_0_pos.T@W2_mtx_target_is_0_pos+np.identity(W2_mtx_target_is_0_pos.shape[1])*delta
        G2_target_is_1 = W2_mtx_target_is_1_pos.T@W2_mtx_target_is_1_pos+np.identity(W2_mtx_target_is_1_pos.shape[1])*delta
        G2_target_is_2 = W2_mtx_target_is_2_pos.T@W2_mtx_target_is_2_pos+np.identity(W2_mtx_target_is_2_pos.shape[1])*delta
        G2_target_is_3 = W2_mtx_target_is_3_pos.T@W2_mtx_target_is_3_pos+np.identity(W2_mtx_target_is_3_pos.shape[1])*delta
        G2_target_is_4 = W2_mtx_target_is_4_pos.T@W2_mtx_target_is_4_pos+np.identity(W2_mtx_target_is_0_pos.shape[1])*delta
        H2_target_is_0 = W2_mtx_target_is_0_neg.T@W2_mtx_target_is_0_neg
        H2_target_is_1 = W2_mtx_target_is_1_neg.T@W2_mtx_target_is_1_neg
        H2_target_is_2 = W2_mtx_target_is_2_neg.T@W2_mtx_target_is_2_neg
        H2_target_is_3 = W2_mtx_target_is_3_neg.T@W2_mtx_target_is_3_neg
        H2_target_is_4 = W2_mtx_target_is_4_neg.T@W2_mtx_target_is_4_neg
        b2_target_is_0 = GEP(G2_target_is_0,H2_target_is_0)
        b2_target_is_1 = GEP(G2_target_is_1,H2_target_is_1)
        b2_target_is_2 = GEP(G2_target_is_2,H2_target_is_2)
        b2_target_is_3 = GEP(G2_target_is_3,H2_target_is_3)
        b2_target_is_4 = GEP(G2_target_is_4,H2_target_is_4)
        b2_target_is_0_list,b2_target_is_1_list, b2_target_is_2_list, b2_target_is_3_list, b2_target_is_4_list = b2_target_is_0.tolist(),b2_target_is_1.tolist(),b2_target_is_2.tolist(),b2_target_is_3.tolist(),b2_target_is_4.tolist()
        pv2_target_is_0, pv2_target_is_1, pv2_target_is_2, pv2_target_is_3, pv2_target_is_4 = b2_target_is_0_list.pop(), b2_target_is_1_list.pop(), b2_target_is_2_list.pop(), b2_target_is_3_list.pop(),b2_target_is_4_list.pop()
        pl2_target_is_0, pl2_target_is_1, pl2_target_is_2, pl2_target_is_3, pl2_target_is_4 = b2_target_is_0_list, b2_target_is_1_list, b2_target_is_2_list, b2_target_is_3_list, b2_target_is_4_list
        X2_test_array, Y2_test_list = X2_test.to_numpy(), Y2_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X2_test.shape[0]) :
            dist1 = (abs(np.dot(X2_test_array[j],pl2_target_is_0)+pv2_target_is_0)/norm(pl2_target_is_0))[0]
            dist2 = (abs(np.dot(X2_test_array[j],pl2_target_is_1)+pv2_target_is_1)/norm(pl2_target_is_1))[0]
            dist3 = (abs(np.dot(X2_test_array[j],pl2_target_is_2)+pv2_target_is_2)/norm(pl2_target_is_2))[0]
            dist4 = (abs(np.dot(X2_test_array[j],pl2_target_is_3)+pv2_target_is_3)/norm(pl2_target_is_3))[0]
            dist5 = (abs(np.dot(X2_test_array[j],pl2_target_is_4)+pv2_target_is_4)/norm(pl2_target_is_4))[0]
            min_index = 0
            min_val = 1E10
            for s,t in enumerate([dist1,dist2,dist3,dist4,dist5]) :
                if t < min_val:
                    min_index = s
                    min_val = t
            temp.append(min_index)
        for k in range(len(temp)) :
            if temp[k] == Y2_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Heart-c dataset ({X2_DF_clean.shape[0]},{X2_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Hepatitis_GEPSVM():
    Hepatitis=fetch_ucirepo(id=46)
    X3_DF = pd.DataFrame(Hepatitis.data.features)
    Y3_DF = pd.DataFrame(Hepatitis.data.targets)
    DF3 = X3_DF.join(Y3_DF*2-3)
    DF3_clean = DF3.dropna()
    X3_DF_clean = DF3_clean[['Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia',
                             'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites',
                             'Varices', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime',
                             'Histology']]
    Y3_DF_clean = DF3_clean[['Class']]
    delta = 35
    prec_fix = 1E-10
    id = [i+1 for i in range(DF3_clean.shape[0])]
    X3_DF_clean_id = X3_DF_clean.assign(id=id)
    Y3_DF_clean_id = Y3_DF_clean.assign(id=id)
    skf_df3 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    delta = 0.01
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df3.split(X3_DF_clean,Y3_DF_clean)) :
        X3_train,Y3_train = X3_DF_clean_id[X3_DF_clean_id["id"].isin(train_index)],Y3_DF_clean_id[Y3_DF_clean_id["id"].isin(train_index)]
        X3_test,Y3_test = X3_DF_clean_id[X3_DF_clean_id["id"].isin(test_index)],Y3_DF_clean_id[Y3_DF_clean_id["id"].isin(test_index)]
        X3_train = X3_train.drop(X3_train.columns[[19]],axis=1)
        Y3_train = Y3_train.drop(Y3_train.columns[[1]],axis=1)
        X3_test = X3_test.drop(X3_test.columns[[19]],axis=1)
        Y3_test = Y3_test.drop(Y3_test.columns[[1]],axis=1)
        W3_train = X3_train.join(Y3_train)
        W3_train_pos = W3_train[W3_train['Class']==1]
        W3_train_neg = W3_train[W3_train['Class']==-1]
        W3_train_pos = W3_train_pos.drop(W3_train_pos.columns[[19]],axis=1)
        W3_train_neg = W3_train_neg.drop(W3_train_neg.columns[[19]],axis=1)
        bias_term_pos = [1]*W3_train_pos.shape[0]
        bias_term_neg = [1]*W3_train_neg.shape[0]
        W3_train_pos_bias = W3_train_pos.assign(bias = bias_term_pos)
        W3_train_neg_bias = W3_train_neg.assign(bias = bias_term_neg)
        W3_mtx_pos = W3_train_pos_bias.to_numpy()
        W3_mtx_neg = W3_train_neg_bias.to_numpy()
        G3_pos = W3_mtx_pos.T@W3_mtx_pos+np.identity(W3_mtx_pos.shape[1])*delta
        G3_neg = W3_mtx_neg.T@W3_mtx_neg+np.identity(W3_mtx_neg.shape[1])*delta
        H3_pos = W3_mtx_pos.T@W3_mtx_pos+np.identity(W3_mtx_pos.shape[1])*prec_fix
        H3_neg = W3_mtx_neg.T@W3_mtx_neg+np.identity(W3_mtx_neg.shape[1])*prec_fix
        b3_pos = GEP(G3_pos,H3_neg)
        b3_neg = GEP(G3_neg,H3_pos)
        b3_pos_list = b3_pos.tolist()
        b3_neg_list = b3_neg.tolist()
        pv3_pos, pv3_neg = b3_pos_list.pop(), b3_neg_list.pop()
        pl3_pos, pl3_neg = b3_pos_list, b3_neg_list
        X3_test_array, Y3_test_list = X3_test.to_numpy(), Y3_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X3_test.shape[0]) :
            dist1 = (abs(np.dot(X3_test_array[j],pl3_pos)+pv3_pos)/norm(pl3_pos))[0]
            dist2 = (abs(np.dot(X3_test_array[j],pl3_neg)+pv3_neg)/norm(pl3_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y3_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Hepatitis dataset ({X3_DF_clean.shape[0]},{X3_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Ionosphere_GEPSVM(): 
    Ionoshpere=fetch_ucirepo(id=52)
    X4_DF = pd.DataFrame(Ionoshpere.data.features)
    Y4_DF = pd.DataFrame(Ionoshpere.data.targets)
    Y4_DF['val'] = pd.factorize(Y4_DF['Class'])[0]*2-1
    Y4_DF = Y4_DF.drop('Class',axis = 1)
    delta = 70
    prec_fix = 1E-10
    skf_df4 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df4.split(X4_DF,Y4_DF)) :
        X4_train,Y4_train = X4_DF[X4_DF.index.isin(train_index)],Y4_DF[Y4_DF.index.isin(train_index)]
        X4_test,Y4_test = X4_DF[X4_DF.index.isin(test_index)],Y4_DF[Y4_DF.index.isin(test_index)]
        W4_train = X4_train.join(Y4_train)
        W4_train_pos = W4_train[W4_train['val']==1]
        W4_train_neg = W4_train[W4_train['val']==-1]
        W4_train_pos = W4_train_pos.drop(W4_train_pos.columns[[34]],axis=1)
        W4_train_neg = W4_train_neg.drop(W4_train_neg.columns[[34]],axis=1)
        bias_term_pos = [1]*W4_train_pos.shape[0]
        bias_term_neg = [1]*W4_train_neg.shape[0]
        W4_train_pos_bias = W4_train_pos.assign(bias = bias_term_pos)
        W4_train_neg_bias = W4_train_neg.assign(bias = bias_term_neg)
        W4_mtx_pos = W4_train_pos_bias.to_numpy()
        W4_mtx_neg = W4_train_neg_bias.to_numpy()
        G4_pos = W4_mtx_pos.T@W4_mtx_pos+np.identity(W4_mtx_pos.shape[1])*delta
        G4_neg = W4_mtx_neg.T@W4_mtx_neg+np.identity(W4_mtx_neg.shape[1])*delta
        H4_pos = W4_mtx_pos.T@W4_mtx_pos+np.identity(W4_mtx_pos.shape[1])*prec_fix
        H4_neg = W4_mtx_neg.T@W4_mtx_neg+np.identity(W4_mtx_neg.shape[1])*prec_fix
        b4_pos = GEP(G4_pos,H4_neg)
        b4_neg = GEP(G4_neg,H4_pos)
        b4_pos_list = b4_pos.tolist()
        b4_neg_list = b4_neg.tolist()
        pv4_pos, pv4_neg = b4_pos_list.pop(), b4_neg_list.pop()
        pl4_pos, pl4_neg = b4_pos_list, b4_neg_list
        X4_test_array, Y4_test_list = X4_test.to_numpy(), Y4_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X4_test.shape[0]) :
            dist1 = (abs(np.dot(X4_test_array[j],pl4_pos)+pv4_pos)/norm(pl4_pos))[0]
            dist2 = (abs(np.dot(X4_test_array[j],pl4_neg)+pv4_neg)/norm(pl4_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y4_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Ionosphere dataset ({X4_DF.shape[0]},{X4_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Sonar_GEPSVM():
    Sonar=fetch_ucirepo(id=151)
    X5_DF = pd.DataFrame(Sonar.data.features)
    Y5_DF = pd.DataFrame(Sonar.data.targets)
    Y5_DF['val'] = pd.factorize(Y5_DF['class'])[0]*2-1
    Y5_DF = Y5_DF.drop('class',axis = 1)
    delta = 25
    prec_fix = 1E-10
    skf_df5 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df5.split(X5_DF,Y5_DF)) :
        X5_train,Y5_train = X5_DF[X5_DF.index.isin(train_index)],Y5_DF[Y5_DF.index.isin(train_index)]
        X5_test,Y5_test = X5_DF[X5_DF.index.isin(test_index)],Y5_DF[Y5_DF.index.isin(test_index)]
        W5_train = X5_train.join(Y5_train)
        W5_train_pos = W5_train[W5_train['val']==1]
        W5_train_neg = W5_train[W5_train['val']==-1]
        W5_train_pos = W5_train_pos.drop(W5_train_pos.columns[[60]],axis=1)
        W5_train_neg = W5_train_neg.drop(W5_train_neg.columns[[60]],axis=1)
        bias_term_pos = [1]*W5_train_pos.shape[0]
        bias_term_neg = [1]*W5_train_neg.shape[0]
        W5_train_pos_bias = W5_train_pos.assign(bias = bias_term_pos)
        W5_train_neg_bias = W5_train_neg.assign(bias = bias_term_neg)
        W5_mtx_pos = W5_train_pos_bias.to_numpy()
        W5_mtx_neg = W5_train_neg_bias.to_numpy()
        G5_pos = W5_mtx_pos.T@W5_mtx_pos+np.identity(W5_mtx_pos.shape[1])*delta
        G5_neg = W5_mtx_neg.T@W5_mtx_neg+np.identity(W5_mtx_neg.shape[1])*delta
        H5_pos = W5_mtx_pos.T@W5_mtx_pos+np.identity(W5_mtx_pos.shape[1])*prec_fix
        H5_neg = W5_mtx_neg.T@W5_mtx_neg+np.identity(W5_mtx_neg.shape[1])*prec_fix
        b5_pos = GEP(G5_pos,H5_neg)
        b5_neg = GEP(G5_neg,H5_pos)
        b5_pos_list = b5_pos.tolist()
        b5_neg_list = b5_neg.tolist()
        pv5_pos, pv5_neg = b5_pos_list.pop(), b5_neg_list.pop()
        pl5_pos, pl5_neg = b5_pos_list, b5_neg_list
        X5_test_array, Y5_test_list = X5_test.to_numpy(), Y5_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X5_test.shape[0]) :
            dist1 = (abs(np.dot(X5_test_array[j],pl5_pos)+pv5_pos)/norm(pl5_pos))[0]
            dist2 = (abs(np.dot(X5_test_array[j],pl5_neg)+pv5_neg)/norm(pl5_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y5_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Sonar dataset ({X5_DF.shape[0]},{X5_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Vote_GEPSVM():
    Vote=fetch_ucirepo(id=105)
    X6_DF = pd.DataFrame(Vote.data.features)
    Y6_DF = pd.DataFrame(Vote.data.targets)
    DF6 = X6_DF.join(Y6_DF) 
    DF6_clean = DF6.dropna()
    DF6_clean_num = DF6_clean.apply(lambda x: x.astype('category').cat.codes)
    DF6_clean_num['Class'] = DF6_clean_num['Class']*2-1 
    X6_DF_clean = DF6_clean_num[['handicapped-infants', 'water-project-cost-sharing',
                             'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                             'el-salvador-aid', 'religious-groups-in-schools',
                             'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
                             'immigration', 'synfuels-corporation-cutback', 'education-spending',
                             'superfund-right-to-sue', 'crime', 'duty-free-exports',
                             'export-administration-act-south-africa']]
    Y6_DF_clean = DF6_clean_num[['Class']]
    delta = 35
    prec_fix = 1E-10
    id = [i+1 for i in range(DF6_clean_num.shape[0])]
    X6_DF_clean_id = X6_DF_clean.assign(id=id)
    Y6_DF_clean_id = Y6_DF_clean.assign(id=id)
    skf_df6 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    delta = 0.01
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df6.split(X6_DF_clean,Y6_DF_clean)) :
        X6_train,Y6_train = X6_DF_clean_id[X6_DF_clean_id["id"].isin(train_index)],Y6_DF_clean_id[Y6_DF_clean_id["id"].isin(train_index)]
        X6_test,Y6_test = X6_DF_clean_id[X6_DF_clean_id["id"].isin(test_index)],Y6_DF_clean_id[Y6_DF_clean_id["id"].isin(test_index)]
        X6_train = X6_train.drop(X6_train.columns[[16]],axis=1)
        Y6_train = Y6_train.drop(Y6_train.columns[[1]],axis=1)
        X6_test = X6_test.drop(X6_test.columns[[16]],axis=1)
        Y6_test = Y6_test.drop(Y6_test.columns[[1]],axis=1)
        W6_train = X6_train.join(Y6_train)
        W6_train_pos = W6_train[W6_train['Class']==1]
        W6_train_neg = W6_train[W6_train['Class']==-1]
        W6_train_pos = W6_train_pos.drop(W6_train_pos.columns[[16]],axis=1)
        W6_train_neg = W6_train_neg.drop(W6_train_neg.columns[[16]],axis=1)
        bias_term_pos = [1]*W6_train_pos.shape[0]
        bias_term_neg = [1]*W6_train_neg.shape[0]
        W6_train_pos_bias = W6_train_pos.assign(bias = bias_term_pos)
        W6_train_neg_bias = W6_train_neg.assign(bias = bias_term_neg)
        W6_mtx_pos = W6_train_pos_bias.to_numpy()
        W6_mtx_neg = W6_train_neg_bias.to_numpy()
        G6_pos = W6_mtx_pos.T@W6_mtx_pos+np.identity(W6_mtx_pos.shape[1])*delta
        G6_neg = W6_mtx_neg.T@W6_mtx_neg+np.identity(W6_mtx_neg.shape[1])*delta
        H6_pos = W6_mtx_pos.T@W6_mtx_pos+np.identity(W6_mtx_pos.shape[1])*prec_fix
        H6_neg = W6_mtx_neg.T@W6_mtx_neg+np.identity(W6_mtx_neg.shape[1])*prec_fix
        b6_pos = GEP(G6_pos,H6_neg)
        b6_neg = GEP(G6_neg,H6_pos)
        b6_pos_list = b6_pos.tolist()
        b6_neg_list = b6_neg.tolist()
        pv6_pos, pv6_neg = b6_pos_list.pop(), b6_neg_list.pop()
        pl6_pos, pl6_neg = b6_pos_list, b6_neg_list
        X6_test_array, Y6_test_list = X6_test.to_numpy(), Y6_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X6_test.shape[0]) :
            dist1 = (abs(np.dot(X6_test_array[j],pl6_pos)+pv6_pos)/norm(pl6_pos))[0]
            dist2 = (abs(np.dot(X6_test_array[j],pl6_neg)+pv6_neg)/norm(pl6_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y6_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Vote dataset ({X6_DF_clean.shape[0]},{X6_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Pima_Indian_GEPSVM():
    file_path = "C:\\Users\\chung\\OneDrive\\文件\\COMP3314 Group project\\diabetes.csv" 
    # change the file path to the current location of the .csv file of the Pima Indian Diabetes dataset
    Pima_Indian = pd.read_csv(file_path)
    X7_DF = Pima_Indian[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                         'BMI', 'DiabetesPedigreeFunction', 'Age']]
    Y7_DF= Pima_Indian[['Outcome']]*2-1
    delta = 1
    skf_df7 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df7.split(X7_DF,Y7_DF)) :
        X7_train,Y7_train = X7_DF[X7_DF.index.isin(train_index)],Y7_DF[Y7_DF.index.isin(train_index)]
        X7_test,Y7_test = X7_DF[X7_DF.index.isin(test_index)],Y7_DF[Y7_DF.index.isin(test_index)]
        W7_train = X7_train.join(Y7_train)
        W7_train_pos = W7_train[W7_train['Outcome']==1]
        W7_train_neg = W7_train[W7_train['Outcome']==-1]
        W7_train_pos = W7_train_pos.drop(W7_train_pos.columns[[8]],axis=1)
        W7_train_neg = W7_train_neg.drop(W7_train_neg.columns[[8]],axis=1)
        bias_term_pos = [1]*W7_train_pos.shape[0]
        bias_term_neg = [1]*W7_train_neg.shape[0]
        W7_train_pos_bias = W7_train_pos.assign(bias = bias_term_pos)
        W7_train_neg_bias = W7_train_neg.assign(bias = bias_term_neg)
        W7_mtx_pos = W7_train_pos_bias.to_numpy()
        W7_mtx_neg = W7_train_neg_bias.to_numpy()
        G7_pos = W7_mtx_pos.T@W7_mtx_pos+np.identity(W7_mtx_pos.shape[1])*delta
        G7_neg = W7_mtx_neg.T@W7_mtx_neg+np.identity(W7_mtx_neg.shape[1])*delta
        H7_pos = W7_mtx_pos.T@W7_mtx_pos
        H7_neg = W7_mtx_neg.T@W7_mtx_neg
        b7_pos = GEP(G7_pos,H7_neg)
        b7_neg = GEP(G7_neg,H7_pos)
        b7_pos_list = b7_pos.tolist()
        b7_neg_list = b7_neg.tolist()
        pv7_pos, pv7_neg = b7_pos_list.pop(), b7_neg_list.pop()
        pl7_pos, pl7_neg = b7_pos_list, b7_neg_list
        X7_test_array, Y7_test_list = X7_test.to_numpy(), Y7_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X7_test.shape[0]) :
            dist1 = (abs(np.dot(X7_test_array[j],pl7_pos)+pv7_pos)/norm(pl7_pos))[0]
            dist2 = (abs(np.dot(X7_test_array[j],pl7_neg)+pv7_neg)/norm(pl7_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y7_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Pima Indian dataset ({X7_DF.shape[0]},{X7_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def Australian_GEPSVM():
    Australian=fetch_ucirepo(id=143)
    X8_DF = pd.DataFrame(Australian.data.features)
    Y8_DF = pd.DataFrame(Australian.data.targets)
    Y8_DF = Y8_DF*2-1
    delta = 25
    prec_fix = 1E-10
    skf_df8 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df8.split(X8_DF,Y8_DF)) :
        X8_train,Y8_train = X8_DF[X8_DF.index.isin(train_index)],Y8_DF[Y8_DF.index.isin(train_index)]
        X8_test,Y8_test = X8_DF[X8_DF.index.isin(test_index)],Y8_DF[Y8_DF.index.isin(test_index)]
        W8_train = X8_train.join(Y8_train)
        W8_train_pos = W8_train[W8_train['A15']==1]
        W8_train_neg = W8_train[W8_train['A15']==-1]
        W8_train_pos = W8_train_pos.drop(W8_train_pos.columns[[14]],axis=1)
        W8_train_neg = W8_train_neg.drop(W8_train_neg.columns[[14]],axis=1)
        bias_term_pos = [1]*W8_train_pos.shape[0]
        bias_term_neg = [1]*W8_train_neg.shape[0]
        W8_train_pos_bias = W8_train_pos.assign(bias = bias_term_pos)
        W8_train_neg_bias = W8_train_neg.assign(bias = bias_term_neg)
        W8_mtx_pos = W8_train_pos_bias.to_numpy()
        W8_mtx_neg = W8_train_neg_bias.to_numpy()
        G8_pos = W8_mtx_pos.T@W8_mtx_pos+np.identity(W8_mtx_pos.shape[1])*delta
        G8_neg = W8_mtx_neg.T@W8_mtx_neg+np.identity(W8_mtx_neg.shape[1])*delta
        H8_pos = W8_mtx_pos.T@W8_mtx_pos+np.identity(W8_mtx_pos.shape[1])*prec_fix
        H8_neg = W8_mtx_neg.T@W8_mtx_neg+np.identity(W8_mtx_neg.shape[1])*prec_fix
        b8_pos = GEP(G8_pos,H8_neg)
        b8_neg = GEP(G8_neg,H8_pos)
        b8_pos_list = b8_pos.tolist()
        b8_neg_list = b8_neg.tolist()
        pv8_pos, pv8_neg = b8_pos_list.pop(), b8_neg_list.pop()
        pl8_pos, pl8_neg = b8_pos_list, b8_neg_list
        X8_test_array, Y8_test_list = X8_test.to_numpy(), Y8_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X8_test.shape[0]) :
            dist1 = (abs(np.dot(X8_test_array[j],pl8_pos)+pv8_pos)/norm(pl8_pos))[0]
            dist2 = (abs(np.dot(X8_test_array[j],pl8_neg)+pv8_neg)/norm(pl8_neg))[0]
            if dist1 <= dist2 :
                temp.append(1)
            else:
                temp.append(-1)
        for k in range(len(temp)) :
            if temp[k] == Y8_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"Australian dataset ({X8_DF.shape[0]},{X8_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")

def CMC_GEPSVM():
    CMC=fetch_ucirepo(id=30)
    X9_DF = pd.DataFrame(CMC.data.features)
    Y9_DF = pd.DataFrame(CMC.data.targets)
    delta = 50
    skf_df9 = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    pred_accuracy = []
    for i,(train_index, test_index) in enumerate(skf_df9.split(X9_DF,Y9_DF)) :
        X9_train,Y9_train = X9_DF[X9_DF.index.isin(train_index)],Y9_DF[Y9_DF.index.isin(train_index)]
        X9_test,Y9_test = X9_DF[X9_DF.index.isin(test_index)],Y9_DF[Y9_DF.index.isin(test_index)]
        W9_train = X9_train.join(Y9_train)
        W9_train_target_is_1 = W9_train[W9_train['contraceptive_method']==1]
        W9_train_target_is_2 = W9_train[W9_train['contraceptive_method']==2]
        W9_train_target_is_3 = W9_train[W9_train['contraceptive_method']==3]
        W9_train_target_is_1_pos = W9_train_target_is_1.drop(W9_train_target_is_1.columns[[9]],axis=1)
        W9_train_target_is_2_pos = W9_train_target_is_2.drop(W9_train_target_is_2.columns[[9]],axis=1)
        W9_train_target_is_3_pos = W9_train_target_is_3.drop(W9_train_target_is_3.columns[[9]],axis=1)
        W9_train_target_is_1_neg = pd.concat([W9_train_target_is_2_pos,W9_train_target_is_3_pos])
        W9_train_target_is_2_neg = pd.concat([W9_train_target_is_1_pos,W9_train_target_is_3_pos])
        W9_train_target_is_3_neg = pd.concat([W9_train_target_is_1_pos,W9_train_target_is_2_pos])
        bias_term_target_is_1_pos = [1]*W9_train_target_is_1_pos.shape[0]
        bias_term_target_is_2_pos = [1]*W9_train_target_is_2_pos.shape[0]
        bias_term_target_is_3_pos = [1]*W9_train_target_is_3_pos.shape[0]
        bias_term_target_is_1_neg = [1]*W9_train_target_is_1_neg.shape[0]
        bias_term_target_is_2_neg = [1]*W9_train_target_is_2_neg.shape[0]
        bias_term_target_is_3_neg = [1]*W9_train_target_is_3_neg.shape[0]
        W9_train_target_is_1_pos_bias = W9_train_target_is_1_pos.assign(bias = bias_term_target_is_1_pos)
        W9_train_target_is_2_pos_bias = W9_train_target_is_2_pos.assign(bias = bias_term_target_is_2_pos)
        W9_train_target_is_3_pos_bias = W9_train_target_is_3_pos.assign(bias = bias_term_target_is_3_pos)
        W9_train_target_is_1_neg_bias = W9_train_target_is_1_neg.assign(bias = bias_term_target_is_1_neg)
        W9_train_target_is_2_neg_bias = W9_train_target_is_2_neg.assign(bias = bias_term_target_is_2_neg)
        W9_train_target_is_3_neg_bias = W9_train_target_is_3_neg.assign(bias = bias_term_target_is_3_neg)
        W9_mtx_target_is_1_pos = W9_train_target_is_1_pos_bias.to_numpy()
        W9_mtx_target_is_2_pos = W9_train_target_is_2_pos_bias.to_numpy()
        W9_mtx_target_is_3_pos = W9_train_target_is_3_pos_bias.to_numpy()
        W9_mtx_target_is_1_neg = W9_train_target_is_1_neg_bias.to_numpy()
        W9_mtx_target_is_2_neg = W9_train_target_is_2_neg_bias.to_numpy()
        W9_mtx_target_is_3_neg = W9_train_target_is_3_neg_bias.to_numpy()
        G9_target_is_1 = W9_mtx_target_is_1_pos.T@W9_mtx_target_is_1_pos+np.identity(W9_mtx_target_is_1_pos.shape[1])*delta
        G9_target_is_2 = W9_mtx_target_is_2_pos.T@W9_mtx_target_is_2_pos+np.identity(W9_mtx_target_is_2_pos.shape[1])*delta
        G9_target_is_3 = W9_mtx_target_is_3_pos.T@W9_mtx_target_is_3_pos+np.identity(W9_mtx_target_is_3_pos.shape[1])*delta
        H9_target_is_1 = W9_mtx_target_is_1_neg.T@W9_mtx_target_is_1_neg
        H9_target_is_2 = W9_mtx_target_is_2_neg.T@W9_mtx_target_is_2_neg
        H9_target_is_3 = W9_mtx_target_is_3_neg.T@W9_mtx_target_is_3_neg
        b9_target_is_1 = GEP(G9_target_is_1,H9_target_is_1)
        b9_target_is_2 = GEP(G9_target_is_2,H9_target_is_2)
        b9_target_is_3 = GEP(G9_target_is_3,H9_target_is_3)
        b9_target_is_1_list, b9_target_is_2_list, b9_target_is_3_list = b9_target_is_1.tolist(), b9_target_is_2.tolist(), b9_target_is_3.tolist()
        pv9_target_is_1, pv9_target_is_2, pv9_target_is_3= b9_target_is_1_list.pop(), b9_target_is_2_list.pop(), b9_target_is_3_list.pop()
        pl9_target_is_1, pl9_target_is_2, pl9_target_is_3= b9_target_is_1_list, b9_target_is_2_list, b9_target_is_3_list
        X9_test_array, Y9_test_list = X9_test.to_numpy(), Y9_test.to_numpy().T.tolist()[0]
        temp = []
        count = 0  
        for j in range(X9_test.shape[0]) :
            dist1 = (abs(np.dot(X9_test_array[j],pl9_target_is_1)+pv9_target_is_1)/norm(pl9_target_is_1))[0]
            dist2 = (abs(np.dot(X9_test_array[j],pl9_target_is_2)+pv9_target_is_2)/norm(pl9_target_is_2))[0]
            dist3 = (abs(np.dot(X9_test_array[j],pl9_target_is_3)+pv9_target_is_3)/norm(pl9_target_is_3))[0]
            min_index = 1
            min_val = 1E10
            for s,t in enumerate([dist1,dist2,dist3]) :
                if t < min_val:
                    min_index = s+1
                    min_val = t
            temp.append(min_index)
        for k in range(len(temp)) :
            if temp[k] == Y9_test_list[k] :
                count += 1
        pred_accuracy.append(count/len(temp))
    pred_accuracy_array = np.array(pred_accuracy)
    accuracy_max = np.max(pred_accuracy_array)
    accuracy_min = np.min(pred_accuracy_array)
    accuracy_mean = np.mean(pred_accuracy_array) 
    accuracy_std =  np.std(pred_accuracy_array)
    print(f"CMC dataset ({X9_DF.shape[0]},{X9_DF.shape[1]+1})")
    print()
    print(f"Maximum accuracy = {accuracy_max:.2%}") 
    print(f"Minimum accuracy = {accuracy_min:.2%}")
    print(f"Overall accuracy = {accuracy_mean:.2%}±{accuracy_std:.2%}")


def main():
    print("x"*100)
    Statlog_GEPSVM()
    print("x"*100)
    Heart_c_GEPSVM()
    print("x"*100)
    Hepatitis_GEPSVM()
    print("x"*100)
    Ionosphere_GEPSVM()
    print("x"*100)
    Sonar_GEPSVM()
    print("x"*100)
    Vote_GEPSVM()
    print("x"*100)
    Pima_Indian_GEPSVM()
    print("x"*100)
    Australian_GEPSVM()
    print("x"*100)
    CMC_GEPSVM()
    print("x"*100)

if __name__ == "__main__":
    main()
