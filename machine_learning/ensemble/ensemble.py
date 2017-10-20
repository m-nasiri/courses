
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import rbf


def ClassifierSelection2(X, t, t_sparse, X_test, clf_list, verbose = False):
    
    classifiers_test_output= []
    classifiers_train_output= []

    if ([i for i,x in enumerate(clf_list) if x=='onn']):
        onn = KNeighborsClassifier(n_neighbors = 1)
        onn.fit(X, t)
        prediction_onn_array = onn.predict(X_test)
        classifiers_test_output.append(prediction_onn_array)
        prediction_onn_array = onn.predict(X)
        classifiers_train_output.append(prediction_onn_array)
        
             
    if ([i for i,x in enumerate(clf_list) if x=='knn']):
        knn = KNeighborsClassifier()
        knn.fit(X, t)
        prediction_knn_array = knn.predict(X_test)
        classifiers_test_output.append(prediction_knn_array)
        prediction_knn_array = knn.predict(X)
        classifiers_train_output.append(prediction_knn_array)
        
    if ([i for i,x in enumerate(clf_list) if x=='bayes']):
        bayes = GaussianNB()
        bayes.fit(X, t)
        prediction_bayes_array = bayes.predict(X_test)
        classifiers_test_output.append(prediction_bayes_array)
        prediction_bayes_array = bayes.predict(X)
        classifiers_train_output.append(prediction_bayes_array)
        
    if ([i for i,x in enumerate(clf_list) if x=='svm_lin']):
       lin_svc = svm.LinearSVC()
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_test_output.append(prediction_lin_svc_array)
       prediction_lin_svc_array = lin_svc.predict(X)
       classifiers_train_output.append(prediction_lin_svc_array)
    
    if ([i for i,x in enumerate(clf_list) if x=='svm_rbf']):
       lin_svc = svm.SVC(kernel='rbf')
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_test_output.append(prediction_lin_svc_array)
       prediction_lin_svc_array = lin_svc.predict(X)
       classifiers_train_output.append(prediction_lin_svc_array)
    
    if ([i for i,x in enumerate(clf_list) if x=='svm_poly']):
       lin_svc = svm.SVC(kernel='poly', degree=3)
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_test_output.append(prediction_lin_svc_array)
       prediction_lin_svc_array = lin_svc.predict(X)
       classifiers_train_output.append(prediction_lin_svc_array)
       
    if ([i for i,x in enumerate(clf_list) if x=='decision_tree']):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X, t)
        prediction_decision_tree_array = decision_tree.predict(X_test)
        classifiers_test_output.append(prediction_decision_tree_array)
        prediction_decision_tree_array = decision_tree.predict(X)
        classifiers_train_output.append(prediction_decision_tree_array)
        
    if ([i for i,x in enumerate(clf_list) if x=='rbf']):

        # Clustering        
        K = t_sparse.shape[1]
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        lables = kmeans.labels_
        
        variances = np.zeros((1,K))
        for cent in range(K):
            variances[0, cent] = np.var(X[(lables == cent),:])
        spreads = np.matrix(np.sqrt(variances))
        
        # RBF  
        WW, average_error_array =rbf.RBF(X, t_sparse.T, K, centers, spreads, mu = 0.1, maxEpoch = 200, verbose=verbose)
        
        # Prediction
        class_result = np.zeros((X_test.shape[0],))
        for idx in range(X_test.shape[0]):
            c = rbf.RBFpredict(X_test[idx,:], WW, K, centers, spreads)          
            class_result[idx] = np.argmax(c)   
        classifiers_test_output.append(class_result.astype(np.int64))
        
        # Prediction
        class_result = np.zeros((X.shape[0],))
        for idx in range(X_test.shape[0]):
            c = rbf.RBFpredict(X_test[idx,:], WW, K, centers, spreads)          
            class_result[idx] = np.argmax(c)   
        classifiers_train_output.append(class_result.astype(np.int64))
        
    return classifiers_test_output, classifiers_train_output 
    
    

def ClassifierCombiners(clfr_out, class_number, combiner):
    
    TN = clfr_out[0].shape[0]
    classifier_numbers = len(clfr_out)
    box = np.zeros((classifier_numbers,))
    combiner_out = np.zeros((TN,1), dtype = int)
    
    if (combiner=='averaging'):
        for sample in range(TN):
            for i in range(classifier_numbers):
                box[i] = clfr_out [i][sample]
            combiner_out [sample, 0] = np.round(np.mean(box))
            
    elif (combiner=='majority_vote'):
        for sample in range(TN):
            for i in range(classifier_numbers):
                box[i] = clfr_out [i][sample]
            vote = np.histogram(box, bins=np.arange(class_number+1))
            combiner_out [sample, 0] = np.argmax(vote[0])

    return combiner_out



def ClassifierSelection(X, t, t_sparse, X_test, clf_list, verbose = True):
    
    classifiers_output= []
    
    if ([i for i,x in enumerate(clf_list) if x=='onn']):
        onn = KNeighborsClassifier(n_neighbors = 1)
        onn.fit(X, t)
        prediction_onn_array = onn.predict(X_test)
        classifiers_output.append(prediction_onn_array)
             
    if ([i for i,x in enumerate(clf_list) if x=='knn']):
        knn = KNeighborsClassifier()
        knn.fit(X, t)
        prediction_knn_array = knn.predict(X_test)
        classifiers_output.append(prediction_knn_array)
    
    if ([i for i,x in enumerate(clf_list) if x=='bayes']):
        bayes = GaussianNB()
        bayes.fit(X, t)
        prediction_bayes_array = bayes.predict(X_test)
        classifiers_output.append(prediction_bayes_array)

    if ([i for i,x in enumerate(clf_list) if x=='svm_lin']):
       lin_svc = svm.LinearSVC()
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_output.append(prediction_lin_svc_array)
    
    if ([i for i,x in enumerate(clf_list) if x=='svm_rbf']):
       lin_svc = svm.SVC(kernel='rbf')
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_output.append(prediction_lin_svc_array)
    
    if ([i for i,x in enumerate(clf_list) if x=='svm_poly']):
       lin_svc = svm.SVC(kernel='poly', degree=3)
       lin_svc.fit(X, t)
       prediction_lin_svc_array = lin_svc.predict(X_test)
       classifiers_output.append(prediction_lin_svc_array)
       
    if ([i for i,x in enumerate(clf_list) if x=='decision_tree']):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X, t)
        prediction_decision_tree_array = decision_tree.predict(X_test)
        classifiers_output.append(prediction_decision_tree_array)
        
    if ([i for i,x in enumerate(clf_list) if x=='rbf']):

        # Clustering        
        K = t_sparse.shape[1]
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        lables = kmeans.labels_
        
        variances = np.zeros((1,K))
        for cent in range(K):
            variances[0, cent] = np.var(X[(lables == cent),:])
        spreads = np.matrix(np.sqrt(variances))
        
        # RBF  
        WW, average_error_array =rbf.RBF(X, t_sparse.T, K, centers, spreads, mu = 0.1, maxEpoch = 100, verbose=verbose)
        
        # Prediction
        class_result = np.zeros((X_test.shape[0],))
        for idx in range(X_test.shape[0]):
            c = rbf.RBFpredict(X_test[idx,:], WW, K, centers, spreads)          
            class_result[idx] = np.argmax(c)
            
        classifiers_output.append(class_result.astype(np.int64))
        
    return classifiers_output
    



