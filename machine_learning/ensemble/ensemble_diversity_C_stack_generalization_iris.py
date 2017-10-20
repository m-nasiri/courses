
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import ensemble
import NN

    
# load dataset
# load dataset
iris = scipy.io.loadmat('..\dataset\iris.mat')
X = iris["irisInputs"].T
t = iris["irisTargets"].T

t_sparse =  t
t_array = np.argmax(t_sparse, axis=0)

# dataset parameters
SN , FN = X.shape # (Number Of Features , Number Of Samples)
CN = t.shape[1]


# input variables
test_ratio=0.1              # 0<test_ratio<1
train_ratio=1-test_ratio

iterration = 20
#clfrs = ['onn', 'bayes', 'knn', 'svm_lin','svm_rbf', 'svm_poly', 'decision_tree', 'rbf']
clfrs = ['onn', 'bayes', 'knn', 'svm_lin', 'decision_tree', 'rbf']
combs = ['stack_generalization']
classifier_numbers = len(clfrs) 
combiner_numbers = len(combs) 
classifier_accuracy_array = np.zeros((classifier_numbers, iterration))
combiner_accuracy_array = np.zeros((combiner_numbers, iterration))

for itr in range(iterration):
    # random selection
    Xp, Xtp, tp_sparse, ttp_sparse = train_test_split(X, t_sparse, test_size=test_ratio)
    tp_array = np.argmax(tp_sparse, axis=1)
    ttp_array = np.argmax(ttp_sparse, axis=1)
    TN = ttp_array.shape[0]
    
    # single classifiers
    classifiers_test_prediction, classifiers_train_prediction = ensemble.ClassifierSelection2(Xp, tp_array, tp_sparse, Xtp, clfrs, verbose=True)
    
    
    # train MLP part
    # MLP network parameters
    Xmlp = np.zeros(((SN-TN), classifier_numbers))  # mlp train input
    for c in range(classifier_numbers):
        Xmlp[:,c] = classifiers_train_prediction[c]

    Xtmlp = np.zeros((TN, classifier_numbers))      # mlp test input
    for c in range(classifier_numbers):
        Xtmlp[:,c] = classifiers_test_prediction[c]

    input_neurons = classifier_numbers
    output_neurons = tp_sparse.shape[1]
    hidden_neurons = int(0.7*input_neurons+output_neurons)
    
    N = np.array([input_neurons, hidden_neurons, output_neurons])            
    mu=0.1
    maxEpoch=100
    
    WW, error_array = NN.mlp(Xmlp, tp_sparse, N, mu, maxEpoch, verbose=True)
    
    # test ensemble system
    combiner_prediction = np.zeros((TN,1), dtype=np.int32)
    for smpl in range(TN):
        OO = NN.mlpPredict(Xtmlp[smpl,:], WW)
        combiner_prediction[smpl] = np.argmax(OO.T) 
    
    
    
    classifier_accuracy = np.zeros((classifier_numbers,))
    for c in range(classifier_numbers):
        classifier_accuracy[c] = np.sum(np.abs(classifiers_test_prediction[c] - ttp_array))
        classifier_accuracy[c] = 100 * (TN - classifier_accuracy[c]) /TN
    classifier_accuracy_array[:,itr] = classifier_accuracy
        

    for cm in range(combiner_numbers):
        error = np.sum(np.abs(combiner_prediction[:,cm] - ttp_array))
        combiner_accuracy_array[cm, itr] = 100 * (TN - error) /TN

    print('Test iteration = {}'.format(itr))
    
colormap = plt.cm.Set1
colors = [colormap(i) for i in np.linspace(0, 1, classifier_numbers+combiner_numbers)]
for cl in range(classifier_numbers):
    plt.plot(classifier_accuracy_array[cl], c = colors[cl], label=clfrs[cl], linewidth = 1, linestyle='--')
    plt.legend(bbox_to_anchor=(0.05, 0.5), loc=2, fontsize=15, borderaxespad=0.)

for cm in range(combiner_numbers):
    plt.plot(combiner_accuracy_array[cm], c = colors[cl+cm], marker = 'o', linewidth = 2, label=combs[cm])
    plt.legend(bbox_to_anchor=(0.05, 0.6), loc=2, fontsize=12, borderaxespad=0.)
    
x1 = 0
x2 = iterration
y1 = np.min([np.min(combiner_accuracy_array),np.min(classifier_accuracy_array)])
y1 = np.max([y1*0.8, 0])
y2 = np.max([np.max(combiner_accuracy_array),np.max(classifier_accuracy_array)])
y2 = np.min([y2*1.2, 105])
plt.axis([x1, x2, y1, y2])
plt.grid()






