
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import ensemble

    
# load dataset
SAMPLE_PER_CLASS=200
samples_class_pos=np.random.multivariate_normal([4, 0],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
samples_class_neg=np.random.multivariate_normal([0, 4],[[2, 0],[0, 2]],SAMPLE_PER_CLASS)
X = np.concatenate((samples_class_pos, samples_class_neg))
t = np.zeros((2*SAMPLE_PER_CLASS, 2), dtype = int)
t[0               :  SAMPLE_PER_CLASS, 0] = 1
t[SAMPLE_PER_CLASS:2*SAMPLE_PER_CLASS, 1] = 1
t_sparse =  t
t_array = np.argmax(t_sparse, axis=0)

# dataset parameters
SN , FN = X.shape # (Number Of Features , Number Of Samples)
CN = t.shape[1]


# input variables
test_ratio=0.2              # 0<test_ratio<1
train_ratio=1-test_ratio

iterration = 20
clfrs = ['onn', 'bayes', 'knn', 'svm_lin','svm_rbf', 'svm_poly', 'decision_tree','rbf']
combs = ['majority_vote', 'averaging']
classifier_numbers = len(clfrs) 
combiner_numbers = len(combs) 
classifier_error_array = np.zeros((classifier_numbers, iterration))
combiner_error_array = np.zeros((combiner_numbers, iterration))

for itr in range(iterration):
    # random selection
    Xp, Xtp, tp_sparse, ttp_sparse = train_test_split(X, t_sparse, test_size=test_ratio)
    tp_array = np.argmax(tp_sparse, axis=1)
    ttp_array = np.argmax(ttp_sparse, axis=1)
    TN = ttp_array.shape[0]
    
    # single classifiers
    classifiers_prediction = ensemble.ClassifierSelection(Xp, tp_array, tp_sparse, Xtp, clfrs, verbose=True)
    
    # combine classifiers
    combiner_prediction = np.zeros((TN, combiner_numbers))
    for cm in range(combiner_numbers):
            cp = ensemble.ClassifierCombiners(classifiers_prediction, class_number=CN, combiner = combs[cm])
            combiner_prediction[:,cm] = cp[:,0]

    #print(np.mean(classifiers_prediction, axis=0))
    classifier_error = np.zeros((classifier_numbers,))
    for c in range(classifier_numbers):
        classifier_error[c] = np.sum(np.abs(classifiers_prediction[c] - ttp_array))
        classifier_error[c] = 100 * (TN - classifier_error[c]) /TN
    classifier_error_array[:,itr] = classifier_error
        

    for cm in range(combiner_numbers):
        error = np.sum(np.abs(combiner_prediction[:,cm] - ttp_array))
        combiner_error_array[cm, itr] = 100 * (TN - error) /TN

    print('iteration = {}'.format(itr))
    
colormap = plt.cm.Set1
colors = [colormap(i) for i in np.linspace(0, 1, classifier_numbers+combiner_numbers)]
for cl in range(classifier_numbers):
    plt.plot(classifier_error_array[cl], c = colors[cl], label=clfrs[cl], linewidth = 1, linestyle='--')
    plt.legend(bbox_to_anchor=(0.05, 0.5), loc=2, fontsize=15, borderaxespad=0.)

for cm in range(combiner_numbers):
    plt.plot(combiner_error_array[cm], c = colors[cl+cm], marker = 'o', linewidth = 2, label=combs[cm])
    plt.legend(bbox_to_anchor=(0.05, 0.6), loc=2, fontsize=12, borderaxespad=0.)
    
x1 = 0
x2 = iterration
y1 = np.min([np.min(combiner_error_array),np.min(classifier_error_array)])
y1 = np.max([y1*0.8, 0])
y2 = np.max([np.max(combiner_error_array),np.max(classifier_error_array)])
y2 = np.min([y2*1.2, 105])
plt.axis([x1, x2, y1, y2])
plt.grid()






