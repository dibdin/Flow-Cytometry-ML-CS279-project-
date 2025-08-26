from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from train_main import train_main

	
names_classifiers = []
names_classifiers.append(('RandomForest', RandomForestClassifier()))
names_classifiers.append(('NaiveBayes', GaussianNB()))
names_classifiers.append(('GradientBoosting', GradientBoostingClassifier()))
names_classifiers.append(('KNN', KNeighborsClassifier()))
names_classifiers.append(('SVC', SVC()))
names_classifiers.append(('AdaBoost', AdaBoostClassifier()))

#Preprocessing types: '', 'low_var', 'norm', 'remove_zero_features', 'top_10_features', 'pca', 'lda', 'autoencoder'
#Classification types: 'all', 'lymphocytes'
#balanced: 'balanced, 'unbalanced'

classification_type = 'all'
balanced = 'balanced'
preprocessing_method = 'autoencoder'
threshold = None

train_main(classification_type, balanced, preprocessing_method, threshold, names_classifiers)
