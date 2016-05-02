#!/usr/bin/python

import sys
import pickle
import numpy
import pandas

import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Load the dictionary containing the dataset
enron_data = pickle.load(open("final_project_dataset.pkl", "r") )

### Load all "person of interest" (POI) names
all_poi = open("poi_names.txt", "r")

### Print available information for Jeffrey Skilling
print ""
print "Sampling dataset..."
print "Print available information for Jeffrey Skilling"
print data_dict["SKILLING JEFFREY K"]

print ""
print "1. Data Exploration"
print ""

### Number of people in the dataset
people = len(data_dict)
print "There are entries for " + str(people) + " people in the dataset."

### Number of features in the dataset
features = len(data_dict['SKILLING JEFFREY K'])
print "For each person in the dataset, there are " + str(features) + " features."

### Number of POI's in the dataset
def poi_count(data):
    count = 0 
    for person in data:
        if data[person]['poi'] == True:
            count += 1
    print str(count) + " people in the dataset are a person of interest (POI)."

poi_count(data_dict)

### Total number of known POI's
all_poi = open("poi_names.txt", "r")
rfile = all_poi.readlines()
poi = len(rfile[2:])
print "A total of " + str(poi) + " people are known to be a POI."


### Task 2: Remove outliers

print ""
print "2. Outlier Investigation and Removal"
print ""
print "Plotting values for bonus and salary..."

### Detect outliers
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

bonus_max, salary_max = data.max(axis=0)
print ""
print "Maximum bonus value before outlier removal: " + str(bonus_max)
print "Maximum salary value before outlier removal: " + str(salary_max)

### Remove outliers from dataset
data_dict.pop( "TOTAL", 0 )
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )
print ""
print "Removed data entry for Total and The Travel Agency In The Park..."

### Plot data without outliers
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

bonus_max, salary_max = data.max(axis=0)
print ""
print "Maximum bonus value after outlier removal: " + str(bonus_max)
print "Maximum salary value after outlier removal: " + str(salary_max)


### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

print ""
print "3. Feature Engineering and Selection"
print ""
print "Intuitive selection of features:"
print features_list

print ""
print "Create engineered feature:"
print "Create 'total_compensation' as sum of 'salary', 'bonus' and 'total_stock_value'..."

### Create new feature: total_compensation
combine_feature_list = ['salary', 'bonus', 'total_stock_value']
for record in my_dataset:
    person = data_dict[record]
    is_NaN = False
    for field in combine_feature_list:
        if person[field] == 'NaN':
            is_NaN = True
    if is_NaN:
        person['total_compensation'] = 'NaN'
    else:
        person['total_compensation'] = sum([person[feature] for feature in combine_feature_list])
### Uncomment below to add 'total_compensaton' to features_list
# features_list = features_list + ['total_compensation']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### K-best features
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)

print ""
print "'K-best features' selection:"
print results_list

### Select top 5 K-best features
print ""
print "Selecting top 5 K-best-scoring features..."
top_5_features = []
count = 1
for result in results_list:
	if count <= 5:
		top_5_features.append(result[1])
	count = count + 1
print top_5_features

features_list = ["poi"] + top_5_features
print ""
print "Features selected for final analysis:"
print features_list

### Extract features and labels for top 5 features
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# 1. Naive Bayes
### import the sklearn module for GaussianNB and create classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)
### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)
### calculate and return the accuracy on the test data
accuracy = accuracy_score(labels_test, pred)
print ""
print "Accuracy of Naive Bayes classifier: ", accuracy

# 2. Decision Tree
### import the sklearn module for DecisionTreeClassifier and create classifier
# from sklearn.tree import DecisionTreeClassifier
# # clf = DecisionTreeClassifier()
# # clf = DecisionTreeClassifier(min_samples_split=2)
# # clf = DecisionTreeClassifier(min_samples_split=5)
# clf = DecisionTreeClassifier(min_samples_split=10)
# ### fit the classifier on the training features and labels
# clf.fit(features_train,labels_train)
# ### use the trained classifier to predict labels for the test features
# pred = clf.predict(features_test)
# ### calculate and return the accuracy on the test data
# accuracy = accuracy_score(pred, labels_test)
# print 'Accuracy of Decision Tree classifier: ', accuracy

# 3. Nearest Neighbors 
# ### import the sklearn module for KNeighborsClassifier and create classifier
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# ### fit the classifier on the training features and labels
# clf.fit(features_train, labels_train)
# ### use the trained classifier to predict labels for the test features
# pred = clf.predict(features_test)
# ### calculate and return the accuracy on the test data
# accuracy = accuracy_score(pred, labels_test)
# print ""
# print "Accuracy of Nearest Neighbors classifier: ", accuracy


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# 1. Naive Bayes with features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
# GaussianNB()
#     Accuracy: 0.85464   Precision: 0.48876  Recall: 0.38050 F1: 0.42789 F2: 0.39814
#     Total predictions: 14000    True positives:  761    False positives:  796   False negatives: 1239   True negatives: 11204

# 2. Naive Bayes with features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'total_compensation', 'salary']
# GaussianNB()
#     Accuracy: 0.83638   Precision: 0.44972  Recall: 0.28400 F1: 0.34815 F2: 0.30660
#     Total predictions: 13000    True positives:  568    False positives:  695   False negatives: 1432True negatives: 10305

# 3. Decision tree with no tuning and features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
#     Accuracy: 0.79593   Precision: 0.27763  Recall: 0.26750 F1: 0.27247 F2: 0.26947
#     Total predictions: 14000    True positives:  535    False positives: 1392   False negatives: 1465   True negatives: 10608

# 4. Decision tree with min_samples_split=5 and features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=5, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
#     Accuracy: 0.80350   Precision: 0.27662  Recall: 0.23250 F1: 0.25265 F2: 0.24016
#     Total predictions: 14000    True positives:  465    False positives: 1216   False negatives: 1535   True negatives: 10784

# 5. Decision tree with min_samples_split=10 and features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=10, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
#     Accuracy: 0.81721   Precision: 0.33233  Recall: 0.27700 F1: 0.30215 F2: 0.28654
#     Total predictions: 14000    True positives:  554    False positives: 1113   False negatives: 1446   True negatives: 10887

# 6. Nearest neighbors with mn_neighbors=5 and features ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
#     Accuracy: 0.87657   Precision: 0.68733  Recall: 0.24950 F1: 0.36610 F2: 0.28593
#     Total predictions: 14000    True positives:  499    False positives:  227   False negatives: 1501True negatives: 11773


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)