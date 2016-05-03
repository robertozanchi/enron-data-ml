#Identify Fraud from Enron Email

This is project 5 of Udacity's Data Analyst Nanodegree, connected to the course Intro to Machine Learning.

### Background
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.

### Project summary
#### Aim
This project applies machine learning techniques and algorithms to answer the question: can the people behind the Enron fraud (aka persons of interest, or POI's) be identified on the basis of their emails and financial data?

#### Approach
The approach I used to achieve the stated aim is based on cleaning the dataset, and the selecting
the best features and the best performing algorithm through testing. The following steps were taken:

1. Data exploration
2. Clean data by identifying and removing outliers
3. Select best features for final analysis
4. Try multiple algorithms and select one for final analysis
5. Evaluate the results

Steps 3, 4 and 5 were taken multiple times in an iterative fashion, to arrive at the best possible results.

#### Conclusions
In the final analysis, a Naive Bayes algorithm was used for supervised classification of POI's
and trained on a selection of six features. The results achieved were: 
```
### Naive Bayes on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
GaussianNB()
Accuracy: 0.85464   Precision: 0.48876  Recall: 0.38050 F1: 0.42789 F2: 0.39814
```

Conclusions and implications are discussed in section 6. Conclusions.  

### 1. Data exploration
At this stage, I explored the Enron dataset provided by Udacity.

Key findings were:
- There are entries for 146 people in the dataset;
- For each person in the dataset, there are 21 features;
- The dataset reveals the identies of 35 known POI's;
- 18 people in the dataset are labelled POI.

### 2. Outlier investigation and removal
Here I plotted the values of features "bonus" and "salary" to identify outliers in the financial data.

Before cleaning, the maximum values of the two features were:

```
Maximum bonus value before outlier removal: 97343619.0
Maximum salary value before outlier removal: 26704229.0
```

A look at exhbit enron61702insiderpay.pdf allowed me to identify two entries to be removed:

- "TOTAL": a row containing the total of all rows, the likely outlier;
- "THE TRAVEL AGENCY IN THE PARK": not an outlier, but an entry that is not a person.

After removing these two entries from the dataset, the new maximum values were:

```
Maximum bonus value after outlier removal: 8000000.0
Maximum salary value after outlier removal: 1111258.0
```

### 3. Feature selection and engineering
At this stage, I tried three different approaches to feature selection:

- Intuitive selection of the most important features in the data set;
- Creation (and testing) of a new feature combining three existing features;
- Univariate feature selection with the SelectKBest method in Scikit-learn.

#### Intuitive selection of features
A selection of features for further analysis based on intuition would lead me to
focus on the those that indicate (as shown below in ```features_list```):
- The financial compensation received by the person;
- Communication between this person and POI's.

```
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 
                 'director_fees']
```
A disadvantage of this approach is that  - by itself - it doesn't offer a method for 
shortlisting the features of greater significance. I therefore decide to apply the
SelectKBest method to select the top 5 features from among thosr in ```features_list```.
This process is described below.

#### Univariate feature selection with SelectKBest
To address the limitations of intuitive selection and improve the performance of the
estmators I used SelectKBest, a method available within Scikit-learn that selects the
best features based on univariate statistical tests. SelectKBest works by removing all
but the k highest scoring features, where k is a parameter.

In my model, I set k = 5 in order to retain variety in features but eliminate most (i.e. 17)
of the least significant features. The top 5 selected through SelectKBest are:
```
['exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
```
In addition to ```['poi']```, the above are the features I included in the final feature set.

#### Creation of the new feature ```total_compensation```
##### Rationale for new feature creation
I created ```total_compensation``` to be the sum of three existing financial features:
```['salary', 'bonus', 'total_stock_value']```. I hypothesized that combining these three
values would make a POI - who may be more likely that others to have higher values for
any of these features - more easily identifiable.

##### Testing of new feature
I tested the effect of the new feature ```total_compensation``` on the final algorithm
performance. I found that Naive Bayes classification using the top 5 SelectKBest features
including ```total_compensation``` scored lower in accuracy, precision and recall than
the selection only using existing features. I decided not to include ```total_compensation```
in the final feature set.

```
GaussianNB()

['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'total_compensation', 'salary']
Accuracy: 0.83638   Precision: 0.44972   Recall: 0.28400 

['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
Accuracy: 0.85464   Precision: 0.48876  Recall: 0.38050
```

### 4. Algorithm selection and tuning
#### Tested algorithms
I tested three different algorithms for supervised learning: Naive Bayes (```GaussianNB()```),
Decision Tree (```DecisionTreeClassifier```) and Nearest Neighbors (```KNeighborsClassifier```).
Classifiers using these methods were trained on a subset of data (i.e. training data) and
tested on another subset (i.e. testing data) to measure their accuracy in predicting which
entries (people) may be POI's.

#### Parameter tuning
Parameter tuning is the search for good and robust parameters for an algorithm. Tuning is
important in order to find the best results for a given problem.

In my model, I tested three different setting for the ```min_samples_split``` parameter in the 
Decision Tree algorithm. In addition to the default value of ```min_samples_split = 2```, I tested
```min_samples_split = 5``` and ```min_samples_split = 10```, with the following results in testing:

```
DecisionTreeClassifier(min_samples_split=2)
Accuracy: 0.79593   Precision: 0.27763  Recall: 0.26750

DecisionTreeClassifier(min_samples_split=5)
Accuracy: 0.80350   Precision: 0.27662  Recall: 0.23250

DecisionTreeClassifier(min_samples_split=10)
Accuracy: 0.81721   Precision: 0.33233  Recall: 0.27700
```
Setting ```min_samples_split=10``` on Decision Tree yields the best accuracy, precision and recall
results for that algorithm. However, even with tuning, Decision Tree is not the best performing method.

#### Final algorithm selection
The algorithm selected for final analysis is Naive Bayes, i.e. GaussianNB() in Scikit-learn. The reason
for this selection are the results achieved in the evaluation and validation phase, discussed below.

### 5. Evaluation and validation
Validation is needed to determine how effective the chosen methods are in solving the data problem.
Here we wanted to choose the most effective among three supervised learning algorithms at predicting
which people in the dataset are likely to be POI's, a classification problem.

Evaluation of results was perfomed multiple times in an iterative fashion in order to identify the best
performing of the three selected algorithms and the effect of parameter tuning. This entailed two steps:

1. Split the data into training and testing sets
2. Calculate accuracy, precision and recall scores

#### Split the data into training and testing sets
The basis for supervised learning is the use of part of the data in the dataset to train a algorithm
and then test its effectivenesss on the remaining subset, i.e. training set. The algorthm's predictions
are then compared with the known information about the training set to measure its accuracy.

I used the ```train_test_split``` method in Scikit-learn to split the data into training and test sets,
which I then used for training and testing in all three algorithms.
```
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
```

#### Calculate accuracy, precision and recall scores
I evaluated performance, precision and recall of my algorithms using Udacity's ```tester.py``` script. 
These metrics can be defined as following:

##### Best result: Naive Bayes
The best results came from using the Naive Bayes algorithm. This yielded accuracy of 0.85464, precision
of 0.48876 and recall of 0.38050, all within the acceptable range for passing this project.


```
# Naive Bayes on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
GaussianNB()
Accuracy: 0.85464; Precision: 0.48876; Recall: 0.38050; F1: 0.42789; F2: 0.39814
Total predictions: 14000; True positives: 761; False positives: 796; False negatives: 1239; True negatives: 11204
```

##### Other algorithms: Decision Tree and Nearest Neighbors
The best results achieved with Decision Tree and Nearest Neighbors did not meet the requrements
for this project.
```
# Decision tree on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                       min_samples_split=10, min_weight_fraction_leaf=0.0,
                       presort=False, random_state=None, splitter='best')
Accuracy: 0.81721; Precision: 0.33233; Recall: 0.27700; F1: 0.30215; F2: 0.28654
Total predictions: 14000; True positives: 554; False positives: 1113   False negatives: 1446   True negatives: 10887
```
```
# Nearest neighbors on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income']
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                     weights='uniform')
Accuracy: 0.87657; Precision: 0.68733; Recall: 0.24950; F1: 0.36610; F2: 0.28593
Total predictions: 14000; True positives: 499; False positives: 227; False negatives: 1501; True negatives: 11773
```

### Conclusions


