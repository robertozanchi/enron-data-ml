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

#### Key findings in data exploration
Key findings were:
- There are entries for 146 people in the dataset;
- For each person in the dataset, there are 21 features;
- 20 out of 21 features have missing values - see table below;
- The dataset reveals the identies of 35 known POI's;
- 18 people in the dataset are labelled POI.

#### Missing values
The table below shows the ratio of missing "NaN" values from the dataset for each feature.
20 features (all except for ```poi```) have missing values to the tune of between 13.7% and 97.3%.

| Features                  | NaN (ratio)     |
| :------------------------ | --------------: |
| poi                       |             0.0 |
| total_stock_value         |          0.1369 |
| total_payments            |          0.1438 |
| email_address             |          0.2397 |
| restricted_stock          |          0.2466 |
| exercised_stock_options   |          0.3014 |
| salary                    |          0.3493 |
| expenses                  |          0.3493 |
| other                     |          0.3630 |
| to_messages               |          0.4110 |
| shared_receipt_with_poi   |          0.4110 |
| from_messages             |          0.4110 |
| from_this_person_to_poi   |          0.4110 |
| from_poi_to_this_person   |          0.4110 |
| bonus                     |          0.4384 |
| long_term_incentive       |          0.5479 | 
| deferred_income           |          0.6644 |
| deferral_payments         |          0.7329 |
| restricted_stock_deferred |          0.8767 |
| director_fees             |          0.8836 |
| loan_advances             |          0.9726 |


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

- First selection of features with least missing values;
- Univariate feature selection with the SelectKBest method in Scikit-learn;
- Creation (and testing) of a new feature combining three existing features;
- I also used feature scaling to use k-Nearest Neighbors effectively.

#### First selection of features with least missing values
A first selection of features for further analysis is made based on how many missing
values the features have. I choose to retain the features that have at most 50% of NaN.
These are shown below in ```features_list```:
```
features_list = ['poi', 'total_stock_value', 'total_payments', 'restricted_stock', 
                 'exercised_stock_options', 'salary', 'expenses', "other", 'to_messages',
                 'shared_receipt_with_poi', 'from_messages', 'from_this_person_to_poi',
                 'from_poi_to_this_person', 'bonus']
```
A limitation of this approach is that  - by itself - it doesn't offer a method for 
finding the number the features that will produce better results. I decide to apply the
SelectKBest method to select the top features from among those in ```features_list```.
This process is described below.

#### Univariate feature selection with SelectKBest
To address the limitations of the first selection and improve the performance of the
estimators I used SelectKBest, a method available within Scikit-learn that selects the
best features based on univariate statistical tests. SelectKBest works by removing all
but the k highest scoring features, where k is a parameter.

To select the top best features, I:

- Ranked all 14 preselected features using ```SelectKBest()```
- Tested accuracy, precision and recall of final model at different levels of k
- Finally chose k = 6, to maximise all scores, as shown in the table below

Performance of GaussianNB() at different k levels, selected with ```SelectKBest()```

| k    | Accuracy  | Precision  | Recall  |
| :--- | --------: | ---------: | ------: |
| 2    |   0.90409 |    0.46055 | 0.32100 |
| 3    |   0.84069 |    0.46889 | 0.26750 |
| 4    |   0.84300 |    0.48581 | 0.35100 |
| 5    |   0.84677 |    0.50312 | 0.32300 |
| 6    |   0.85393 |    0.48327 | 0.32500 |
| 7    |   0.85293 |    0.41227 | 0.24200 |
| 8    |   0.84453 |    0.37309 | 0.24400 |
| 9    |   0.84267 |    0.36425 | 0.24150 |
| 10   |   0.84267 |    0.36446 | 0.24200 |
| 11   |   0.84027 |    0.34109 | 0.21250 |
| 12   |   0.84000 |    0.33793 | 0.20850 |
| 13   |   0.83287 |    0.30455 | 0.19750 |
| 14   |   0.83353 |    0.30781 | 0.19900 |

The top 6 features selected through SelectKBest are:
```
Features selected for final analysis:
['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'restricted_stock']
```

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

#### Feature scaling
I also used feature scaling, required by the Nearest Neighbors algorithm (see next section).

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

##### Accuracy, precision and recall
These metrics can be defined as following:

- Accuracy is the extent to which the set of labels predicted for a sample exactly matches the 
corresponding set of labels in the testing set.
- Precision is the ratio ```tp / (tp + fp)``` where ```tp``` is the number of true positives and ```fp```
the number of false positives.
- Recall is the ratio ```tp / (tp + fn)``` where ```tp``` is the number of true positives and ```fn```
the number of false negatives.

The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
In other words, how exact the prediction is when it reports a POI in this case.

The recall is intuitively the ability of the classifier to find all the positive samples. In order words, how
complete the prediction is in reporting all POIs, in this case.

##### Best result: Naive Bayes
The best results came from using the Naive Bayes algorithm. This yielded accuracy of 0.85464, precision
of 0.48876 and recall of 0.38050, all within the acceptable range for passing this project.
```
# Naive Bayes on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'restricted_stock']
GaussianNB()
Accuracy: 0.85393; Precision: 0.48327; Recall: 0.32500; F1: 0.38864; F2: 0.34778
Total predictions: 14000; True positives: 650; False positives: 695; False negatives: 1350; True negatives: 11305
```

##### Other algorithms: Decision Tree and Nearest Neighbors
The best results achieved with Decision Tree and Nearest Neighbors did not meet the requrements
for this project.
```
# Decision tree on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'restricted_stock']
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

Accuracy: 0.81307; Precision: 0.30462; Recall: 0.24050; F1: 0.26879; F2: 0.25107
Total predictions: 14000; True positives: 481; False positives: 1098; False negatives: 1519; True negatives: 10902
```

```
# Nearest neighbors on ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'restricted_stock']
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

Accuracy: 0.86986; Precision: 0.59933; Recall: 0.26850; F1: 0.37086; F2: 0.30182
Total predictions: 14000; True positives: 537; False positives: 359; False negatives: 1463; True negatives: 11641
```

### Conclusions

- The predictions of who is a POI in the dataset using Naive Bayes had an accuracy of 85.39%. This is the extent to
which predicted POIs and non-POIs match the actual data. In a dataset with a majority of non-POIs, any prediction 
with a majority of non-POIs will achieve good accuracy, so this may not be the most important metric.
- Precision, which scored 48.33% in this dataset with Naive Bayes, is a more important metric. It tells us that when
a person is predicted to be a POI, there's almost a 50% chance they actually will be. A person identified as POI 
with this algorithm would require further investigation to be confirmed to be a POI. This is a direction for future 
research.
- Recall, which scored 38.86% in this dataset with Naive Bayes, indicates that the algorithm only "captures" 39%
of the POIs in the dataset overall. This means that additional tools and methods need to be deployed in order to
identify all the POIs - another possible direction for further research.


