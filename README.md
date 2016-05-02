#Identify Fraud from Enron Email

This is project 5 of Udacity's Data Analyst Nanodegree, connected to the course Intro to Machine Learning.

### Background
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.

### Introduction
### Aim of the project
This project applies machine learning techniques and algorithms to answer the question: can the people behind the Enron fraud (aka persons of interest, or POI's) be identified on the basis of their emails and financial data?

#### Approach


#### Key results


### 1. Data exploration

At this stage, I explored the Enron dataset provided by Udacity.

Key findings were:
- There are entries for 146 people in the dataset;
- For each person in the dataset, there are 21 features;
- The dataset reveals the identies of 35 known POI's;
- 18 people in the dataset are labelled as a POI.

### 2. Outlier Investigation and Removal

Here I plotted the values of features "bonus" and "salary" to identify outliers.

Before cleaning, the maximum values of the two features were:

```Maximum bonus value before outlier removal: 97343619.0```

```Maximum salary value before outlier removal: 26704229.0```

A look at exhbit enron61702insiderpay.pdf allowed me to identify two entries to be removed:

- "TOTAL": a row containing the total of all rows, the likely outlier;
- "THE TRAVEL AGENCY IN THE PARK": a seemingly valid entry that is however not a person.

After removing these two entries from the dataset, the new maximum values were:

```Maximum bonus value after outlier removal: 8000000.0```

```Maximum salary value after outlier removal: 1111258.0```


