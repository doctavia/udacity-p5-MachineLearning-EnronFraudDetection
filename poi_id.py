#!/usr/bin/python

import sys
import pickle
sys.path.append("C:/Users/Dewi Octavia/Documents/machinelearning_tools/")
import warnings
warnings.filterwarnings("ignore")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier, load_classifier_and_data
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import os
import texttable
from pprint import pprint
from collections import OrderedDict
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

separator = '=============================================================================================='

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Features List
financial_features = ['salary',
                      'bonus', 
                      'long_term_incentive', 
                      'deferred_income', 
                      'deferral_payments',
                      'loan_advances',
                      'other',
                      'expenses',
                      'director_fees',
                      'total_payments', 
                      'exercised_stock_options', 
                      'restricted_stock',
                      'restricted_stock_deferred', 
                      'total_stock_value'] 
                     
email_features = ['to_messages',  
                  'from_poi_to_this_person', 
                  'from_messages', 
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi',
                  'email_address']

poi = ['poi']

features_list = poi + financial_features + email_features



### Load the dictionary containing the dataset
#os.chdir('C:/Users/Dewi Octavia/Documents/machinelearning_tools/')

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Total number of data points
print "Total number of data points: ", len(data_dict)

# Allocation of POI/non-POI
count_poi = 0
count_nonpoi = 0
poi_names = []

for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        count_poi += 1
        poi_names.append(name)
    else:
        count_nonpoi += 1 
        
print "Allocation of POI/non-POI :", count_poi, '/', count_nonpoi

# Features with missing values
t_nan = texttable.Texttable()
t_nan.set_cols_width([20, 15, 22, 25])
t_nan.add_row(['FEATURES', 'NUMBERS OF NaN', 'NUMBERS OF NaN IN POI', 'POI NAMES'])

for feature in features_list:
    if feature != 'poi':
        count_nan = 0
        count_nan_poi = 0
        feature_poinames = []
        ffeature_poinames = []
        not_in_poi = []
        
        for name in data_dict.keys():
            if data_dict[name][feature] == 'NaN':
                count_nan += 1 

                if data_dict[name]['poi'] == True:
                    count_nan_poi += 1
                    feature_poinames.append(name)                   
                    
                    if count_nan_poi <= 10:
                        ffeature_poinames = feature_poinames
                    elif count_nan_poi == 18:
                        ffeature_poinames = 'All POIs'
                    elif count_nan_poi > 10 & count_nan_poi <18:
                        not_in_poi = list(set(poi_names) - set(feature_poinames))
                        ffeature_poinames = 'All NaN except ' + str(len(not_in_poi)) + ': ' + ', '.join(not_in_poi)
                    
        
        t_nan.add_row([feature, count_nan, count_nan_poi, str(ffeature_poinames)]) 

print "\n Summary of Features with Missing Data"
print t_nan.draw()

# Remove 'email_address' from features_list before coonverted into numpy in featureFormat.py
del features_list[-1]


### Task 2: Remove outliers
print "\n Data Visualization for Outliers"
data = featureFormat( data_dict, features_list )

for point in data:
    salary = point[1]
    bonus = point[2]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

print "Outlier found and removed:"
max_salary = max(data[:, 1])

outlier_index = np.where(data[:,1] == max_salary)

print 'Name: ', data_dict.keys()[int(outlier_index[0])]
print 'Salary: ', data_dict[data_dict.keys()[int(outlier_index[0])]]['salary']

# remove data point 'TOTAL'
data_dict.pop("TOTAL",None)

data = featureFormat( data_dict, features_list, sort_keys = False )

print "There is now", len(data_dict), "data points left in dataset."


print "\n\n Audit Financial Features"

def audit_finance_data(data_dict):
    data = featureFormat( data_dict, features_list, sort_keys = False )
    payment_count = 0
    stock_count = 0

    for i in range(len(data_dict)):
        if sum(data[i][1:10]) != data[i][10]:
            payment_count += 1
            print "\n Unbalanced payment account:"
            print payment_count, " - index = ", i, data_dict.keys()[i]
            print "calc total payment = ", sum(data[i][1:10])
            print "total payments = ", data[i][10]
            pprint , data_dict.values()[i]
            print "data : ", data[i][1:11]

    for i in range(len(data_dict)):        
        if sum(data[i][11:14]) != data[i][14]:
            stock_count += 1 
            print "\n Unbalanced stock account: "
            print stock_count, " - index = ", i, data_dict.keys()[i]
            print "calc total stock = ", sum(data[i][11:13])
            print "total stock = ", data[i][14]

audit_finance_data(data_dict)

print "\n Found 2 unbalanced accounts, they are corrected and reaudited \n"
# Correcting BELFER ROBERT and BHATNAGAR SANJAY's data in data_dict

# BELFER ROBERT's data
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'


# BHATNAGAR SANJAY's data 
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

audit_finance_data(data_dict)


### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == 'NaN' or all_messages == 'NaN':
        fraction = 0
    else:
        fraction = float(poi_messages)/float(all_messages)

    return fraction


submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    
    
#for name in submit_dict.keys():
#    if data_dict[name]["poi"] == True:
#        plt.scatter(submit_dict[name]['from_poi_to_this_person'], submit_dict[name]['from_this_person_to_poi'], color = "red")
#    else:
#        plt.scatter(submit_dict[name]['from_poi_to_this_person'], submit_dict[name]['from_this_person_to_poi'], color = "blue")
#        plt.xlabel("fraction of emails this person gets from POI's")
#        plt.ylabel("fraction of emails this person sends to POI's")
            
        
# Add new features into features_list
features_list = features_list  + ['fraction_from_poi', 'fraction_to_poi']
print "features_list: ", features_list


### Store to my_dataset for easy export below.
my_dataset = data_dict

# Convert my_dataset values to absolute values (for SelectKBest)
for name in my_dataset:
    for feature in ['deferred_income', 'restricted_stock_deferred'] :
        if my_dataset[name][feature] != 'NaN':
            my_dataset[name][feature] = abs(int(my_dataset[name][feature]))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### Task 4: Try a variety of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.


# Uncomment algorithm for testing
classifiers_list = [#('NaiveBayes', GaussianNB()),
                    #('LogisticRegression', LogisticRegression()),
                    ('DecisionTree', DecisionTreeClassifier()),
                    #('KNN', KNeighborsClassifier()),
                    #('AdaBoost', AdaBoostClassifier()),
                    #('RandomForest', RandomForestClassifier())
                    ]

classifiers_list = OrderedDict(classifiers_list)

# Scores Function for Evaluation
def test_score(clf, dataset, feature_list):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    for train_indices, test_indices in cv:
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 1 and truth == 1:
                true_pos += 1
            elif prediction == 0 and truth == 0:
                true_neg += 1
            elif prediction == 0 and truth == 1:
                false_neg += 1
            elif prediction == 1 and truth == 0:
                false_pos += 1

    if true_pos == 0:
        return (0, 0, 0, accuracy)
    else:
        total_predictions = float(true_pos) + float(true_neg) + float(false_neg) + float(false_pos)
        accuracy = (float(true_pos) + float(true_neg))/total_predictions
        precision = round(float(true_pos)/(float(true_pos) + float(false_pos)),5)
        recall = round(float(true_pos)/(float(true_pos) + float(false_neg)), 5)
        f1 = round(2*float(true_pos)/(2*float(true_pos) + float(false_pos)+ float(false_neg)),5)
        
        return (accuracy, precision, recall, f1)

    
# Make a list to store classifiers' results
results_final = []
    
# Initiate table to compile test scores
t = texttable.Texttable()
t.set_cols_width([25, 15, 15, 15, 15])
t.add_row(['CLASSIFIERS', 'ACCURACY', 'PRECISION', 'RECALL', 'F1'])

score_list = []


for i, item in enumerate(classifiers_list):
    t0 = time()
    print "\n\n", i, item
    estimator = classifiers_list.values()[i]
    
    # Apply feature scaling on LR and KNN
    if item in ['LogisticRegression', 'KNN']:
        data = MinMaxScaler().fit_transform(data)
    
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test =  train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42, 
        stratify=labels
        )

    # Make an StratifiedShuffleSplit iterator for cross-validation in GridSearchCV
      
    sss = StratifiedShuffleSplit(
        labels,
        n_iter = 20,
        test_size = 0.5,
        random_state = 42
        )
    
    pipeline = make_pipeline(
        SelectKBest(),
        estimator)
    
    pipeline_steps = []
    
    for i in range(len(pipeline.steps)):
        pipeline_steps.append(pipeline.steps[i][0])
        
    all_params = OrderedDict([
            ('selectkbest__k', [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            ('selectkbest__score_func', [f_classif, chi2]),
            ('decisiontreeclassifier__min_samples_split', [2,5,10,15,25,50]),
            ('decisiontreeclassifier__criterion', ('gini', 'entropy')),
            ('decisiontreeclassifier__random_state', [42]),
            ('kneighborsclassifier__n_neighbors', [3,5,7,9]),
            ('kneighborsclassifier__weights', ('uniform', 'distance')),
            ('kneighborsclassifier__algorithm', ('ball_tree', 'kd_tree', 'brute', 'auto')),
            ('kneighborsclassifier__p', [1,2]),
            ('adaboostclassifier__base_estimator', [DecisionTreeClassifier()]),
            ('adaboostclassifier__n_estimators', [5, 10, 50, 100, 150]),
            ('adaboostclassifier__algorithm', ('SAMME', 'SAMME.R')),
            ('adaboostclassifier__learning_rate', [0.001, 0.1, 0.5, 1, 1.5, 2]),
            ('adaboostclassifier__random_state', [42]),
            ('randomforestclassifier__n_estimators', [2, 3, 5, 8]),
            ('randomforestclassifier__criterion', ('gini', 'entropy')), 
            ('randomforestclassifier__min_samples_split', [2,5,10,15,25,50,80,100]),
            ('logisticregression__penalty', ('l1', 'l2')),
            ('logisticregression__C', [0.001, 0.1, 1, 10, 100]), #
            ('logisticregression__fit_intercept', [True]),
            ('logisticregression__class_weight', ['balanced']),
            ('logisticregression__intercept_scaling', [0.5, 1.0, 1.5]), #
            ('logisticregression__n_jobs', [-1])   
        ])
      
# Dictionary of tuning parameter is built based on classifier    
    grid_params = {}
    
    for step in pipeline_steps:
        for parameter in all_params:
            step_in_parameter = parameter.split('__')[0]
            
            if step_in_parameter == step:
                grid_params[parameter] = all_params[parameter]
                
                
    print "\n grid_params: ", grid_params
    
    clf = GridSearchCV(pipeline, 
                       grid_params, 
                       verbose = 0, 
                       cv = sss,
                       scoring = 'f1'                       
                      )
    
    clf.fit(features, labels)
       
    
    best_estimator = clf.best_estimator_
    print "\n Best estimator = ", best_estimator
    
    
    
    # Find features_selected
    features_k = clf.best_params_['selectkbest__k']
    score_func = clf.best_params_['selectkbest__score_func']

    SKB_k = SelectKBest(score_func = score_func, k = features_k)
    SKB_k.fit_transform(features_train, labels_train)

    
    # Features selected from SelectKBest
    features_selected=[features_list[1:][i]for i in SKB_k.get_support(indices=True)]
    print "\n Selected features are: ", features_selected, "\n"
    
    
    # SelectKBest Score Summary
    print "\n Score Summary:"
    KBest_scores =  [(feature, round(score,3), round(p,3)) 
                     for feature, score, p in zip(features_list[1:], SKB_k.scores_, SKB_k.pvalues_)]
    KBest_scores = pd.DataFrame.from_records(KBest_scores, index = "feature",
                               columns=['feature', 'score', 'p-value'])
    
    KBest_scores.sort_values(by = 'score', ascending = False, inplace = True)
    print KBest_scores, "\n\n"
    
    
    # compile my_features_list
    my_features_list = ['poi'] + features_selected
    
    
    # Assign best_estimator from SelectKBest to clf and have it tested
    clf = clf.best_estimator_   
    
    # Feature Importance for Decision Tree Classifier
    if item == 'DecisionTree':
        importances = clf.named_steps['decisiontreeclassifier'].feature_importances_
        importances_sort = sorted(importances, reverse = True) 
        importances_table = [(feature, round(importance,3)) 
                             for feature, importance in zip(KBest_scores.index.values[:(len(importances))], importances_sort)]

        importances_table = pd.DataFrame.from_records(importances_table, index='Feature',
                                                     columns=['Feature', 'Importance Score'])

        importances_table.sort_values(by='Importance Score', ascending=False, inplace =True)
        
        
        
        print "\n\n Decision Tree Feature Importances:"
        print importances_table, "\n"
    
    # Test Scores
    score= test_score(clf, my_dataset, my_features_list)
    score_list.append((item, score[0], score[1], score[2], score[3]))
    score_list = sorted(score_list, reverse = True, key=lambda x: x[3])
    
    
    # Store result of each classififer in results_final
    results_final.append((item, grid_params, clf, features_selected, KBest_scores, score))
    
    
    print "Time : ", round(time()-t0, 3), "s"
    print separator
    
print "\n\n Classifiers Test Score Summary"
print "=============================="
t.add_rows(score_list, header = False)
print t.draw()

# Save algorithms test scores table
algoTestScore_final = t

#Test clf against testing script
test_classifier(clf, my_dataset, my_features_list)
 
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)
