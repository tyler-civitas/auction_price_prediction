
_Notes From EDA and Gridding for Tyler/Will on 04/21 Regression Case Study_

## 1 Make a scoring function
Make a function to populate the scoring= parameter in GridSearchCV. Otherwise, you are 'flying blind' about what will win the competition. Make sure the scoring function is identical to the competition function or it will throw you off.
```python
def rmsle_scoring(estimator, X, y):
    predictions = estimator.predict(X)
    return np.sqrt(np.mean((np.log(predictions+1) - np.log(y+1)) ** 2))
```
#### Scikit learn's Best Estimator is broken
The best_estimator_ attribute that you get from GridSearchCV will be **wrong** since it considers the best 'score' to be the highest value. This is why the default R^2 scoring sucks.

Appreciate the statistics... We are comparing 'expensive' and 'cheap' equipment so the mean squared error needs to be used on a log scale. The log scale turns a 'relative' difference into an 'absolute' difference.

## 2 Use Verbose Gridding
Set GridSearchCV parameter/argument "verbose=10"
This will let you see the grid search as it works, and will help you with the insanity of watching a frozen computer. At least you will know what it is doing

## 3 Refine Grid Search
Find best values for one parameter, then try varying other parameters.
You can sort parameters and results using the following:
```python
pd.DataFrame(YOURGRIDSEARCHFUNCTION.cv_results_).sort_values(‘mean_test_score’).ix[:, [‘mean_test_score’, ‘params’]].values
```
Take best parameters from each run, then re-use them and continue grid searching with other un-searched parameters. Even so... it still took thousands of grid searches to finish (Ran grids during the happy hour, and later while watching shows at home)

## 4 USE EDA TO FIGURE OUT THE PROPER FEATURES!!!!
We did alot of EDA to find the proper features but most of them turned out to be too detailed. We ended up with over 2,000 dummy variables.  
 As EDA, histograms, and alot of .count() methods revealed terrible data (such as Machine Hours), we were able to automate the process of choosing which features to use and which to discard. Alex/Wallace's EDA also gave us a good idea about which features were the best to use.  
 **Abstract The Workflow!**   
We abstracted our 'Feature Choice' system. We had a function that let us 'plug and play' various features into the dataset.


## 5 Use ML Algorithms you understand
Adaboost.R2 and XGBoost are supposed to be the best for this one, according to kaggle. They are also supposed to be the most powerful machine learning algorithms, solve any problem, cure world hunger, etc.... but Adaboost.R2 was not even as effective as RandomForest. Apparently you have to actually understand the algorithms and statistics to use them! Crazy!

## 6 Cross Validation might stab you in the back
The machine hours feature helped provide excellent cross validation results, but did not contribute to test data results at all. This is because many machine hour entries were missing. This seemed to create a small y-intercept in the response variable - resulting in many negative price predictions
