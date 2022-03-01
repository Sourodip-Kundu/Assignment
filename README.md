# Assignment

## Step 1- EDA
1.Checking data types of the feature and contian NAN values or not

2. Checking Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution

3. Checking for multicolinearityCancel changes

4. Checking the ditribution of the Features. 

5. Cheking if the features have any outlier or not.

6. Checking the frequency of the target varibale. If it's a imbalanced calssification or not.

## Step 2- Preprocessing
1. Remove the highly correleated features.
2. Remove the outlier from the features
3. Done scaling to the features
4. Applied Oversampling technique to solve the imbalanced class problem.

## Step 3- Spiltiing, Modeling and Feature Importance
1. Split the the dataset with ratio of 4:1
2. Fit the Random Forest Model
3. Make a list of top 10 important feature

## Step 4- Final Modelling
1. Select top 10 most important from the train_dataset
2. Done Preprocessing on the selected dataset
3. Split the dataset with 4:1 ratio
4. Train the train-data using Random Forest

## Step 5- Evaluation
Model evaluation has been done by some of the most important metics-
1. ROC_AUC Score
2. Confusion Matrix
3.Precision
4. Recall
5. F1-Score

## Step6- Hyperparameter Optimization
1. Tried different paramater using Grid Search to check if it's possible to increase the model performance or not.

