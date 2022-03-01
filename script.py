import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from imblearn import over_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


class Model:
    def __init__(self, datafile = 'training_set.csv'):
        self.train_data = pd.read_csv(datafile)
        self.top_10_list = ['X52', 'X53', 'X7', 'X16', 'X56', 'X57', 'X25', 'X55', 'X21', 'X24']
        self.train_selected_data = self.train_data[self.top_10_list]
        self.train_target = self.train_data['Y']
        self.random_forest = RandomForestClassifier()

    def preprocessing(self):
        self.Robust_Scaler = RobustScaler().fit(self.train_selected_data)
        self.train_top_10_transform = self.Robust_Scaler.transform(self.train_selected_data)
        self.min_max_scaler = MinMaxScaler().fit(self.train_top_10_transform)
        self.train_top_10_scaler = self.min_max_scaler.transform(self.train_top_10_transform)

    def split(self):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.train_top_10_scaler, self.train_target, test_size = 0.2, random_state=42)
        
    def OverSampling(self):
        self.sm = over_sampling.SMOTE(random_state=0)
        self.X_train_smote, self.y_train_smote = self.sm.fit_resample(self.X_train, self.y_train)

    def fit(self):
        self.model = self.random_forest
        self.model.fit(self.X_train_smote, self.y_train_smote)

    def predict(self):
        test_data = pd.read_csv("test_set.csv")
        test_data_selected = test_data[self.top_10_list]
        self.test_data_transform = self.Robust_Scaler.transform(test_data_selected)
        self.test_data_scale = self.min_max_scaler.transform(self.test_data_transform)
        predict_data = self.model.predict(self.test_data_scale)
        test_data['pred_y'] = predict_data 
        test_data.to_csv("Pred.csv", index = False)



if __name__ == "__main__":
    model_instance = Model()
    model_instance.preprocessing()
    model_instance.split()
    model_instance.OverSampling()
    model_instance.fit()
    print("The ROC and AUC score of the model is", roc_auc_score(model_instance.y_valid, model_instance.model.predict_proba(model_instance.X_valid)[:, 1]),"\n")
    print("Classification Report for the Model:-")
    target_names = ['class 0', 'class 1']
    print(classification_report(model_instance.y_valid, model_instance.model.predict(model_instance.X_valid), target_names=target_names))
    print("The confusion Matrix for the model is:-")
    print(confusion_matrix(model_instance.y_valid, model_instance.model.predict(model_instance.X_valid)))
    model_instance.predict()




