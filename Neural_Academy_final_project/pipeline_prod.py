import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import RFE, VarianceThreshold
import pickle
import joblib
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn import FunctionSampler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


class String2UniqueFeature(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, str_leng):
        self.column = column
        self.str_leng = str_leng
        
    def fit(self, X, y):    
        return self
    
    def transform(self, X):
        
        for i in range(self.str_leng):
            X[f'ch{i}'] = X[self.column].str.get(i).apply(ord)
        
        X["unique_characters"] = X[self.column].apply(lambda x: len(set(x)))
        
        X = X.drop(columns = [self.column], axis = 1)
        
        return X


# data_test loading
df_test_raw = pd.read_csv('test.csv')
df_test = df_test_raw.copy()
X_test = df_test.drop(columns = ['id'], axis=1)

# load the model from disk
model = 'may2022_kaggle_comp.pkl'
loaded_model = joblib.load(model)

# predictions
preds_array = np.round(loaded_model.predict_proba(X_test)[:, 1], 2)
preds = pd.DataFrame(preds_array).set_axis(['target'], axis= 'columns') # target probability = 1
submission = pd.concat([df_test_raw.id, preds], axis=1)
submission.to_csv('submission.csv', index=False)
print(submission)