import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform, norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle


class Judge():
    
    def __init__(self):
        pass
    
    def set_data(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def set_algorithms(self, algorithms):
        self.algorithms = algorithms
        return self
    
    def set_metrics(self, metrics):
        self.metrics = metrics
        return self
    
    def set_models(self, models):
        self.models = models
        return self
    
    def set_nested_cv(self, tuning_method, hpars, inner_cv, outer_cv, rscv_random_state = None):
        self.tuning_method = tuning_method
        self.hpars = hpars
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.rscv_random_state = rscv_random_state 
        return self
    
    def __get_performance_from_algorithm(self):
        
        print('Hyper-parameters optimization method: {}'.format(self.tuning_method))
        
        matrix_score = np.array(np.zeros(len(self.models)*len(self.metrics))).reshape(len(self.models),len(self.metrics)) # Matrix score
        
        for j, metric in enumerate(self.metrics):
            
            for i, (k, v) in enumerate(self.algorithms.items()):
                
                if self.tuning_method == 'GridSearchCV':
                    try:
                        clf = GridSearchCV(estimator=v, param_grid=self.hpars[k], cv=self.inner_cv)  # Hyper-params optimization
                    except:
                        clf = v # Default classifier
                        
                elif self.tuning_method == 'RandomizedSearchCV':
                    try:
                        clf = RandomizedSearchCV(estimator=v, param_distributions=self.hpars[k], random_state=self.rscv_random_state) # Hyper-params optimization
                    except:
                        clf = v # Default classifier
                    
                cv_results = round(cross_val_score(clf, self.X, self.y, scoring = metric, cv=self.outer_cv).mean()*100, 2) # Nested-CV scores
                matrix_score[i,j] = cv_results
            
        return matrix_score
    
    def get_table(self):
        
        matrix_score = self.__get_performance_from_algorithm()
        tab = pd.DataFrame(matrix_score, columns = self.metrics).set_axis(self.models).rename_axis('Model')
        return tab
    
    @staticmethod
    def info_class():
        
        info = print("""
        
        class name -> Judge
        
        -methods
        
        set_data -> X = features, y = targets
        set_algorithms -> algorithms (dict)
        set_metrics -> metrics (string list)
        set_models -> models (string list)
        set_nested_cv -> nested cross validation parameters
        get_table -> return DataFrame with evaluated metrics for each model
        
        -class parameters
        
        tuning_method -> 'GridSearchCV' or 'RandomizedSearchCV'
        hpars -> hyperparameters input for the selected tuning_method
        inner_cv -> hyperparameter tuning cross-validation splitting strategy
        outer_cv -> model selection cross-validation splitting strategy
        
        
        """)
        
        return info


# Processing categorical variables

def ohe_dummies(data, column): 
    col_ohe = pd.get_dummies(data[column], prefix = str.lower(column)) # one-hot encoding with get_dummies function
    data.drop([column], axis = 1, inplace = True)
    data = pd.concat([data, col_ohe], axis = 1)
    return data

def get_replace(column, old_list, new_list):
    data = column.replace(old_list, new_list, inplace = True)
    return data

def get_first_letter(column):
    return column.astype(str).str[0]

def get_str_to_num(column): # convert strings to numerical values
    return column.str.extract('(\d+)')
    

# Data Visualization

def dis_plot(df, feature, target, stat):
    plots = sns.displot(df, x = feature, hue = target, stat = stat)
    return plots

def pair_plot(df_subset):
    plots = sns.pairplot(df_subset)
    return plots

# Normalization

def min_max_scaler(column):
    x = column.values.reshape(-1, 1) # returns a numpy array
    mms = MinMaxScaler()
    x_scaled = mms.fit_transform(x)
    return x_scaled

def standard_scaler(column):
    x = column.values.reshape(-1, 1) # returns a numpy array
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    return x_scaled

# Summary statistics

def location(X):
    
    X_num = X.select_dtypes(include=np.number)
    
    modes = {}
    medians = {}
    means = {}
    
    for i in X_num.columns:
    
        mode = st.mode(X_num[i])
        median = round(np.median(X_num[i]), 2)
        mean = round(np.mean(X_num[i]), 2)
    
        modes[i] = mode
        medians[i] = median
        means[i] = mean

    statistics = {'([mode][count])' : modes, 'median' : medians, 'mean' : means}
    return pd.DataFrame(statistics)

def plot_correlation_matrix(X, y, colors):
    df_num = pd.concat([y, X], axis=1)
    fig,ax = plt.subplots(figsize=(20,10))
    correlated_matrix = df_num.corr()
    sns.heatmap(correlated_matrix, annot = True, cmap= colors)
    ax.set_title('Correlation Matrix')

# Handling missing values

def replace_null_with_mean(column):
    mean_value = column.mean()
    return column.fillna(mean_value)

def replace_null_with_median(column):
    median_value = column.median()
    return column.fillna(median_value)

def replace_null_with_invalid_value(data, column_name, value = -1, suffix = "_invalid"):
    data_copy = data.copy()
    is_null = data_copy[column_name].isnull()
    data_copy[column_name] = data_copy[column_name].fillna(value)
    data_copy[column_name + suffix] = np.where(is_null, 1, 0)
    return data_copy

def replace_null_with_model_predict(estimator, data, target_name, feature_names):
    dc = data.copy()
    
    bool_target_not_null = dc[target_name].notnull()
    bool_target_is_null = dc[target_name].isnull()
    
    row_indexes_where_target_notnull = dc.index[np.where(bool_target_not_null)]
    row_indexes_where_target_isnull = dc.index[np.where(bool_target_is_null)]
    
    X_train = dc.loc[row_indexes_where_target_notnull, feature_names]
    y_train = dc.loc[row_indexes_where_target_notnull, target_name]
    X_test = dc.loc[row_indexes_where_target_isnull, feature_names]
    
    estimator.fit(X_train, y_train)
    
    preds = estimator.predict(X_test) 
    
    dc.loc[row_indexes_where_target_isnull, target_name] = preds
    
    return dc[target_name]

# Feature selection

def fs_corr_matrix(X, y, n = 1):
    '''
    n (int) number of features to select
    '''
    df_num = pd.concat([y, X], axis=1)
    correlation_matrix_abs = df_num.corr().abs()
    correlation_y = correlation_matrix_abs[y.name]
    correlation_y_sorted = correlation_y.sort_values(ascending = False)
    fs = correlation_y_sorted[1:(n+1)]  # 1 to skip y itself
    return fs.index
    