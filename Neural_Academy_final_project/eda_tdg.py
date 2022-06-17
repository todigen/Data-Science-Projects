# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0

# General functions

def hist_plots(X, hue_target = None, str_palette = None, color = None, figsize = (20, 20), n_bins = 'auto', ticks_fontsize = 12, label_size = 12):
    
    X = X.select_dtypes(include=np.number)
    
    plt.figure(figsize = figsize)
    for i, col in enumerate(X.columns):
        ax = plt.subplot(len(X.columns), 4, i+1)
        sns.histplot(data = X, x = col, hue = hue_target, bins = n_bins, ax=ax, palette = str_palette, color = color)  
        plt.yticks(weight = 'bold', fontsize = ticks_fontsize)
        plt.xticks(weight = 'bold', fontsize = ticks_fontsize)
        plt.ylabel(' ')
        ax.set_xlabel(str(col), fontsize = label_size, weight = 'bold')
    return (plt.show())


def kde_plots(X, hue_target = None, str_palette = None, color = None, figsize = (20, 20), ticks_fontsize = 12, label_size = 12):
    
    X = X.select_dtypes(include=np.number)
    
    plt.figure(figsize = figsize)
    for i, col in enumerate(X.columns):
        ax = plt.subplot(len(X.columns), 4, i+1)
        sns.kdeplot(data = X, x = col, hue = hue_target, ax=ax, palette = str_palette, color = color)  
        plt.yticks(weight = 'bold', fontsize = ticks_fontsize)
        plt.xticks(weight = 'bold', fontsize = ticks_fontsize)
        plt.ylabel(' ')
        ax.set_xlabel(str(col), fontsize = label_size, weight = 'bold')
    return (plt.show())

        
def pair_grid(df_subset, hue_str, str_palette, diag_map, off_diag_map, bool_diag_sharey, bool_corner):
    
    g = sns.PairGrid(df_subset, hue= hue_str, palette = str_palette, diag_sharey=bool_diag_sharey, corner=bool_corner)
    g.map_diag(diag_map)
    g.map_offdiag(off_diag_map)
    g.add_legend()
    return plt.show()
    
    
def dis_plot(df, feature, target, stat, bools):
    plots = sns.displot(df, x = feature, hue = target, stat = stat, kde = bools)
    return plots


def feature_plots1(X, y, str_palette = None, n_bins = 'auto', kde = False):
    
    X_num = X.select_dtypes(include=np.number)
    
    for col in X_num.columns:
        
        fig, axs = plt.subplots(1, 2, figsize=(21,3))
        sns.histplot(data = X_num, x = col, hue = y, bins = n_bins, kde = kde, ax=axs[0], palette = str_palette) #bins = n° di rettangoli
        sns.boxplot(data = X_num, x = col, hue = y, ax=axs[1], palette = str_palette)
        
    return plt.show()

def feature_plots2(X, y, str_palette = None, n_bins = 'auto', kde = False, figsize=(14,3)):
    
    X_num = X.select_dtypes(include=np.number)
    
    for col in X_num.columns:

        mean = X_num[col].mean()
        median = X_num[col].median()
        mode = X_num[col].mode().values[0]
        
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw = {"height_ratios": (0.2, 1)}, figsize = figsize)
        ax3 = fig.add_subplot(111, zorder= -1)
        
        for _, spine in ax3.spines.items():
               spine.set_visible(False)

        ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False)
        ax3.get_shared_x_axes().join(ax3, ax_box)

        sns.boxplot(data= X_num, x = col, ax=ax_box, palette = str_palette)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
#         ax_box.axvline(mode, color='b', linestyle='-')

        sns.histplot(data= X_num, x = col, hue = y, bins = n_bins, ax=ax_hist, kde = kde, palette = str_palette)  
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
#         ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")

        fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax_box.set(xlabel='')
        
    return plt.show()

    
# Summary statistics

def location(X):

    modes = {}
    medians = {}
    means = {}

    for col in X.columns:
        
        mode = st.mode(X[col])
        modes[col] = mode
        
    X_num = X.select_dtypes(include=np.number).dropna()
        
    for col in X_num.columns:
        
        median = round(np.median(X_num[col]), 2)
        mean = round(np.mean(X_num[col]), 2)

        medians[col] = median
        means[col] = mean
            

    statistics = {'([mode][count])' : modes, 'median' : medians, 'mean' : means}
    return pd.DataFrame(statistics).fillna('-')

def plot_correlation_matrix(X, y, bool_mask, colors):
    df_num = pd.concat([y, X], axis=1)
    fig, ax = plt.subplots(figsize=(20,10))
    correlated_matrix = df_num.corr()
    mask = np.zeros_like(correlated_matrix)
    mask[np.triu_indices_from(mask)] = bool_mask
    sns.heatmap(correlated_matrix, mask=mask, annot = True, cmap= colors)
    ax.set_title('Correlation Matrix')

# Nan detecting

def nan_info(data):

    percs = {}
    nan_vs = {}

    for col in data.columns:

        nan_v = data[col].isna().sum()
        perc = round((nan_v/len(data[col]))*100, 2) # % NaN values for column

        nan_vs[col] = nan_v
        percs[col] = perc

    nan_tab = pd.DataFrame({'NaN_count' : nan_vs, '%NaN' : percs}).sort_values(by='%NaN', ascending=False)

    tot_nan = data.isna().sum().sum()
    tot_nonan = data.notna().sum().sum()
    tot_percs = round(tot_nan/(tot_nan+tot_nonan)*100,2)

    print('tot_NaN: {}, ({} %)'.format(tot_nan, tot_percs))

    return nan_tab


# Outlier detection & boxplot parameters

def outliers_by_boxplots(X, columns):
    
    X = X.dropna()
    
    q1 = {}
    q2 = {}
    q3 = {}
    med = {}
    iqr = {}
    upper_bound = {}
    lower_bound = {}
    outl_c = {}
    ret = {}
    
    for col in columns:
    
        q1[col] = np.quantile(X[col], 0.25)
        q3[col] = np.quantile(X[col], 0.75)
        med[col] = np.median(X[col])

        iqr[col] = q3[col] - q1[col]

        upper_bound[col] = q3[col] + (1.5 * iqr[col])
        lower_bound[col] = q1[col] - (1.5 * iqr[col])

        ret[col] = (X[col] > upper_bound[col]) | (X[col] < lower_bound[col]) # bools values: False = inliers, True = outliers
        
        outl_c[col] = np.array(ret[col]).sum() # n° outliers 
        
    tot_outliers = np.array(list(outl_c.values())).sum()
    tot_percs = round(tot_outliers/(X.shape[0]*X.shape[1])*100, 2)
        
    print('tot_outliers: {}, ({} %)'.format(tot_outliers, tot_percs))
    
    return (pd.DataFrame({'25% quantile': q1, '50% quantile': med, '75% quantile': q3, 'IQR': iqr,
                        'upper_bound': upper_bound, 'lower_bound': lower_bound, 'outlier_count': outl_c}))


# Feature selection by correlation matrix

def fs_corr_matrix(X, y, n = 1, redundancy_threshold = 1):
    
    '''
    n (int) number of features to select
    '''
    
    df_num = pd.concat([y, X], axis=1)
    corr_matrix_abs = df_num.corr().abs()

    # upper_tri = upper triangular correlation matrix of features

    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
    upper_tri = upper_tri.drop(y.name, axis=0).drop(y.name, axis=1)

    # to_drop = list of redundant highly correlated features

    to_drop = [col_name for col_name in upper_tri.columns if any(upper_tri[col_name] > redundancy_threshold)]

    corr_matrix_abs.drop(corr_matrix_abs[to_drop], axis=0, inplace=True) 

    correlation_y = corr_matrix_abs[y.name]
    correlation_y_sorted = correlation_y.sort_values(ascending = False)
    fs = correlation_y_sorted[1:(n+1)]  # 1 to skip y itself
    index = fs.index
    
    return X[index]
                 
# Imbalanced data


def imbalanced_data(df, y, str_palette = 'Set2'):
    
    mask0 = (y == 0.0)
    mask1 = (y == 1.0)
    Yes = len(y[mask1]) 
    No = len(y[mask0])
    lab0 = round(No/(Yes+No)*100, 2) # labels
    lab1 = round(Yes/(Yes+No)*100, 2)
    imbd = round(abs(lab0-lab1), 2)

    print('{}% imbalanced'.format(imbd))
    print(y.value_counts())

    sns.catplot(data=df, x = y.name, kind = 'count', palette = str_palette)
    
    return plt.show()


# Column names: remove white spaces and convert to lower case

def columns_strip_lower(X, y):
    
    X.columns = X.columns.str.strip().str.lower()
    y.name = y.name.strip().lower()
    return X, y

