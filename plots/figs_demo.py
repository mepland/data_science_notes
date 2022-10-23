# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

# TODO 
import sys,os
sys.path.append(os.path.expanduser('~/dtreeviz'))

# TODO remove
sys.path.append(os.path.expanduser('~/imodels'))

########################################################
# python
import time
import pandas as pd
import numpy as np
import scipy.stats
norm = scipy.stats.norm
import bisect

########################################################
# figs, xgboost, sklearn
import imodels
from imodels import FIGSClassifier

import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve

# TODO
# from sklearn.tree import plot_tree, DecisionTreeClassifier
# from sklearn import metrics

########################################################
# dtreeviz
# must follow the package README to properly install all dependencies!

from dtreeviz import trees
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

########################################################
# skompiler
from skompiler import skompile

########################################################
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore', message='Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.')

########################################################
# set global rnd_seed for reproducibility
rnd_seed = 42
np.random.seed(rnd_seed)

# %%
from plotting import * # load plotting code

# %%
inline=True # plot inline or to pdf
output = './output' # output dir
os.makedirs(output, exist_ok=True)

# %% [markdown] tags=[]
# ***
# # Generate Random Data
# Include additive structure that FIGS does well on

# %%
mc_params_all = {'n_samples': int(1e5), 'n_classes': 2, 'shuffle': False, 'shift': 0.0, 'scale': 1.0, 'hypercube': True}

mc_params = [
    {'n_features': 20, 'n_informative': 6, 'n_redundant': 4, 'n_repeated': 0,
     'n_clusters_per_class': 5, 'weights': [0.5], 'flip_y': 0.05, 'class_sep': 0.9},
    {'n_features': 10, 'n_informative': 4, 'n_redundant': 2, 'n_repeated': 0,
     'n_clusters_per_class': 2, 'weights': [0.7], 'flip_y': 0.1, 'class_sep': 0.9},
    {'n_features': 5, 'n_informative': 2, 'n_redundant': 2, 'n_repeated': 0,
     'n_clusters_per_class': 2, 'weights': [0.6], 'flip_y': 0.04, 'class_sep': 0.9},
]

X = None
y = None
feat_names = []

for i_mc_param, mc_param in enumerate(mc_params):
    param = {**mc_params_all, **mc_param, 'random_state': rnd_seed+i_mc_param}
    X_i, y_i = make_classification(**param)
    if X is None:
        X = X_i
    else:
        X = np.concatenate([X, X_i], axis=1)
    if y is None:
        y = y_i
    else:
        y = np.logical_and(y, y_i).astype(int)
    feat_names += [f'x_{i_mc_param}_{_}' for _ in range(X_i.shape[1])]
    del X_i; del y_i;

# %% [markdown]
# Make Train, Validation, and Holdout Sets

# %%
X_trainVal, X_holdout, y_trainVal, y_holdout = train_test_split(X, y, test_size=0.15, random_state=rnd_seed, stratify=y)
del X; del y;

X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2, random_state=rnd_seed, stratify=y_trainVal)
# del X_trainVal; del y_trainVal;

# %% [markdown] tags=[]
# ***
# # FIGS
# Note we are not using early stopping with FIGS so use `X_trainVal` during training to take advantage of all rows.

# %%
model_figs = FIGSClassifier(max_rules=20)

# %%
time_figs_start = time.time()
model_figs.fit(X_trainVal, y_trainVal, feature_names=feat_names);
time_figs_end = time.time()
print(f'FIGS ran in {time_figs_end-time_figs_start:.0f} seconds')


# %%
def count_splits_figs(model):
    splits = []
    for tree_ in model.trees_:
        node_counter = iter(range(1, int(1e06)))
        def _count_node(node):
            if node.left is None:
                return
            node_id=next(node_counter)
            _count_node(node.left)
            _count_node(node.right)

        _count_node(tree_)
        splits.append(next(node_counter)-1)
    return sum(splits)

n_splits_figs = count_splits_figs(model_figs)

# %%
print(f'FIGS used {len(model_figs.trees_)} trees and {n_splits_figs} splits')

# %%
print(model_figs)

# %%
print(model_figs.print_tree(X_train, y_train))

# %%
model_figs.plot(fig_size=7)

# %% [markdown] tags=[]
# ***
# # XGBoost

# %%
params_default = {'max_depth': 6, 'learning_rate': 0.3, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0}

# %%
fixed_setup_params = {
    'max_num_boost_rounds': 500, # maximum number of boosting rounds to run / trees to create
    'xgb_objective': 'binary:logistic', # objective function for binary classification
    'xgb_verbosity': 0, #  The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    'xgb_n_jobs': -1, # Number of parallel threads used to run XGBoost. -1 makes use of all cores in your system
    'eval_metric': 'auc', # evaluation metric for early stopping
    'early_stopping_rounds': 10, # must see improvement over last num_early_stopping_rounds or will halt
}

# %%
fixed_fit_params = {
    'eval_set': [(X_val, y_val)], # data sets to use for early stopping evaluation
    'verbose': False, # even more verbosity control
}

# %%
model_xgboost = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                                  objective=fixed_setup_params['xgb_objective'],
                                  verbosity=fixed_setup_params['xgb_verbosity'],
                                  eval_metric=fixed_setup_params['eval_metric'],
                                  early_stopping_rounds=fixed_setup_params['early_stopping_rounds'],
                                  random_state=rnd_seed+3, **params_default, use_label_encoder=False)

# %%
time_xgboost_start = time.time()
model_xgboost.fit(X_train, y_train, **fixed_fit_params);
time_xgboost_end = time.time()
print(f'XGBoost ran in {time_xgboost_end-time_xgboost_start:.0f} seconds')

# %%
n_splits_xgboost = sum([tree.count('"split"') for tree in model_xgboost.get_booster().get_dump(dump_format='json')[0:model_xgboost.best_ntree_limit]])

# %%
print(f'XGBoost used {model_xgboost.best_ntree_limit} trees and {n_splits_xgboost} splits')

# %% [markdown] tags=[]
# ***
# # TODO

# %% [markdown] tags=[]
# ***
# # Evaluate

# %%
try:
    print(f'XGBoost used {n_splits_xgboost} splits vs FIGS {n_splits_figs}')
    print(f'That is {n_splits_xgboost-n_splits_figs}, or {(n_splits_xgboost-n_splits_figs)/n_splits_figs:.0%}, more splits!')
except:
    pass


# %%
def make_rocs(model, X, y, best_iteration=False):
    if best_iteration:
        y_pred = model.predict_proba(X, iteration_range=(0, model.best_iteration+1))[:,1]
    else:
        y_pred = model.predict_proba(X)[:,1]
    y_pred_sorted = sorted(y_pred)

    fpr, tpr, thr_of_fpr_tpr = roc_curve(y, y_pred)
    n_predicted_positive_of_fpr_tpr = [len(y_pred_sorted) - bisect.bisect_left(y_pred_sorted, _thr) for _thr in thr_of_fpr_tpr]
    dfp_eval_fpr_tpr = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thr': thr_of_fpr_tpr, 'n_predicted_positive': n_predicted_positive_of_fpr_tpr})
    dfp_eval_fpr_tpr = dfp_eval_fpr_tpr.sort_values(by='thr').reset_index(drop=True)

    precision, recall, thr_of_precision_recall = precision_recall_curve(y, y_pred)
    thr_of_precision_recall = np.insert(thr_of_precision_recall, 0, [0])
    n_predicted_positive_of_precision_recall = [len(y_pred_sorted) - bisect.bisect_left(y_pred_sorted, _thr) for _thr in thr_of_precision_recall]
    dfp_eval_precision_recall = pd.DataFrame({'precision': precision, 'recall': recall, 'thr': thr_of_precision_recall, 'n_predicted_positive': n_predicted_positive_of_precision_recall})
    dfp_eval_precision_recall['f1'] = 2*(dfp_eval_precision_recall['precision'] * dfp_eval_precision_recall['recall']) / (dfp_eval_precision_recall['precision'] + dfp_eval_precision_recall['recall'])

    return {'dfp_eval_fpr_tpr': dfp_eval_fpr_tpr, 'dfp_eval_precision_recall': dfp_eval_precision_recall}


# %%
roc_figs_train = make_rocs(model_figs, X_train, y_train)
roc_figs_holdout = make_rocs(model_figs, X_holdout, y_holdout)
roc_xgboost_train = make_rocs(model_xgboost, X_train, y_train, best_iteration=True)
roc_xgboost_holdout = make_rocs(model_xgboost, X_holdout, y_holdout, best_iteration=True)

# %%
# models_for_roc_dict = {
#     {**{'name': 'FIGS_train', 'nname': 'FIGS (Train)', 'c': 'C2', 'ls': '-'}, **roc_},
#     {**{'name': 'XGBoost', 'nname': 'XGBoost', 'c': 'black', 'ls': '--'}, **roc_},
# }

# %%
pop_PPV_train = len(np.where(y_train == 1)[0]) / len(y_train) # P / (P + N)
pop_PPV_holdout = len(np.where(y_holdout == 1)[0]) / len(y_holdout)

# %% [markdown] tags=[]
# ### Standard TPR vs FPR ROC

# %%
# plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=False, inverse_log=False, inline=inline)

# %% [markdown] tags=[]
# ### Precision vs Recall ROC

# %%
# plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=False, inverse_log=False, precision_recall=True,
#     pop_PPV=pop_PPV, y_axis_params={'min': -0.05}, inline=inline)

# %% [markdown] tags=[]
# ***
# # Tree Plots

# %% [markdown] tags=[]
# ## FIGS

# %%
dt_figs_0 = extract_sklearn_tree_from_figs(model_figs, tree_num=0, n_classes=2)
sk_figs_0 = ShadowSKDTree(dt_figs_0, X_train, y_train, feat_names, 'y', [0, 1])

dt_figs_1 = extract_sklearn_tree_from_figs(model_figs, tree_num=1, n_classes=2)
sk_figs_1 = ShadowSKDTree(dt_figs_1, X_train, y_train, feat_names, 'y', [0, 1])

# %%
from dtreeviz.colors import color_blind_friendly_colors # mpl_colors

# %%
trees.dtreeviz(sk_figs_0)

# %%
trees.dtreeviz(sk_figs_1)

# %%
trees.ctreeviz_leaf_samples(sk_figs_0)

# %%
trees.ctreeviz_leaf_samples(sk_figs_1)

# %%
expr_figs_0 = skompile(dt_figs_0.predict_proba, feat_names)

# %%
print(expr_figs_0.to('sqlalchemy/sqlite', component=1, assign_to='tree_0'))

# %% tags=[]
print(expr_figs_0.to('python/code'))

# %%
expr_figs_1 = skompile(dt_figs_1.predict_proba, feat_names)

# %%
print(expr_figs_1.to('sqlalchemy/sqlite', component=1, assign_to='tree_1'))

# %% tags=[]
print(expr_figs_1.to('python/code'))

# %% [markdown] tags=[]
# ## XGBoost
# Tree 0 only

# %%
