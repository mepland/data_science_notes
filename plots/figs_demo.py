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

# TODO remove
import sys,os
sys.path.append(os.path.expanduser('~/imodels'))
sys.path.append(os.path.expanduser('~/dtreeviz'))

########################################################
# python
import os
import time
import pandas as pd
import numpy as np
import scipy.stats
norm = scipy.stats.norm
import bisect

########################################################
# figs (imodels), xgboost, sklearn
import imodels
from imodels import FIGSClassifier

import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance

########################################################
# dtreeviz
# must follow the package README to properly install all dependencies!

from dtreeviz import trees
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs

from wand.image import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

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

datasets = ['train', 'holdout']

# %%
from plotting import * # load plotting code

# %%
inline=True # plot inline or to pdf
output = './figs_demo_output' # output dir
os.makedirs(output, exist_ok=True)


# %%
def save_dtreeviz(viz, m_path, fname, tag=''):
    os.makedirs(m_path, exist_ok=True)
    full_path = f'{m_path}/{fname}{tag}'

    # svg
    viz.save(f'{full_path}.svg')

    # pdf via svglib
    renderPDF.drawToFile(svg2rlg(f'{full_path}.svg'), f'{full_path}.pdf')

    # png via wand / ImageMagick
    img = Image(filename=f'{full_path}.svg', resolution=500)
    img.format = 'png'
    img.save(filename=f'{full_path}.png')

    # clean up graphviz dot file (no extension)
    os.remove(full_path)


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
# Note we are not using early stopping with FIGS, so use `X_trainVal` during training to take advantage of all rows.

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
print(f'FIGS used {len(model_figs.trees_)} trees and {n_splits_figs:,} splits')

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
print(f'XGBoost used {model_xgboost.best_ntree_limit} trees and {n_splits_xgboost:,} splits')

# %% [markdown] tags=[]
# ***
# # Evaluate

# %%
print(f'XGBoost used {n_splits_xgboost:,} splits vs FIGS {n_splits_figs:,}')
print(f'That is {n_splits_xgboost-n_splits_figs:,}, or {(n_splits_xgboost-n_splits_figs)/n_splits_figs:,.0%}, more splits!')


# %%
def classifier_metrics(model, X_train, y_train, X_holdout, y_holdout, feature_names, do_permutation_importance=True, print_classification_report=False):
    model_metrics = {}
    dfp_importance = pd.DataFrame({'feature': feature_names})
    dfp_importance['icolX'] = dfp_importance.index

    for dataset in datasets[::-1]:
        if dataset == 'holdout':
            X = X_holdout
            y = y_holdout
        elif dataset == 'train':
            X = X_train
            y = y_train
        y_pred = model.predict(X)
        # only want positive class prob
        try:
            # use best_iteration for XGBoost
            y_pred_prob = model.predict_proba(X, iteration_range=(0, model.best_iteration+1))[:, 1]
        except:
            y_pred_prob = model.predict_proba(X)[:, 1]

        model_metrics[dataset] = {}
        model_metrics[dataset]['accuracy_score'] = metrics.accuracy_score(y, y_pred)
        model_metrics[dataset]['precision_score'] = metrics.precision_score(y, y_pred, zero_division=0) # zero_division=0 hides divide by zero warnings that come up with LR doesn't converge
        model_metrics[dataset]['recall_score'] = metrics.recall_score(y, y_pred)
        model_metrics[dataset]['f1_score'] = metrics.f1_score(y, y_pred)
        model_metrics[dataset]['roc_auc_score'] = metrics.roc_auc_score(y, y_pred_prob)
        model_metrics[dataset]['average_precision_score'] = metrics.average_precision_score(y, y_pred_prob) # PR ROC AUC
        model_metrics[dataset]['log_loss'] = metrics.log_loss(y, y_pred)
        model_metrics[dataset]['cohen_kappa_score'] = metrics.cohen_kappa_score(y, y_pred)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        CM = metrics.confusion_matrix(y, y_pred)
        model_metrics[dataset]['confusion_matrix'] = CM
        model_metrics[dataset]['TN'] = CM[0][0]
        model_metrics[dataset]['FP'] = CM[0][1]
        model_metrics[dataset]['FN'] = CM[1][0]
        model_metrics[dataset]['TP'] = CM[1][1]
        model_metrics[dataset]['TNR'] = CM[0][0] / (CM[0][0] + CM[0][1]) # TN / (TN + FP)
        model_metrics[dataset]['NPV'] = CM[0][0] / (CM[0][0] + CM[1][0]) # TN / (TN + FN)

        model_metrics[dataset]['pop_PPV'] = len(np.where(y == 1)[0]) / len(y) # P / (P + N)

        # model_metrics[dataset]['dfp_y'] = pd.DataFrame( {'y': y, 'y_pred_prob': y_pred_prob, 'y_pred': y_pred} )

        # for LR models
        # model_converged = (model.n_iter_ < model.max_iter)[0]

        if print_classification_report:
            print(f'For {dataset}:')
            print(metrics.classification_report(y, y_pred))

        # ROC Curves
        def get_n_predicted_positive_vs_thr(y_pred_prob, thr):
            y_pred_prob_sorted = sorted(y_pred_prob)
            return [len(y_pred_prob_sorted) - bisect.bisect_left(y_pred_prob_sorted, _thr) for _thr in thr]

        fpr, tpr, thr_of_fpr_tpr = roc_curve(y, y_pred_prob)
        n_predicted_positive_vs_thr_of_fpr_tpr = get_n_predicted_positive_vs_thr(y_pred_prob, thr_of_fpr_tpr)
        dfp_eval_fpr_tpr = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thr': thr_of_fpr_tpr, 'n_predicted_positive': n_predicted_positive_vs_thr_of_fpr_tpr}).sort_values(by='thr').reset_index(drop=True)

        precision, recall, thr_of_precision_recall = precision_recall_curve(y, y_pred_prob)
        thr_of_precision_recall = np.insert(thr_of_precision_recall, 0, [0])
        n_predicted_positive_vs_thr_of_precision_recall = get_n_predicted_positive_vs_thr(y_pred_prob, thr_of_precision_recall)
        dfp_eval_precision_recall = pd.DataFrame({'precision': precision, 'recall': recall, 'thr': thr_of_precision_recall, 'n_predicted_positive': n_predicted_positive_vs_thr_of_precision_recall})
        dfp_eval_precision_recall['f1'] = 2*(dfp_eval_precision_recall['precision'] * dfp_eval_precision_recall['recall']) / (dfp_eval_precision_recall['precision'] + dfp_eval_precision_recall['recall'])

        model_metrics[dataset]['dfp_eval_fpr_tpr'] = dfp_eval_fpr_tpr
        model_metrics[dataset]['dfp_eval_precision_recall'] = dfp_eval_precision_recall

        if do_permutation_importance:
            # print('Start do_permutation_importance func')
            # Permutation feature importance
            # slow for thousands of features!
            # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
            # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
            # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
            # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
            _permutation_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc')
            # print('End permutation_importance func')

            importance_permutation_mean = _permutation_importance['importances_mean']
            importance_permutation_std = _permutation_importance['importances_std']
            dfp_importance_permutation = pd.DataFrame({f'importance_permutation_{dataset}_mean': importance_permutation_mean, f'importance_permutation_{dataset}_std': importance_permutation_std})
            dfp_importance_permutation['icolX'] = dfp_importance_permutation.index
            dfp_importance_permutation[f'importance_permutation_{dataset}_pct'] = dfp_importance_permutation[f'importance_permutation_{dataset}_mean'].rank(pct=True)
            dfp_importance = pd.merge(dfp_importance, dfp_importance_permutation, on='icolX', how='left')

    # for LR models
    # dfp_coef = pd.DataFrame({'coefficients': model.coef_[0]})
    # dfp_coef['abs_coeff'] = dfp_coef['coefficients'].abs()
    # dfp_coef['icolX'] = dfp_coef.index
    # dfp_importance = pd.merge(dfp_importance, dfp_coef, on='icolX', how='left')

    # TODO
    # Gini impurity importance - a mean decrease in impurity (MDI) importance (both RF and BDT)
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_
    # importances_gini = model.feature_importances_
    # estimators = model.estimators_
    # importances_gini_std = np.std([tree.feature_importances_ for tree in estimators], axis=0)
    # dfp_importance_gini = pd.DataFrame({'importance_gini': importances_gini, 'importance_gini_std': importances_gini_std})
    # dfp_importance_gini['icolX'] = dfp_importance_gini.index
    # dfp_importance_gini['importance_gini_pct'] = dfp_importance_gini['importance_gini'].rank(pct=True)
    # dfp_importance = pd.merge(dfp_importance, dfp_importance_gini, on='icolX', how='left')

    target_cols_importance = [
        'feature',
        # 'coefficients',
        'importance_permutation_holdout_mean',
        'importance_permutation_holdout_std',
        'importance_permutation_holdout_pct',
        'importance_permutation_train_mean',
        'importance_permutation_train_std',
        'importance_permutation_train_pct',
        # 'importance_gini',
        # 'importance_gini_std',
        # 'importance_gini_pct',
        'icolX',
        # 'abs_coeff',
    ]
    _cols = [_col for _col in target_cols_importance if _col in dfp_importance.columns] + [_col for _col in dfp_importance.columns if _col not in target_cols_importance]
    dfp_importance = dfp_importance[_cols]
    if 'importance_permutation_holdout_mean' in dfp_importance.columns:
        sort_col = 'importance_permutation_holdout_mean'
    elif 'importance_gini' in dfp_importance.columns:
        sort_col = 'importance_gini'
    else:
        sort_col = 'icolX'
    dfp_importance = dfp_importance.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    dfp_importance = dfp_importance.drop(['icolX'], axis=1)
    # if 'abs_coeff' in dfp_importance.columns:
    #     dfp_importance = dfp_importance.drop(['abs_coeff'], axis=1)

    model_metrics['dfp_importance'] = dfp_importance

    return model_metrics


# %%
model_metrics_figs = classifier_metrics(model_figs, X_trainVal, y_trainVal, X_holdout, y_holdout, feat_names)
model_metrics_xgboost = classifier_metrics(model_xgboost, X_train, y_train, X_holdout, y_holdout, feat_names)

metric_rows = []
for model, name, model_metrics in [(model_figs, 'FIGS', model_metrics_figs), (model_xgboost, 'XGBoost', model_metrics_xgboost)]:
    for dataset in datasets[::-1]:
        dataset_metrics = {'model': name, 'dataset': dataset}
        for k,v in model_metrics[dataset].items():
            if k not in ['confusion_matrix', 'dfp_eval_fpr_tpr', 'dfp_eval_precision_recall', 'dfp_y']:
                dataset_metrics[k] = v
        metric_rows.append(dataset_metrics)

dfp_metrics = pd.DataFrame(metric_rows)
dfp_metrics = dfp_metrics.sort_values(by=['model', 'dataset'], ascending=[True, True]).reset_index(drop=True)
display(dfp_metrics)

# %%
print('FIGS Feature Importances')
_dfp = model_metrics_figs['dfp_importance']
display(_dfp.loc[0 < _dfp['importance_permutation_holdout_mean']])

# %%
print('XGBoost Feature Importances')
_dfp = model_metrics_xgboost['dfp_importance']
display(_dfp.loc[0 < _dfp['importance_permutation_holdout_mean']])

# %% [markdown] tags=[]
# ***
# # TODO

# %%
# models_for_roc_dict = {
#     {**{'name': 'FIGS_train', 'nname': 'FIGS (Train)', 'c': 'C2', 'ls': '-'}, **roc_},
#     {**{'name': 'XGBoost', 'nname': 'XGBoost', 'c': 'black', 'ls': '--'}, **roc_},
# }

# %% [markdown] tags=[]
# ### Standard TPR vs FPR ROC

# %%
# plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=False, inverse_log=False, inline=inline)

# %% [markdown] tags=[]
# ### Precision vs Recall ROC

# %%
# plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=False, inverse_log=False, precision_recall=True,
#     pop_PPV=model_metrics[dataset]['pop_PPV'], y_axis_params={'min': -0.05}, inline=inline)

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
from dtreeviz.colors import color_blind_friendly_colors # mpl_colors # TODO

# %%
viz = trees.dtreeviz(sk_figs_0)

# %%
save_dtreeviz(viz, output, 'dtreeviz_figs_0')

# %%
viz = trees.dtreeviz(sk_figs_1)

# %%
viz = trees.ctreeviz_leaf_samples(sk_figs_0)

# %%
viz = trees.ctreeviz_leaf_samples(sk_figs_1)

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
