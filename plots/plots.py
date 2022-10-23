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

# %% [markdown]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

########################################################
# python
import pandas as pd
import numpy as np
import scipy.stats
norm = scipy.stats.norm
import bisect

########################################################
# xgboost, sklearn
import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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

# %% [markdown] tags=[]
# ***
# # ROC Curve Demo
# A demonstration of TPR vs FPR and Precision vs Recall ROC curves on a synthetic dataset with XGBoost

# %% [markdown] tags=[]
# ## Generate Random Data

# %%
X, y = make_classification(n_samples=int(1e5), n_features=50, n_informative=20, n_redundant=10, n_repeated=2,
                           n_classes=2, n_clusters_per_class=5, weights=[0.7], flip_y=0.2, class_sep=0.9,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=rnd_seed)

# %% [markdown]
# Make Train, Validation, and Holdout Sets

# %%
X_trainVal, X_holdout, y_trainVal, y_holdout = train_test_split(X, y, test_size=0.33, random_state=rnd_seed, stratify=y)
del X; del y;

X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.2, random_state=rnd_seed, stratify=y_trainVal)
del X_trainVal; del y_trainVal;

# %% [markdown]
# #### Set hyperparameters

# %%
params_default = {'max_depth': 6, 'learning_rate': 0.3, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 1.0}

# %%
params_bad = {'max_depth': 2, 'learning_rate': 1.0, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}

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

# %% [markdown]
# ## Setup XGBClassifiers
# #### Run with initial hyperparameters as a baseline

# %%
model_default = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                                  objective=fixed_setup_params['xgb_objective'],
                                  verbosity=fixed_setup_params['xgb_verbosity'],
                                  random_state=rnd_seed+3, **params_default, use_label_encoder=False)
model_default.fit(X_train, y_train, **fixed_fit_params);

# %% [markdown]
# #### Run with bad hyperparameters to compare

# %%
model_bad = xgb.XGBClassifier(n_estimators=round(0.25*fixed_setup_params['max_num_boost_rounds']),
                                  objective=fixed_setup_params['xgb_objective'],
                                  verbosity=fixed_setup_params['xgb_verbosity'],
                                  random_state=rnd_seed+4, **params_bad, use_label_encoder=False)
model_bad.fit(X_train, y_train, **fixed_fit_params);


# %% [markdown]
# ## Evaluate

# %%
def eval_model(model, X, y):
    y_pred = model.predict_proba(X, iteration_range=(0, model.best_iteration+1))[:,1]
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
models_for_roc= [
    {**{'name': 'model_1', 'nname': 'Model 1', 'c': 'C2', 'ls': '-'}, **eval_model(model_default, X_holdout, y_holdout)},
    {**{'name': 'model_2', 'nname': 'Model 2', 'c': 'black', 'ls': '--'}, **eval_model(model_bad, X_holdout, y_holdout)},
]

# %%
pop_PPV = len(np.where(y_holdout == 1)[0]) / len(y_holdout) # P / (P + N)

# %% [markdown] tags=[]
# ### Standard TPR vs FPR ROC

# %%
plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=True, inverse_log=False, inline=inline)

# %% [markdown]
# #### Inverse Log TPR vs FPR ROC

# %%
plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=True, inverse_log=True,
    x_axis_params={'max': 0.6}, y_axis_params={'min': 1e0, 'max': 1e1}, inline=inline)

# %% [markdown]
# ### Precision vs Recall ROC

# %%
plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=True, inverse_log=False, precision_recall=True,
    pop_PPV=pop_PPV, y_axis_params={'min': -0.05}, inline=inline)

# %% [markdown]
# #### Inverse Log Precision vs Recall ROC

# %%
plot_rocs(models_for_roc, m_path=f'{output}/roc_curves', rndGuess=False, inverse_log=True, precision_recall=True, pop_PPV=pop_PPV, inline=inline)

# %% [markdown]
# ### Precision vs Recall ROC with Additional Plots

# %%
plot_rocs(models_for_roc[:1], m_path=f'{output}/roc_curves', tag='_f1', rndGuess=True, inverse_log=False, precision_recall=True, pop_PPV=pop_PPV,
    y_axis_params={'min': -0.05}, inline=inline, better_ann=False,
    plot_f1=True, plot_n_predicted_positive=False)

# %%
plot_rocs(models_for_roc[:1], m_path=f'{output}/roc_curves', tag='_n_pos', rndGuess=True, inverse_log=False, precision_recall=True, pop_PPV=pop_PPV,
    y_axis_params={'min': -0.05}, inline=inline, better_ann=False,
    plot_f1=False, plot_n_predicted_positive=True)

# %%
plot_rocs(models_for_roc[:1], m_path=f'{output}/roc_curves', tag='_f1_n_pos', rndGuess=True, inverse_log=False, precision_recall=True, pop_PPV=pop_PPV,
    y_axis_params={'min': -0.05}, inline=inline, better_ann=False,
    plot_f1=True, plot_n_predicted_positive=True)

# %% [markdown]
# ***
# # Hypothesis Testing Power Example

# %%
Z_a = norm.ppf(1-0.05) + np.sqrt(100)*(10-10.5)/2
print(f'Z_a = {Z_a:.4f}')
print(f'Power = 1-beta = {1-norm.cdf(Z_a):.4f}')

# %% [markdown]
# ***
# # inverse_transform_sampling_normal_dist
# Adapted from https://commons.wikimedia.org/wiki/File:Inverse_transform_sampling.png

# %%
norm = scipy.stats.norm
x = np.linspace(-2, 2, 100)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axhline(y=0, color='k', lw=1)
ax.axvline(x=0, color='k', lw=1)

ax.set_xlim([-2.,2.])
ax.set_ylim([-2.,2.])
ax.set_xlabel('$x$', labelpad=7)

ax.plot(x, norm.pdf(x), label=r'PDF $P(x)$')
ax.plot(x, norm.cdf(x), label=r'CDF $F_{X}(x)$')
ax.plot(x, norm.ppf(x), label='$F^{-1}_{X}(U)$')
ax.plot(x,x,'--k', lw=1)
ax.plot([norm.ppf(0.2),0.2],[0.2,norm.ppf(0.2)],'o--k', ms=4, lw=1)

ax.xaxis.set_ticks(np.arange(-2, 3, 1))
ax.yaxis.set_ticks(np.arange(-2, 3, 1))

plt.text(-0.5, -0.5, 'Invert', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', bbox=dict(edgecolor='white', facecolor='white', alpha=1))

dx = 0/72.; dy = -5/72.
offsetx = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
offsety = matplotlib.transforms.ScaledTranslation(dy, dx, fig.dpi_scale_trans)
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offsetx)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offsety)

leg = ax.legend(loc='upper left',frameon=False)
leg.get_frame().set_facecolor('none')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/inverse_transform_sampling_normal_dist.pdf')
    plt.close('all')

# %% [markdown] tags=[]
# ***
# # rejection_sampling
# Adapted from https://www.data-blogger.com/2016/01/24/the-mathematics-behind-rejection-sampling/

# %%
# The multiplication constant to make our probability estimation fit
M = 3
# Number of samples to draw from the probability estimation function
N = 5000

# The target probability density function
f = lambda x: 0.6 * norm.pdf(x, 0.35, 0.05) + 0.4 * norm.pdf(x, 0.65, 0.08)

# The approximated probability density function
g = lambda x: norm.pdf(x, 0.45, 0.2)

# A number of samples, drawn from the approximated probability density function
np.random.seed = 42
x_samples = M * np.random.normal(0.45, 0.2, (N,))

# A number of samples in the interval [0, 1]
u = np.random.uniform(0, 1, (N, ))

# Now examine all the samples and only use the samples found by rejection sampling
samples = [(x_samples[i], u[i] * M * g(x_samples[i])) for i in range(N) if u[i] < f(x_samples[i]) / (M * g(x_samples[i]))]

# %%
fig, ax = plt.subplots()

ax.set_xlim([0.,1.])
ax.set_ylim([0.,6.5])

ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False)

x = np.linspace(0, 1, 500)

ax.plot(x, f(x), '-', label='$f(x)$')
ax.plot(x, M * g(x), '-', label='$M \cdot g(x)$')

ax.plot([sample[0] for sample in samples], [sample[1] for sample in samples], '.', label='Samples')

leg = ax.legend(loc='upper right',frameon=False)
leg.get_frame().set_facecolor('none')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/rejection_sampling.pdf')
    plt.close('all')

# %% [markdown] tags=[]
# ***
# # Hypergeometric PMF
# Adapted from https://en.wikipedia.org/wiki/File:HypergeometricPDF.png and https://en.wikipedia.org/wiki/File:Geometric_pmf.svg

# %%
fig, ax = plt.subplots()

ax.tick_params(
    axis='both',
    which='minor',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=True,
    labelbottom=True)

x = np.linspace(0, 60, 61)

# (N, K, n)
params = [
    [500, 50, 100],
    [500, 60, 200],
    [500, 70, 300],
]

colors = ['orange', 'purple', 'lightblue']

for param, color in zip(params, colors):
    pmf = scipy.stats.hypergeom(*param).pmf(x)
    ax.plot(x, pmf, '-', c='grey', lw=1)
    ax.plot(x, pmf, 'o', c=color, markeredgecolor='black', lw=3, label=f'$N$ = {param[0]}, $K$ = {param[1]}, $n$ = {param[2]}')

ax.set_xlim([-2.,62.])
ax.set_ylim([0.,0.16])
ax.set_xlabel('k', labelpad=7)
ax.set_ylabel('P(X = k)', labelpad=7)

leg = ax.legend(loc='upper right',frameon=False)
leg.get_frame().set_facecolor('none')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/hypergeometric_pmf.pdf')
    plt.close('all')

# %% [markdown] tags=[]
# ***
# # Spearman Correlation
# Adapted from https://en.wikipedia.org/wiki/File:Spearman_fig1.svg and https://en.wikipedia.org/wiki/File:Spearman_fig3.svg

# %%
fig, ax = plt.subplots()

ax.tick_params(
    axis='both',
    which='minor',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=True,
    labelbottom=True)

rnd_state = np.random.RandomState(43)
x = rnd_state.uniform(size=100)
y = np.log(x/(1-x))
y = np.sign(y)*np.abs(y)**1.4

Rx = np.argsort(np.argsort(x))
Ry = np.argsort(np.argsort(y))

cs = np.cov(Rx,Ry)
cs = cs[0,1]/np.sqrt(cs[0,0]*cs[1,1])

cp = np.cov(x,y)
cp = cp[0,1]/np.sqrt(cp[0,0]*cp[1,1])

ax.plot(x, y, 'o', color='orange', markeredgecolor='black', lw=3)
fig.suptitle(f'Spearman correlation = {cs:.2f}\nPearson correlation = {cp:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/spearman_corr_non_para.pdf')
    plt.close('all')

# %%
fig, ax = plt.subplots()

ax.tick_params(
    axis='both',
    which='minor',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=True,
    labelbottom=True)

r = 0.8

rnd_state = np.random.RandomState(0)
x = rnd_state.normal(size=100)
y = r*x + np.sqrt(1-r**2)*rnd_state.normal(size=100)

ii = np.argsort(-x)
x[ii[0:5]] *= 3

Rx = np.argsort(np.argsort(x))
Ry = np.argsort(np.argsort(y))

cs = np.cov(Rx,Ry)
cs = cs[0,1]/np.sqrt(cs[0,0]*cs[1,1])

cp = np.cov(x,y)
cp = cp[0,1]/np.sqrt(cp[0,0]*cp[1,1])

ax.plot(x, y, 'o', color='orange', markeredgecolor='black', lw=3)
fig.suptitle(f'Spearman correlation = {cs:.2f}\nPearson correlation = {cp:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/spearman_corr_outliers.pdf')
    plt.close('all')

# %% [markdown] tags=[]
# ***
# # PCA Scree Plot

# %%
n_dimensions = 20
n_observations = 100
X = np.random.randn(n_observations, n_dimensions)
pca = PCA()
pca.fit(scale(X));

# %%
plot_scree(pca, m_path=output, plot_cumsum=True, reference_lines=True, inline=inline)

# %% [markdown] tags=[]
# ***
# # Gini Impurity vs Information Entropy

# %%
fig, ax = plt.subplots()

ax.tick_params(
    axis='both',
    which='minor',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=True,
    labelbottom=True)

xs = np.linspace(0, 1, 100)
gini = [6*x*(1-x) for x in xs]
H = [-4*x*np.log(x) if x != 0. else 0. for x in xs]

ax.plot(xs, gini, '-', c='C0', lw=1, label='Gini Impurity')
ax.plot(xs, H, '--', c='C1', lw=1, label='Information Entropy $H$')

ax.xaxis.set_ticks(np.linspace(0, 1, 5))
ax.yaxis.set_ticks(np.linspace(0, 2, 5))
ax.set_xlabel('$p$', labelpad=7)
ax.set_ylabel('Normalized Impurity', labelpad=7)

leg = ax.legend(loc='upper right',frameon=False)
leg.get_frame().set_facecolor('none')

plt.tight_layout()
if inline:
    fig.show()
else:
    os.makedirs(output, exist_ok=True)
    fig.savefig(f'{output}/gini_vs_info_entropy.pdf')
    plt.close('all')
