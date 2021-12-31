# python
import os
import numpy as np

from sklearn.metrics import auc

########################################################
# plotting
import matplotlib as mpl
# mpl.use('Agg', warn=False)
# mpl.rcParams['font.family'] = ['HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Arial', 'Lucida Grande', 'sans-serif']
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.labelsize']     = 13
mpl.rcParams['ytick.labelsize']     = 13
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['xtick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['xtick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['xtick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['xtick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['xtick.minor.pad']     = 1.4  # distance to the minor tick label in points
mpl.rcParams['ytick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['ytick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['ytick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['ytick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['ytick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['ytick.minor.pad']     = 1.4  # distance to the minor tick label in points
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.ticker as ticker
# from matplotlib.ticker import LogLocator

########################################################
# Set common plot parameters
vsize = 11 # inches
# aspect ratio width / height
aspect_ratio_single = 4./3.
aspect_ratio_multi = 1.

plot_png=False
png_dpi=200

std_ann_x = 0.80
std_ann_y = 0.94

########################################################
# plot overlaid roc curves for many models
def plot_rocs(models, m_path='output', fname='roc', tag='', rndGuess=False, better_ann=True, grid=False, inverse_log=False, precision_recall=False, pop_PPV=None, plot_f1=False, plot_n_predicted_positive=False, additional_axes_in_legend=False, x_axis_params=None, y_axis_params=None, inline=False):

    if not precision_recall:
        plot_f1 = False
    extra_y_space = plot_f1 and plot_n_predicted_positive

    fig, ax = plt.subplots()
    # fig.set_size_inches(aspect_ratio_single*vsize, vsize)
    if extra_y_space:
        fig.subplots_adjust(right=0.7)

    if plot_f1 or plot_n_predicted_positive:
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')

    if plot_f1:
        ax_f1 = ax.twinx()
        c_f1 = 'C0'
    if plot_n_predicted_positive:
        ax_n_predicted_positive = ax.twinx()
        c_n_predicted_positive = 'C1'
        if extra_y_space:
            # Offset the right spine of ax_f1. The ticks and label have already been placed on the right by twinx above.
            ax_n_predicted_positive.spines.right.set_position(('axes', 1.2))

    leg_objects = []

    if fname == 'roc':
        if precision_recall:
            fname = f'{fname}_precision_recall'
        if inverse_log:
            fname = f'{fname}_inverse_log'

    for model in models:
        # models is a list of dicts with keys for: name, nname, c (color), ls (linestyle), dfp_eval_fpr_tpr, dfp_eval_precision_recall
        # dfp_eval_fpr_tpr has columns: fpr, tpr, thr, n_predicted_positive
        # dfp_eval_precision_recall has columns: precision, recall, thr, n_predicted_positive, f1

        if not precision_recall:
            dfp_eval = model['dfp_eval_fpr_tpr'].copy()
            dfp_eval = dfp_eval.rename({'fpr': 'x', 'tpr': 'y'}, axis='columns')
        else:
            dfp_eval = model['dfp_eval_precision_recall'].copy()
            dfp_eval = dfp_eval.rename({'recall': 'x', 'precision': 'y'}, axis='columns')

        auc_value = auc(dfp_eval['x'], dfp_eval['y'])
        if inverse_log:
            with np.errstate(divide='ignore'):
                dfp_eval['y'] = np.divide(1., dfp_eval['y'])

        label=f"{model['nname']}, AUC: {auc_value:.4f}"

        leg_objects.append(ax.plot(dfp_eval['x'], dfp_eval['y'], lw=2, c=model.get('c', 'blue'), ls=model.get('ls', '-'), label=label)[0])

        if plot_f1:
            p_f1, = ax_f1.plot(dfp_eval['x'], dfp_eval['f1'], lw=2, c=c_f1, ls='--', label='$F_{1}$')
            if additional_axes_in_legend:
                leg_objects.append(p_f1)
        if plot_n_predicted_positive:
            p_n_predicted_positive, = ax_n_predicted_positive.plot(dfp_eval['x'], dfp_eval['n_predicted_positive'], lw=2, c=c_n_predicted_positive, ls='--', label='# Predicted Positive')
            if additional_axes_in_legend:
                leg_objects.append(p_n_predicted_positive)

        fname = f"{fname}_{model['name']}"

    if grid:
        ax.grid()

    if rndGuess:
        if not precision_recall:
            if inverse_log:
                x = np.linspace(1e-10, 1.)
                leg_objects.append(ax.plot(x, 1/x, color='grey', linestyle=':', linewidth=2, label='Random Guess')[0])
            else:
                x = np.linspace(0., 1.)
                leg_objects.append(ax.plot(x, x, color='grey', linestyle=':', linewidth=2, label='Random Guess')[0])
        else:
            if pop_PPV is None:
                raise ValueError('Need pop_PPV to plot random guess curve for precision_recall!')
            if inverse_log:
                x = np.linspace(1e-10, 1.)
                y = pop_PPV*np.ones(len(x))
                leg_objects.append(ax.plot(x, 1/y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')[0])
            else:
                x = np.linspace(0., 1.)
                y = pop_PPV*np.ones(len(x))
                leg_objects.append(ax.plot(x, y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')[0])

    ax.set_zorder(10)
    ax.patch.set_alpha(0.0)
    if len(leg_objects) > 0:
        if not precision_recall:
            loc_x='right';
        else:
            loc_x='left';
        if inverse_log:
            loc_y='upper';
        else:
            loc_y='lower';
        leg = ax.legend(leg_objects, [ob.get_label() for ob in leg_objects], loc=f'{loc_y} {loc_x}', ncol=1)
        leg.get_frame().set_fill(True)
        leg.set_zorder(11)
        leg.get_frame().set_edgecolor('none')
        if not (plot_f1 or plot_n_predicted_positive):
            leg.get_frame().set_facecolor('none')
        else:
            leg.get_frame().set_alpha(0.8)

    xlabel = 'False Positive Rate'
    ylabel = 'True Positive Rate'
    if precision_recall:
        xlabel = 'Recall (Sensitivity, TPR)'
        ylabel = 'Precision (PPV)'

    ax.set_xlim([0.,1.])
    ax.set_xlabel(xlabel)
    if inverse_log:
        ax.set_yscale('log')
        ax.set_ylabel(f'Inverse {ylabel}')
    else:
        ax.set_ylabel(ylabel)

    if not isinstance(x_axis_params, dict):
        x_axis_params = dict()
    x_min_current, x_max_current = ax.get_xlim()
    x_min = x_axis_params.get('min', x_min_current)
    x_max = x_axis_params.get('max', x_max_current)
    ax.set_xlim([x_min, x_max])

    if not isinstance(y_axis_params, dict):
        y_axis_params = dict()
    y_min_current, y_max_current = ax.get_ylim()
    y_min = y_axis_params.get('min', y_min_current)
    y_max = y_axis_params.get('max', y_max_current)
    ax.set_ylim([y_min, y_max])

    if better_ann:
        if not precision_recall:
            if inverse_log:
                better_x = -0.07; better_y = -0.12; better_rot = -45;
            else:
                better_x = -0.07; better_y = 1.08; better_rot = 45;
        else:
            if inverse_log:
                better_x = 1.07; better_y = -0.12; better_rot = 45;
            else:
                better_x = 1.07; better_y = 1.08; better_rot = -45;

        plt.text(better_x, better_y, 'Better', size=12, rotation=better_rot, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

    if plot_f1:
        # ax_f1.set_ylim([-0.05,1.])
        # just use Precision (PPV) range, should be 0 to 1
        ax_f1.set_ylim([y_min, y_max])
        ax_f1.set_ylabel('$F_{1}$')
        ax_f1.spines.right.set_color(c_f1)
        ax_f1.yaxis.label.set_color(c_f1)
        ax_f1.tick_params(axis='y', colors=c_f1, which='major')
        ax_f1.tick_params(axis='y', colors=c_f1, which='minor')

    if plot_n_predicted_positive:
        ax_n_predicted_positive.set_ylabel('# Predicted Positive')
        ax_n_predicted_positive.spines.right.set_color(c_n_predicted_positive)
        ax_n_predicted_positive.yaxis.label.set_color(c_n_predicted_positive)
        ax_n_predicted_positive.tick_params(axis='y', colors=c_n_predicted_positive, which='major')
        ax_n_predicted_positive.tick_params(axis='y', colors=c_n_predicted_positive, which='minor')


    plt.tight_layout()
    if inline:
        fig.show()
    else:
        os.makedirs(m_path, exist_ok=True)
        if plot_png:
            fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
        fig.savefig(f'{m_path}/{fname}{tag}.pdf')
        plt.close('all')

########################################################
# Create scree plot from sklearn pca object
def plot_scree(pca, m_path='output', fname='scree', tag='', grid=False, plot_cumsum=False, plots_in_legend=False, reference_lines=False, x_axis_params=None, y_axis_params=None, inline=False):
    eigenvalues = pca.explained_variance_
    cumulative_explained_variance = 100*np.cumsum(pca.explained_variance_ratio_)
    principle_components = np.arange(len(eigenvalues))+1

    fig, ax = plt.subplots()
    # fig.set_size_inches(aspect_ratio_single*vsize, vsize)
    if plot_cumsum:
        fig.subplots_adjust(right=0.7)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax_cumsum = ax.twinx()
        c_cumsum = 'C1'

    leg_objects = []

    p_eigenvalues, = ax.plot(principle_components, eigenvalues, marker='o', lw=2, c='C0', label='Eigenvalues')
    if plots_in_legend:
        leg_objects.append(p_eigenvalues)

    if plot_cumsum:
        p_cumsum, = ax_cumsum.plot(principle_components, cumulative_explained_variance, lw=2, c=c_cumsum, ls='--', label='Explained Variance [%]')
        if plots_in_legend:
            leg_objects.append(p_cumsum)

    if reference_lines:
        leg_objects.append(ax.axhline(y=1., c='C0', lw=0.8, ls=':', label='Kaiser Criterion'))
        if plot_cumsum:
            leg_objects.append(ax_cumsum.axhline(y=80., c=c_cumsum, lw=0.8, ls=':', label='80% Explained Variance'))

    if grid:
        ax.grid()

    ax.set_zorder(10)
    ax.patch.set_alpha(0.0)
    if len(leg_objects) > 0:
        loc_x='right';
        loc_y='lower';
        leg = ax.legend(leg_objects, [ob.get_label() for ob in leg_objects], loc=f'{loc_y} {loc_x}', ncol=1)
        leg.get_frame().set_fill(True)
        leg.set_zorder(11)
        leg.get_frame().set_edgecolor('none')
        leg.get_frame().set_facecolor('none')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalues')

    if not isinstance(x_axis_params, dict):
        x_axis_params = dict()
    x_min_current, x_max_current = ax.get_xlim()
    x_min = x_axis_params.get('min', x_min_current)
    x_max = x_axis_params.get('max', x_max_current)
    ax.set_xlim([x_min, x_max])

    if not isinstance(y_axis_params, dict):
        y_axis_params = {'min': 0}
    y_min_current, y_max_current = ax.get_ylim()
    y_min = y_axis_params.get('min', y_min_current)
    y_max = y_axis_params.get('max', y_max_current)
    ax.set_ylim([y_min, y_max])

    if plot_cumsum:
        ax_cumsum.set_ylim([0., 100.])
        ax_cumsum.set_ylabel('Explained Variance [%]')
        ax_cumsum.spines.right.set_color(c_cumsum)
        ax_cumsum.yaxis.label.set_color(c_cumsum)
        ax_cumsum.tick_params(axis='y', colors=c_cumsum, which='major')
        ax_cumsum.tick_params(axis='y', colors=c_cumsum, which='minor')

    plt.tight_layout()
    if inline:
        fig.show()
    else:
        os.makedirs(m_path, exist_ok=True)
        if plot_png:
            fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
        fig.savefig(f'{m_path}/{fname}{tag}.pdf')
        plt.close('all')
