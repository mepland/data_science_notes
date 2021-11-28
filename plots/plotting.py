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
def plot_rocs(models, m_path='output', fname='roc', tag='', rndGuess=False, better_ann=True, grid=False, inverse_log=False, precision_recall=False, pop_PPV=None, x_axis_params=None, y_axis_params=None, inline=False):
	fig, ax = plt.subplots()

	x_var = 'fpr'
	y_var = 'tpr'
	if precision_recall:
		x_var = 'rec'
		y_var = 'pre'

	if fname == 'roc':
		if precision_recall:
			fname = f'{fname}_precision_recall'
		if inverse_log:
			fname = f'{fname}_inverse_log'

	for model in models:
		# models is a list of dicts with keys name, nname, fpr, tpr, (or pre, rec), c (color), ls (linestyle)

		if inverse_log:
			with np.errstate(divide='ignore'):
				y_values = np.divide(1., model[y_var])
		else:
			y_values = model[y_var]

		label=f"{model['nname']}, AUC: {auc(model[x_var],model[y_var]):.4f}"

		ax.plot(model[x_var], y_values, lw=2, c=model.get('c', 'blue'), ls=model.get('ls', '-'), label=label)

		fname = f"{fname}_{model['name']}"

	if grid:
		ax.grid()

	if inverse_log:
		leg_loc = 'upper right'
	else:
		leg_loc = 'lower right'

	if rndGuess:
		if not precision_recall:
			if inverse_log:
				x = np.linspace(1e-10, 1.)
				ax.plot(x, 1/x, color='grey', linestyle=':', linewidth=2, label='Random Guess')
			else:
				x = np.linspace(0., 1.)
				ax.plot(x, x, color='grey', linestyle=':', linewidth=2, label='Random Guess')
		else:
			if pop_PPV is None:
				raise ValueError('Need pop_PPV to plot random guess curve for precision_recall!')
			if inverse_log:
				x = np.linspace(1e-10, 1.)
				y = pop_PPV*np.ones(len(x))
				ax.plot(x, 1/y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')
			else:
				x = np.linspace(0., 1.)
				y = pop_PPV*np.ones(len(x))
				ax.plot(x, y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')

	leg = ax.legend(loc=leg_loc,frameon=False)
	leg.get_frame().set_facecolor('none')

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
		ax.set_xlim([0.,1.])
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
				plt.text(-0.07, -0.12, 'Better', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
			else:
				plt.text(-0.07, 1.08, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
		else:
			if inverse_log:
				plt.text(1.07, -0.12, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
			else:
				plt.text(1.07, 1.08, 'Better', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

	plt.tight_layout()
	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')
