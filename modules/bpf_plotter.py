import numpy as np
import utility as ut
import  matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import re
import os
import tables
import copy

def random_color(as_str=True, alpha=0.5):
	rgb = [random.randint(0,255),
		   random.randint(0,255),
		   random.randint(0,255)]
	if as_str:
		return "rgba"+str(tuple(rgb+[alpha]))
	else:
		# Normalize & listify
		return list(np.array(rgb)/255) + [alpha]


class EnsemblePlotter:
	"""
	Description:
		Plots evolution of ensembles
	"""
	def __init__(self, fig_size = (10, 10), pt_size = 1, size_factor = 5, dpi = 300):
		self.fig_size = fig_size
		self.pt_size = pt_size
		self.size_factor = size_factor

		# generate figure for the plot
		self.fig = plt.figure(figsize = self.fig_size, dpi = dpi)

	@ut.timer
	def plot_weighted_ensembles_2D(self, ensembles, labels = None, weights = None, colors = None, file_path = 'ensemble_plot.png.',\
								   alpha = 0.5, log_size = False, weight_histogram = True, log_weight = False,\
								   extra_data = [], extra_plt_fns = [], extra_styles = [], extra_labels = [], extra_colors = [],\
								   max_particles = 1000):
		"""
		Description:
			Plots a 2D weighted ensemble
		Args:
			ensembles: ensemble to be plotted
			weights: weights of particles
			file_path: path where the plot is to be saved
			ax: axes of the plot
			color: color of the points plotted
			log_size: bool, True if ensemble member sizes are to determined according to their log weights
		"""
		# check the number of ensembles to be plotted
		if len(np.array(ensembles).shape) < 3:
			ensembles = [ensembles]
			if weights is not None:
				weights = [weights]
		l = len(ensembles)
		ensembles = np.array(ensembles)
		# set up unsupplied arguments
		if labels is None:
			labels = ['ensemble_{}'.format(i) for i in range(l)]
		if weights is None:
			weights = [[1.0]*len(e) for e in ensembles]
		if colors is None:
			colors = [random_color(as_str=False, alpha=alpha)]
		# normalize weights
		for k in range(l):
			weights[k] = np.array(weights[k])
			weights[k] /= weights[k].sum()
		
		# sort particles according to weights
		for k in range(l):
			idx = np.argsort(weights[k])
			weights[k] = weights[k][idx][::-1]
			ensembles[k] = ensembles[k][idx][::-1]

		log_weights = np.log(weights)
		log_weights_max = np.amax(log_weights)
		weights_max = np.amax(weights)
		self.fig.clf()
		plt.figure(self.fig.number)
		ax = plt.subplot2grid((self.size_factor*l, self.size_factor*l), (0, 0), rowspan = self.size_factor*l, colspan = (self.size_factor - 1)*l)
		# plot ensembles
		for k, ensemble in enumerate(ensembles):
			sz = np.ones(len(ensemble))
			if len(ensemble) > max_particles:
				ensembles = ensemble[:max_particles]
			# plot weighted points
			for i, _ in enumerate(ensemble):
				if log_size:
					sz[i] = self.pt_size * (log_weights_max / log_weights[k][i])
				else:
					sz[i] = self.pt_size * (weights[k][i] / weights_max)
			ax.scatter(ensemble[:, 0], ensemble[:, 1], s = sz, color = colors[k], label = labels[k] if i == 0 else None, alpha = alpha)
			# plot weight histogram if needed
			if weight_histogram:
				if log_weight:
					w = log_weights[k]
				else:
					w = weights[k]
				w = w[w > -1e300]
				h_ax = plt.subplot2grid((self.size_factor*l, self.size_factor*l), (self.size_factor*k, self.size_factor*l-2), rowspan = l, colspan = l)
				h_ax.hist(w, label = labels[k])
				h_ax.yaxis.tick_right()
				h_ax.legend()
		# plot extra data if needed
		for k, ed in enumerate(extra_data):
			ed = np.array(ed)
			if len(ed.shape) < 2:
				getattr(ax, extra_plt_fns[k])(ed[0], ed[1], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
			elif ed.shape[0] > 2:
				getattr(ax, extra_plt_fns[k])(ed[:, 0], ed[:, 1], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
			else:
				getattr(ax, extra_plt_fns[k])(ed[0, :], ed[1, :], color = extra_colors[k], label = extra_labels[k], **extra_styles[k])
		# save and clear figure for reuse
		ax.legend()
		plt.savefig(file_path)

	@ut.timer
	def stich(self, folder, img_prefix, pdf_path, clean_up = True, resolution = 300):
		pages = []
		imgs = []
		if folder.endswith('/'):
			folder = folder[:-1]
		for img in os.listdir(folder):
			if img.startswith(img_prefix):
				pages.append(int(re.findall(r'[0-9]+', img)[-1]))
				im_path = folder + '/' + img
				im = Image.open(im_path)
				rgb_im = Image.new('RGB', im.size, (255, 255, 255))  # white background
				rgb_im.paste(im, mask=im.split()[3])
				imgs.append(rgb_im)
				if clean_up:
					os.remove(im_path)
		imgs = [imgs[i] for i in np.array(pages).argsort()]
		print(imgs)
		imgs[0].save(pdf_path, "PDF", resolution = resolution, save_all = True, append_images = imgs[1:])

@ut.timer
def plot_ensemble_evol(db_path, hidden_path, time_factor=1,\
					   hidden_color = 'red', prior_mean_color = 'purple', posterior_mean_color = 'maroon',\
					   obs_inv = None, obs_inv_color = 'black',\
					   fig_size = (10, 10), pt_size = 1, size_factor = 5,\
					   dpi = 300, ens_colors = ['orange', 'green'], alpha = 0.5, pdf_resolution = 300):
	"""
	Description:
		Plots prior and posterior on a single page in a pdf
	"""
	hdf5 = tables.open_file(db_path, 'r')
	ep = EnsemblePlotter(fig_size = fig_size, pt_size = pt_size, size_factor = size_factor, dpi = dpi)
	try:
		os.mkdir('ensemble_plots')
	except:
		pass
	folder = 'ensemble_plots'
	#"""
	particle_count = len(getattr(hdf5.root.particles, 'time_' + str(0)).read().tolist())
	weights_prior = np.ones(particle_count)/particle_count
	observations = hdf5.root.observation.read().tolist()
	for t, observation in enumerate(observations):
		ens_pr = getattr(hdf5.root.particles, 'time_' + str(t*time_factor)).read().tolist()
		ens_po = ens_pr
		weights_posterior = np.array(getattr(hdf5.root.weights, 'time_' + str(t)).read().tolist()).reshape(particle_count,)
		file_path = folder + '/ensembles_{}.png'.format(t)
		prior_mean =  np.average(ens_pr, weights = weights_prior, axis = 0)
		posterior_mean =  np.average(ens_po, weights = weights_posterior, axis = 0)
		extra_data = [hidden_path[t], prior_mean, posterior_mean]
		extra_plt_fns = ['scatter', 'scatter', 'scatter']
		extra_styles = [{'marker': '$T$', 's': pt_size}, {'marker': '$\mu$', 's': pt_size}, {'marker': '$M$', 's': pt_size}]
		extra_labels = ['true state', 'prior mean', 'posterior mean']
		extra_colors = [hidden_color, prior_mean_color, posterior_mean_color]

		if obs_inv is not None:
			if isinstance(obs_inv, np.ndarray):
				obs_i = obs_inv[t]
			else:
				obs_i = observation
			extra_data.append(obs_i)
			extra_plt_fns.append('scatter')
			extra_styles.append({'marker': '$O$', 's': pt_size})
			extra_labels.append('inverse of observation')
			extra_colors.append(obs_inv_color)
		ep.plot_weighted_ensembles_2D(ensembles = [ens_pr, ens_po], weights=[weights_prior, weights_posterior], labels = ['prior', 'posterior'],\
									  colors = ens_colors, file_path = file_path, alpha = alpha, log_size = False,\
									  weight_histogram = True, log_weight = True, extra_data = extra_data,\
									  extra_plt_fns = extra_plt_fns, extra_styles = extra_styles, extra_labels = extra_labels,\
									  extra_colors = extra_colors)
		weights_prior = copy.deepcopy(weights_posterior)

	ep.stich(folder = folder, img_prefix = 'ensembles', pdf_path= os.path.dirname(db_path) + '/bpf_evolution.pdf',\
	 		clean_up = True, resolution = pdf_resolution)
	hdf5.close()
