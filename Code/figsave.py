import os
import matplotlib.pyplot as plt

def save_fig(image_path,
		fig_name,
		bbox_inches ='tight', 
		fig_extension ='png', 
		reselution = 800,
		verbose = True
		):
	""" a function to save fig
		image_path - path
		fig_name - name of img
		tight_layout - automatically adjust subplot parameters to give specified padding
		fig_extension - default 'png'
		reselution - default 300
		verbose - print fig_name
	"""
	path = os.path.join(image_path,fig_name + '.' + fig_extension)
	if verbose:
		print('Saving figture', fig_name)
	# if tight_layout:
	# 	plt.tight_layout()
	plt.savefig(path,dpi=reselution,bbox_inches = bbox_inches)