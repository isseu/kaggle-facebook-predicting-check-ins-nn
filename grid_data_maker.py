from __future__ import division
import pandas as pd
import numpy as np

min_x = 0
max_x = 1
min_y = 0
max_y = 1

divisions = 5

print '[+] Reading data'
data = pd.read_csv('data/train.csv')
print '[+] Data readed, separating in areas'
for x_count in np.arange(0, divisions, 0.5):
	for y_count in np.arange(0, divisions, 0.5):
		print '\t[-] Area ' + str(x_count) + '-' + str(y_count)
		min_x_count = max_x / divisions * x_count
		max_x_count = max_x / divisions * ( x_count + 1)
		min_y_count = max_y / divisions * y_count
		max_y_count = max_y / divisions * ( y_count + 1)
		filter_data = data.query(
			'x >= {0} and x <= {1} and y >= {2} and y <= {3}'.format(min_x_count, max_x_count, min_y_count, max_y_count)
		)
		np.save('grid/data-{0}-{1}.npy'.format(x_count, y_count), filter_data.as_matrix())
print '[+] Process finish'