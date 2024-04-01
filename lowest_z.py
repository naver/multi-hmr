import numpy as np

file_path = "/mnt/mnt_0/sjr/work/multi-hmr/output/output_060/camera_rear/1710308670513070.npy"

data = np.load(file_path).reshape((-1, 3))

x_min_idx = np.argmin(data[:, 0])
x_max_idx = np.argmax(data[:, 0])
y_min_idx = np.argmin(data[:, 1])
y_max_idx = np.argmax(data[:, 1])
z_min_idx = np.argmin(data[:, 2])
z_max_idx = np.argmax(data[:, 2])
x_min = data[x_min_idx, 0]
x_max = data[x_max_idx, 0]
y_min = data[y_min_idx, 1]
y_max = data[y_max_idx, 1]
z_min = data[z_min_idx, 2]
z_max = data[z_max_idx, 2]



# print(x_max - x_min, y_max - y_min, z_max - z_min)
print(y_max)