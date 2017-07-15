import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import data
from skimage import filters
from skimage import img_as_float
from skimage import color
import scipy
from scipy import fftpack
import matplotlib as mpl
import imageio
import warnings

import bob.ip.facelandmarks as menpo
from bob.ip.facelandmarks.utils import detect_landmarks, draw_landmarks, save_landmarks, Result
from bob.ip.facelandmarks.utils import detect_landmarks_on_boundingbox
from align_image_code import align_images


warnings.filterwarnings("ignore")
#start_image_path = 'kunal_dude/kunal.jpg'
#end_image_path = 'kunal_dude/bleacher_dude.jpg'
#gif_name = 'kunal_dude/kunal_dude.gif'
all_frames = True
ts = np.arange(0,1.01,1./20)[::-1]
images = []
kargs = { 'duration': 0.1 }
left_eye_index = 36
right_eye_index = 45

def save_inputs_to_file(start_inputs, end_inputs):
	with open(start_inputs_file, 'wu') as out:
		csv_out=csv.writer(out)
		for row in start_inputs:
			csv_out.writerow(row)

	with open(end_inputs_file, 'wb') as out:
		csv_out=csv.writer(out)
		for row in end_inputs:
			csv_out.writerow(row)


def load_inputs():
	start_inputs = []
	end_inputs = []
	with open(start_inputs_file, 'rU') as out:
		csv_out=csv.reader(out)
		for row in csv_out:
			start_inputs.append((float(row[0]), float(row[1])))

	with open(end_inputs_file, 'rU') as out:
		csv_out=csv.reader(out)
		for row in csv_out:
			end_inputs.append((float(row[0]), float(row[1])))
	return start_inputs, end_inputs

def convert_to_matrix_form(a):
	return np.append(a.transpose(), [1], axis=0)

def get_points(s, e, t):
	t_points = []
	for i in range(len(s)):
		avg_x = t * s[i][0] + (1-t) * e[i][0]
		avg_y = t * s[i][1] + (1-t) * e[i][1]
		t_points.append((avg_x, avg_y))
	return t_points

def compute_affine_matrices(start_tri, tri_pts, end_pts):
	simplices = start_tri.simplices
	end_tri_pts = end_pts[simplices]
	start_tri_pts = tri_pts[simplices]
	end_tri_pts = np.apply_along_axis(convert_to_matrix_form, 2, end_tri_pts)
	start_tri_pts = np.apply_along_axis(convert_to_matrix_form, 2, start_tri_pts)
	
	matrices = np.zeros((start_tri_pts.shape[0],3,3))
	start_tri_pts[:] = np.transpose(start_tri_pts[:], (0,2,1))

	end_tri_pts[:] = np.transpose(end_tri_pts[:], (0,2,1))

	for i in range(start_tri_pts.shape[0]):
		end = end_tri_pts[i]
		start = start_tri_pts[i]
		matrices[i] = np.dot(end,np.linalg.inv(start))

	return matrices

def get_simplices_in_img(start_aligned, avg_tri):
	start_r = start_aligned[:,:,0]
	start_g = start_aligned[:,:,1]
	start_b = start_aligned[:,:,2]

	coords = np.mgrid[0:start_r.shape[0], 0:start_r.shape[1]].reshape(2,-1).T
	coords = np.reshape(coords, (start_r.shape[0], start_r.shape[1], 2))
	coords = np.swapaxes(np.swapaxes(coords, 0, 2)[::-1], 0, 2)

	simplices = scipy.spatial.tsearch(avg_tri, coords)
	return coords, simplices

def get_coords_in_img(mid_coords, simplices, matrices):
	new_coords = np.zeros((mid_coords.shape), dtype=(float,1))
	count = 0
	total = 0
	for i in range(mid_coords.shape[0]):
		for j in range(mid_coords.shape[1]):
			simplex = simplices[i][j]
			if simplex > -1:
				matrix = matrices[simplex]
				coords = mid_coords[i][j]
				coords = np.append(coords, [1], axis=0)
				new_coords[i][j] = np.dot(matrix, coords)[:2]	
				count += 1
			else:
				new_coords[i][j] = mid_coords[i][j]
			total += 1
	return new_coords


def generate_mid_way(start_aligned, start_inputs, end_aligned, end_inputs, t):
	avg_points = np.array(get_points(start_inputs, end_inputs, t))
	avg_tri = scipy.spatial.Delaunay(avg_points)
	mid_to_start_matrices = compute_affine_matrices(avg_tri, avg_points, np.array(start_inputs))
	mid_to_end_matrices = compute_affine_matrices(avg_tri, avg_points, np.array(end_inputs))
	start_r = start_aligned[:,:,0]
	start_g = start_aligned[:,:,1]
	start_b = start_aligned[:,:,2]
	end_r = end_aligned[:,:,0]
	end_g = end_aligned[:,:,1]
	end_b = end_aligned[:,:,2]

	# Doing it for the start image only
	mid_coords, simplices = get_simplices_in_img(start_aligned, avg_tri)
	s_coords = get_coords_in_img(mid_coords, simplices, mid_to_start_matrices)
	e_coords = get_coords_in_img(mid_coords, simplices, mid_to_end_matrices)


	s_pix_r = np.reshape(scipy.ndimage.interpolation.map_coordinates(start_r, [np.ravel(s_coords[:,:,1]), np.ravel(s_coords[:,:,0])]), start_r.shape)
	s_pix_g = np.reshape(scipy.ndimage.interpolation.map_coordinates(start_g, [np.ravel(s_coords[:,:,1]), np.ravel(s_coords[:,:,0])]), start_g.shape)
	s_pix_b = np.reshape(scipy.ndimage.interpolation.map_coordinates(start_b, [np.ravel(s_coords[:,:,1]), np.ravel(s_coords[:,:,0])]), start_b.shape)
	e_pix_r = np.reshape(scipy.ndimage.interpolation.map_coordinates(end_r, [np.ravel(e_coords[:,:,1]), np.ravel(e_coords[:,:,0])]), end_r.shape)
	e_pix_g = np.reshape(scipy.ndimage.interpolation.map_coordinates(end_g, [np.ravel(e_coords[:,:,1]), np.ravel(e_coords[:,:,0])]), end_g.shape)
	e_pix_b = np.reshape(scipy.ndimage.interpolation.map_coordinates(end_b, [np.ravel(e_coords[:,:,1]), np.ravel(e_coords[:,:,0])]), end_b.shape)

	return np.dstack([s_pix_r, s_pix_g, s_pix_b]), np.dstack([e_pix_r, e_pix_g, e_pix_b])

def add_edges(inputs, img):
	inputs.append((0,0))
	inputs.append((0,img.shape[1]-1))
	inputs.append((img.shape[0]-1, 0))
	inputs.append((img.shape[0]-1, img.shape[1]-1))

	return inputs

def morpher_main(start_image_path, end_image_path,  gif_name):	
	start_image = mpl.pyplot.imread(start_image_path)
	start_gray = sk.color.rgb2gray(start_image)

	end_image = mpl.pyplot.imread(end_image_path)
	end_gray = sk.color.rgb2gray(end_image)

	start_keypoints = menpo.utils.detect_landmarks(start_gray, 1)
	end_keypoints = menpo.utils.detect_landmarks(end_gray, 1)

	start_landmarks = start_keypoints[0].landmarks
	end_landmarks = end_keypoints[0].landmarks

	

	start_left_eye = (start_landmarks[left_eye_index][1], start_landmarks[left_eye_index][0])
	start_right_eye = (start_landmarks[right_eye_index][1], start_landmarks[right_eye_index][0])
	end_left_eye = (end_landmarks[left_eye_index][1], end_landmarks[left_eye_index][0])
	end_right_eye = (end_landmarks[right_eye_index][1], end_landmarks[right_eye_index][0])

	start_aligned, end_aligned = align_images(start_image, end_image, (start_left_eye, start_right_eye, end_left_eye, end_right_eye))
	start_aligned = img_as_float(start_aligned)
	end_aligned = img_as_float(end_aligned)

	start_aligned_gray = sk.color.rgb2gray(start_aligned)
	end_aligned_gray = sk.color.rgb2gray(end_aligned)

	start_keypoints = menpo.utils.detect_landmarks(start_aligned_gray, 1)
	end_keypoints = menpo.utils.detect_landmarks(end_aligned_gray, 1)

	start_inputs = np.flip(np.array(start_keypoints[0].landmarks),1)
	end_inputs = np.flip(np.array(end_keypoints[0].landmarks),1)

	for i in range(len(ts)):
		t = ts[i]
		print i
		mid_way_s, mid_way_e = generate_mid_way(start_aligned, start_inputs, end_aligned, end_inputs, t)
		final_mid = t * mid_way_s + (1-t) * mid_way_e
		images.append(final_mid)

	images.extend(images[::-1])
	imageio.mimsave(gif_name, images, 'GIF', **kargs)










