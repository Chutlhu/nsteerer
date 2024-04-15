import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from collections import namedtuple
import src.utils.geo_utils as geo

# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, mic_pos, mic_orV, mic_pattern, center, n_mics')
SourceSetup = namedtuple('SourceSetup', 'doa, signal, path_to_signal, fs')

def get_echo_array(n_mics, mic_radius, arr_center=None):

	if arr_center is None:
		arr_center = np.zeros([3,1])

	R_3M = pra.circular_2D_array(arr_center[:2, 0], n_mics, 0, mic_radius)
	R_3M = np.concatenate((R_3M, np.ones((1, n_mics)) * arr_center[2, 0]), axis=0)

	return ArraySetup(arrayType='circular',
		orV = None,
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = arr_center,
		n_mics = n_mics
	) 


def get_linear_array(n_mics, spacing, arr_center=None):
	if arr_center is None:
		arr_center = np.zeros([3,1])

	R_3M = pra.linear_2D_array(arr_center[:2, 0], n_mics, 0, spacing)
	R_3M = np.concatenate((R_3M, np.ones((1, n_mics)) * arr_center[2, 0]), axis=0)

	return ArraySetup(arrayType='linear',
		orV = None,
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = arr_center,
		n_mics = n_mics
	) 

def get_easycom_array(arr_center=None, binaural_mic=True):
	if arr_center is None:
		arr_center = np.zeros([3,1])
	arr_center = geo.check_geo_dim_convention(arr_center)

	# Channel/Mic 	# 	X (mm) 	Y (mm) 	Z (mm)
	# 				1 	 82		-5 		-29
	# 				2 	-01		-1	 	 30
	#				3 	-77 	-2 		 11
	# 				4 	-83 	-5 		-60
	# 				5	N/A 	N/A 	N/A
	# 				6	N/A 	N/A 	N/A
	R_3M = np.array([
		[ 0.082, -0.005, -0.029],
		[-0.001, -0.001,  0.030],
		[-0.077, -0.002,  0.011],
		[-0.083, -0.005, -0.060],
		[ 0.052, -0.010, -0.060],
		[-0.053, -0.005, -0.060],
	]).T
	# use my usual convention (left hand)
	R_3M = R_3M[[2,0,1],:]
	# R_3M = R_3M[[1,0,2],:]
	# R_3M[1,:] *= -1
	
	if binaural_mic: 
		n_mics = 6
	else:
		n_mics = 4
		R_3M = R_3M[:,:4]
 

	P_3M = R_3M + arr_center
		
	assert R_3M.shape == (3, n_mics)

	return ArraySetup(
		arrayType='easycom',
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = arr_center,
		n_mics = n_mics,
	)

def get_oculus_array(arr_center=None):
	if arr_center is None:
		arr_center = np.zeros([3,1])
	arr_center = geo.check_geo_dim_convention(arr_center)

	n_mics = 5
	R_3M = np.array([
		[-0.016, -0.055,  0.],
		[-0.016, 0.055,  0.],
		[0., 0., 0.],
		[-0.016, -0.021, -0.085],
		[-0.016, 0.039, -0.085]
		]).T
		
	P_3M = R_3M + arr_center
		
	assert R_3M.shape == (3, n_mics)

	return ArraySetup(
		arrayType='oculus2',
		mic_pos = R_3M,
		mic_orV = None,
		mic_pattern = 'omni',
		center = arr_center,
		n_mics = n_mics,
	)

# echo_array_setup = get_echo_array(5, 0.04)
# linear_array_setup = get_linear_array(5, 0.04)