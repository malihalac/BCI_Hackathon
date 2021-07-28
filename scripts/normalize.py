import csv
import os
import numpy as np
from scipy import stats
import statistics as st
import pandas as pd

#### Author: Mali Halac
'''
This script is to normalize the features at the session level.
'''


if __name__ == '__main__':

	directory = "/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/"

	# features.csv files to read (across sessions, individual, group)
	# we have to normalize them across these three different levels
	file_1 = directory + "P01_S1_diff_features.csv"		# session_1
	file_2 = directory + "P01_S2_diff_features.csv"		# session_2


	# read features.csv files across sessions
	f1_lines = []
	with open( file_1,'r' ) as inData:
		next(inData)			# skip the header row
		for line in inData:		# convert scientific notation (str) to float
			lines = list( map( float,   filter( None  , [ x for x in line.strip().split(',') ] )) )
			f1_lines.append(lines)

	f2_lines = []
	with open( file_2,'r' ) as inData:
		next(inData)			# skip the header row
		for line in inData:		# convert scientific notation (str) to float
			lines = list( map( float,   filter( None  , [ x for x in line.strip().split(',') ] )) )
			f2_lines.append(lines)


	# perform normalization
	# loop over each electrode
	i=0
	for f1_channel, f2_channel in zip(f1_lines,f2_lines):
		normalized_features = []
		
		# Delta Power
		all_delta = [f1_channel[1], f2_channel[1]]
		normalized_delta_1 = (all_delta[0] - st.mean(all_delta)) / st.stdev(all_delta)
		normalized_delta_2 = (all_delta[1] - st.mean(all_delta)) / st.stdev(all_delta)
		normalized_delta = [normalized_delta_1, normalized_delta_2]

		# Theta Power
		all_theta = [f1_channel[2], f2_channel[2]]
		normalized_theta_1 = (all_theta[0] - st.mean(all_theta)) / st.stdev(all_theta)
		normalized_theta_2 = (all_theta[1] - st.mean(all_theta)) / st.stdev(all_theta)
		normalized_theta = [normalized_theta_1, normalized_theta_2]

		# Alpha Power
		all_alpha = [f1_channel[3], f2_channel[3]]
		normalized_alpha_1 = (all_alpha[0] - st.mean(all_alpha)) / st.stdev(all_alpha)
		normalized_alpha_2 = (all_alpha[1] - st.mean(all_alpha)) / st.stdev(all_alpha)
		normalized_alpha = [normalized_alpha_1, normalized_alpha_2]

		# Beta Power
		all_beta = [f1_channel[4], f2_channel[4]]
		normalized_beta_1 = (all_beta[0] - st.mean(all_beta)) / st.stdev(all_beta)
		normalized_beta_2 = (all_beta[1] - st.mean(all_beta)) / st.stdev(all_beta)
		normalized_beta = [normalized_beta_1, normalized_beta_2]

		# Coefficient of variation
		all_cvar = [f1_channel[5], f2_channel[5]]
		normalized_cvar_1 = (all_cvar[0] - st.mean(all_cvar)) / st.stdev(all_cvar)
		normalized_cvar_2 = (all_cvar[1] - st.mean(all_cvar)) / st.stdev(all_cvar)
		normalized_cvar = [normalized_cvar_1, normalized_cvar_2]

		# Mean of vertex to vertex slope
		all_mver = [f1_channel[6], f2_channel[6]]
		normalized_mver_1 = (all_mver[0] - st.mean(all_mver)) / st.stdev(all_mver)
		normalized_mver_2 = (all_mver[1] - st.mean(all_mver)) / st.stdev(all_mver)
		normalized_mver = [normalized_mver_1, normalized_mver_2]

		# Variance of vertex to vertex slope
		all_vver = [f1_channel[7], f2_channel[7]]
		normalized_vver_1 = (all_vver[0] - st.mean(all_vver)) / st.stdev(all_vver)
		normalized_vver_2 = (all_vver[1] - st.mean(all_vver)) / st.stdev(all_vver)
		normalized_vver = [normalized_vver_1, normalized_vver_2]

		# Hjorth Parameters -- Activity
		all_act = [f1_channel[8], f2_channel[8]]
		normalized_act_1 = (all_act[0] - st.mean(all_act)) / st.stdev(all_act)
		normalized_act_2 = (all_act[1] - st.mean(all_act)) / st.stdev(all_act)
		normalized_act = [normalized_act_1, normalized_act_2]

		# Hjorth Parameters -- Mobility
		all_mob = [f1_channel[9], f2_channel[9]]
		normalized_mob_1 = (all_mob[0] - st.mean(all_mob)) / st.stdev(all_mob)
		normalized_mob_2 = (all_mob[1] - st.mean(all_mob)) / st.stdev(all_mob)
		normalized_mob = [normalized_mob_1, normalized_mob_2]

		# Hjorth Parameters -- Complexity
		all_com = [f1_channel[10], f2_channel[10]]
		normalized_com_1 = (all_com[0] - st.mean(all_com)) / st.stdev(all_com)
		normalized_com_2 = (all_com[1] - st.mean(all_com)) / st.stdev(all_com)
		normalized_com = [normalized_com_1, normalized_com_2]

		# Kurtosis
		all_kur = [f1_channel[11], f2_channel[11]]
		normalized_kur_1 = (all_kur[0] - st.mean(all_kur)) / st.stdev(all_kur)
		normalized_kur_2 = (all_kur[1] - st.mean(all_kur)) / st.stdev(all_kur)
		normalized_kur = [normalized_kur_1, normalized_kur_2]

		# Second Difference Mean
		all_sdm = [f1_channel[12], f2_channel[12]]
		normalized_sdm_1 = (all_sdm[0] - st.mean(all_sdm)) / st.stdev(all_sdm)
		normalized_sdm_2 = (all_sdm[1] - st.mean(all_sdm)) / st.stdev(all_sdm)
		normalized_sdm = [normalized_sdm_1, normalized_sdm_2]

		# Second Difference Max
		all_dmax = [f1_channel[13], f2_channel[13]]
		normalized_dmax_1 = (all_dmax[0] - st.mean(all_dmax)) / st.stdev(all_dmax)
		normalized_dmax_2 = (all_dmax[1] - st.mean(all_dmax)) / st.stdev(all_dmax)
		normalized_dmax = [normalized_dmax_1, normalized_dmax_2]

		# Skewness
		all_skew = [f1_channel[14], f2_channel[14]]
		normalized_skew_1 = (all_skew[0] - st.mean(all_skew)) / st.stdev(all_skew)
		normalized_skew_2 = (all_skew[1] - st.mean(all_skew)) / st.stdev(all_skew)
		normalized_skew = [normalized_skew_1, normalized_skew_2]

		# First Difference Mean
		all_fdm = [f1_channel[15], f2_channel[15]]
		normalized_fdm_1 = (all_fdm[0] - st.mean(all_fdm)) / st.stdev(all_fdm)
		normalized_fdm_2 = (all_fdm[1] - st.mean(all_fdm)) / st.stdev(all_fdm)
		normalized_fdm = [normalized_fdm_1, normalized_fdm_2]

		# First Difference Max
		all_difmax = [f1_channel[16], f2_channel[16]]
		normalized_difmax_1 = (all_difmax[0] - st.mean(all_difmax)) / st.stdev(all_difmax)
		normalized_difmax_2 = (all_difmax[1] - st.mean(all_difmax)) / st.stdev(all_difmax)
		normalized_difmax = [normalized_difmax_1, normalized_difmax_2]


		# save the normalized features
		# different .csv file for each electrode
		normalized_features_dir = directory + "normalized_features/" + str(i) + ".csv" 

		normalized_features_dict = {
					"Delta": normalized_delta,
					"Theta": normalized_theta,
					"Alpha": normalized_alpha,
					"Beta": normalized_beta,
					"Coefficient of variation": normalized_cvar,
					"Mean of vertex to vertex slope": normalized_mver,
					"Variance of vertex to vertex slope": normalized_vver,
					"Hjorth activity": normalized_act,
					"Hjorth mobility": normalized_mob,
					"Hjorth complexity": normalized_com,
					"Kurtosis": normalized_kur,
					"Second difference mean": normalized_sdm,
					"Second difference max": normalized_dmax,
					"Skewness": normalized_skew,
					"First difference mean": normalized_fdm,
					"First difference max": normalized_difmax
										}
		
		# index column stands for the session
		# S01 and S02
		df = pd.DataFrame.from_dict(normalized_features_dict)
		df.to_csv(normalized_features_dir, index=True)

		i+=1





