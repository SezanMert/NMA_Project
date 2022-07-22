import numpy as np
import pandas as pd

# Load the data from the file
fname = 'motor_imagery.npz'
alldat = np.load(fname, allow_pickle=True)['dat']

# initialize the data frame with column names
df = pd.DataFrame(columns=['Subject', 'Trial', 'Brodmann Area', 'Movement Type', 'Re vs Im', 'V'])

# this is a variable for setting the index of every row of this data frame
idx = 0

# iterate through all subjects with this loop
for subject in np.arange(alldat.shape[0]):

    # iterate through real vs imagery with this loop
    for r_vs_i in np.arange(alldat.shape[1]):

        # iterate through all trials with this loop
        for trial_num in np.arange(len(alldat[subject][r_vs_i]['t_on'])):

            # iterate through all channels with this loop
            for ch_num in np.arange(len(alldat[subject][r_vs_i]['locs'])):

                # t_on is the start of cue onset of the specific trial indicated by "trial_num"
                t_on = alldat[subject][r_vs_i]['t_on'][trial_num]

                # t_off is the end of cue onset of the specific trial indicated by "trial_num"
                t_off = alldat[subject][r_vs_i]['t_off'][trial_num]

                # place gather the relevant information with the column order as a dictionary and place this dictionary
                # to the respective row (indexed by variable idx) of the data frame
                df.loc[idx] = {'Subject': subject,
                               'Trial': trial_num,
                               'Brodmann Area': alldat[subject][r_vs_i]['Brodmann_Area'][ch_num],
                               'No_BA': int(alldat[subject][r_vs_i]['Brodmann_Area'][ch_num][14:]),
                               'Movement Type': alldat[subject][r_vs_i]['stim_id'][trial_num],
                               'Re vs Im': r_vs_i,
                               'V': alldat[subject][r_vs_i]['V'][t_on:t_off][:, ch_num],
                               'scale_uv': alldat[subject][r_vs_i]['scale_uv'][ch_num]}

                # increment the index number to put the upcoming data to the next row
                idx += 1

# save the created pandas DataFrame as a pickle file to read it later
df.to_pickle("framed_data.pkl")

