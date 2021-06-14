import pandas as pd
from math import isnan
import numpy as np

def average_data_3d(new_data, created_file):
    # Where the average values will be stored
    averaged_edi_data = pd.DataFrame(columns=['L', 'MLT', 'MLAT', 'ex', 'ey', 'ez'])

    # Averaging incoming dataframe
    for L_counter in range(0, 30): # 10 is not the highest possible data value for L
        for MLT_counter in range(0, 24):
            for MLAT_counter in range(-90, 90, 10):
                the_values_to_be_averaged = new_data[
                    (new_data['L'] >= L_counter) & (new_data['L'] < L_counter + 1) &
                    (new_data['MLT'] >= MLT_counter) & (new_data['MLT'] < MLT_counter + 1) &
                    (new_data['MLAT'] >= MLAT_counter) & (new_data['MLAT'] < MLAT_counter + 10)]

                # Average values for 1 bin
                averages = the_values_to_be_averaged.mean(axis=0)
                EDI_X_avg = averages[0]
                EDI_Y_avg = averages[1]
                EDI_Z_avg = averages[2]

                # Convert to Polar

                #EDI_p_avg = np.sqrt(EDI_X_avg**2 + EDI_Y_avg**2 + EDI_Z_avg**2) # Same units, but may want to figure out what that is
                #EDI_theta_avg = np.arctan(EDI_Y_avg / EDI_X_avg)  # Units = Radians
                #EDI_phi_avg = np.arccos(EDI_Z_avg / EDI_p_avg)  # Also Radians

                #print(EDI_p_avg, EDI_theta_avg, EDI_phi_avg)

                # Inputting newly found averages into dataframe
                new_row = pd.DataFrame([[L_counter + .5, MLT_counter + .5, MLAT_counter+5, EDI_X_avg, EDI_Y_avg, EDI_Z_avg]],
                                       columns=['L', 'MLT', 'MLAT', 'ex', 'ey', 'ez'])

                if averaged_edi_data.empty:
                    # If there is no data in the dataframe, add the first row (even if edi averages are NaN)
                    averaged_edi_data = new_row
                else:
                    # For every value after the first row, combine the new row with the existing numbers
                    averaged_edi_data = pd.concat([averaged_edi_data, new_row], ignore_index=True, sort=False)

    # created_file must remain true for the rest of the run. So it must be returned to make sure it stays this way.
    # There is probably a better way to do this
    created_file = combine_new_data(averaged_edi_data, created_file)
    return created_file


def average_data_2d(new_data, created_file):
    # Where the average values will be stored
    averaged_edi_data = pd.DataFrame(columns=['L', 'MLT', 'er', 'etheta', 'ephi'])

    # Averaging incoming dataframe
    for L_counter in range(0, 30): # 10 is not the highest possible data value for L
        for MLT_counter in range(0, 24):
            the_values_to_be_averaged = new_data[
                (new_data['L'] >= L_counter) & (new_data['L'] < L_counter + 1) &
                (new_data['MLT'] >= MLT_counter) & (new_data['MLT'] < MLT_counter + 1)]

            # Average values for 1 bin
            averages = the_values_to_be_averaged.mean(axis=0)
            EDI_X_avg = averages[0]
            EDI_Y_avg = averages[1]
            EDI_Z_avg = averages[2]

            # Convert to Polar

            EDI_r_avg = np.sqrt(EDI_X_avg**2 + EDI_Y_avg**2 + EDI_Z_avg**2) # Same units, but may want to figure out what that is
            EDI_theta_avg = np.arctan(EDI_Y_avg / EDI_X_avg)  # Units = Radians. Bad?
            EDI_phi_avg = np.arccos(EDI_Z_avg / EDI_r_avg)  # Also Radians. Bad also?

            # Inputting newly found averages into dataframe
            new_row = pd.DataFrame([[L_counter + .5, MLT_counter + .5, EDI_r_avg, EDI_theta_avg, EDI_phi_avg]],
                                   columns=['L', 'MLT', 'er', 'etheta', 'ephi'])

            if averaged_edi_data.empty:
                # If there is no data in the dataframe, add the first row (even if edi averages are NaN)
                averaged_edi_data = new_row
            else:
                # For every value after the first row, combine the new row with the existing numbers
                averaged_edi_data = pd.concat([averaged_edi_data, new_row], ignore_index=True, sort=False)

    #
    created_file = combine_new_data(averaged_edi_data, created_file)
    return created_file


def combine_new_data(new_data, created_file):
    if not created_file:
        # If the total data file has not been created, create it with the incoming new data
        new_data.to_csv("binned.csv", index=False)
        created_file = True
    else:
        # If the data file has already been created, combine the new data with the existing data
        file_data = pd.read_csv("binned.csv")

        # Must identify the non-NaN values to combine together.
        for counter in range(len(file_data)):

            # If there is data at a bin in both the existing file and new dataframe, average the two existing values
            if not isnan(file_data.loc[counter][2]) and not isnan(new_data.loc[counter][2]):
                file_data.loc[counter] = (new_data.loc[counter] + file_data.loc[counter]) / 2

            # If there is data at a bin that is not already in the existing datafile,
            # supplant the NaN in the file with the new data
            elif isnan(file_data.loc[counter][2]) and not isnan(new_data.loc[counter][2]):
                file_data.loc[counter] = new_data.loc[counter]

        # Export the file with the updated values
        file_data.to_csv("binned.csv", index=False)

    return created_file