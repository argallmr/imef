import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import download_data as dd
import data_manipulation as dm
import argparse
import storage_objects
import datetime as dt


def main():
    # parser = argparse.ArgumentParser(
    #     description='PUT DESCRIPTION HERE'
    # )
    #
    # parser.add_argument('sc', type=str, help='Spacecraft Identifier. Eg:MMS1')
    #
    # parser.add_argument('mode', type=str, help='Data rate mode. Eg:srvy')
    #
    # parser.add_argument('level', type=str, help='Data level. Eg:l2')
    #
    # parser.add_argument('start_date', type=str, help='Start date of the data interval: "YYYY-MM-DDTHH:MM:SS"')
    #
    # parser.add_argument('end_date', type=str, help='End date of the data interval:  "YYYY-MM-DDTHH:MM:SS"')
    #
    # parser.add_argument('filename', type=str, help='Output file name. Do not include file extension')
    #
    # # If the polar plot is updated to spherical, update this note (and maybe change -p to -s)
    # parser.add_argument('-p', '--polar', help='Convert the electric field values to polar (default is cartesian)',
    #                     action='store_true')
    #
    # args = parser.parse_args()
    # IDK what args I will need, or how I want to set this up yet. Figure it out network of NFL

    # I think we train the model here, and we make another file that actually uses the model to predict.

    # Data that is needed: Index/Driver for 5 hours. L, cos(MLT), sin(MLT), and efield
    # So need EDI, MEC, and index/driver of choice(s)
    # Should I use the data gathered from sample_data for this? And just input a file this way we don't have to gather 6 years of data every time? actually probably yeah

    # For now I will train on everything. Though whether this should actually be done will have to be revisited
    # Should I train on other mms probes or just mms1?

    sc='mms1'
    mode='srvy'
    level='l2'
    start_date = dt.datetime(2015, 9, 10)
    end_date = dt.datetime(2015, 9, 11)

    # I think I should check with matt and see if I should use download parameters at all or if I should just leave them as arguments. It may keep it simpler for new people using the code
    #download_parameters = storage_objects.DownloadParameters('mms1', 'srvy', 'l2', start_date, end_date)

    # For now. This will end up being removed in favor of a file created by sample_data
    edi_data=dd.get_edi_data(sc, mode, level, start_date, end_date, binned=True)
    mec_data=dd.get_mec_data(sc, mode, level, start_date, end_date, binned=True)
    # Kp is for now, since it is what I have downloaded. This will probably be replaced since it only will have 2 values (Kp is updated every 3 hours, so 5 hours prior gives 2 values)
    kp_data=dd.get_kp_data(start_date, end_date, expand=edi_data['time'])

    # Do I need to make the times into timestamps?
    # Also I think the electric field will need to be converted to polar since it is being used for potential. Though this may also be able to be done at the end after approximation is done
    # Also should I use magnitude of E?

    # Make the data into a tensor for pytorch to use. We want the data to be in the format (Kp_t-60 ... Kp_t-1, L, cos(MLT), sin(MLT), E?) Should E be in this? I think so
    # I think the first couple data points will have to be ignored. (actually no they wont, we can just get the Kp values from before that. But for other indices this may not be possible.)

    # This doesn't include the electric field values rn. Because it is an array it causes the whole design matrix to break when changing to a tensor
    design_matrix_array = np.array([[]])
    for counter in range(60, len(edi_data['time'].values)):
        wanted_index_data = kp_data['Kp'].values[counter-60:counter-1]
        the_rest_of_the_data = np.array([mec_data['L'].values[counter], np.cos(mec_data['MLT'].values[counter]), np.sin(mec_data['MLT'].values[counter])])
        new_data_line = np.concatenate((wanted_index_data, the_rest_of_the_data), axis=None)
        if counter == 60:
            design_matrix_array = [new_data_line]
        else:
            design_matrix_array = np.concatenate((design_matrix_array, [new_data_line]), axis=0)

    # I guess there are some nan's in EDI data. Should troubleshoot that at some point
    # Also there's the whole 3 terms in E. Do I need 3 different models? Idk what to do here
    design_matrix = torch.from_numpy(design_matrix_array)

    print(f"Shape of tensor: {design_matrix.shape}")
    print(f"Datatype of tensor: {design_matrix.dtype}")
    print(f"Device tensor is stored on: {design_matrix.device}")



if __name__ == '__main__':
    main()