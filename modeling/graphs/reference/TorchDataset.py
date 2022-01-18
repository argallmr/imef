from torch.utils.data import Dataset
import numpy as np
import xarray as xr

class IMEFDataset(Dataset):
    def __init__(self, data_filename, mode):
        imef_data = xr.open_dataset(data_filename)

        imef_data = imef_data.where(np.isnan(imef_data['E_GSE'][:, 0]) == False, drop=True)

        if mode == 'x':
            coordinate_index = 0
        elif mode == 'y':
            coordinate_index = 1
        elif mode == 'z':
            coordinate_index = 2
        else:
            print('The mode must be \'x\', \'y\', or \'z\'')
            quit()

        for counter in range(60, len(imef_data['time'].values)):
            wanted_index_data = imef_data['Kp'].values[counter - 60:counter - 1].tolist()
            the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(imef_data['MLT'].values[counter]),
                                             np.sin(imef_data['MLT'].values[counter])]).tolist()
            efield_data = [imef_data['E_GSE'].values[counter, coordinate_index]]
            new_data_line = [wanted_index_data + the_rest_of_the_data, efield_data]
            if counter == 60:
                design_matrix_array = [new_data_line]
            else:
                design_matrix_array.append(new_data_line)

        # design_matrix = torch.tensor(design_matrix_array)
        self.samples=design_matrix_array

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]