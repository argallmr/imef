import datetime as dt
import numpy as np
import torch
import argparse
import imef.efield.model_creation.NN_functions as NN_func
import visualizations.plot_nc_data as xrplot
import imef.data.database as db


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    parser.add_argument('model_filename', type=str,
                        help='Name of the file containing the trained NN. Do not include file extension')

    parser.add_argument('time_to_predict', type=str,
                        help='The time that the user wants predict the electric field and electric potential for. %Y-%m-%dT%H:%M:%S')

    args = parser.parse_args()

    model_filename = args.model_filename + '.pth'

    layers = model_filename.split('$')[0].split('-')
    values_to_use = model_filename.split('$')[1].split('-')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    NN_layout = np.array([NN_func.get_predictor_counts(len(values_to_use))])
    NN_layout = np.append(NN_layout, np.array(layers))
    NN_layout = np.append(NN_layout, np.array([3])).astype(int)

    model = NN_func.get_NN(NN_layout, device=device)

    model.load_state_dict(torch.load(model_filename))

    time = dt.datetime.strptime(args.time_to_predict, '%Y-%m-%dT%H:%M:%S')

    imef_data, potential = db.predict_efield_and_potential(model, time=time, values_to_use=values_to_use)

    xrplot.plot_efield(imef_data, 'predicted_efield_polar_iLiMLT', mode='polar', count=False)

    xrplot.plot_potential(imef_data, potential, vmin=-120, vmax=120)


if __name__ == '__main__':
    main()