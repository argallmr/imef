import datetime as dt
import numpy as np
import torch
import argparse
import imef.efield.model_creation.Neural_Networks as NN
import visualizations.plot_nc_data as xrplot
import imef.data.database as db


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    parser.add_argument('model_filename', type=str,
                        help='Name of the file containing the trained NN. Do not include file extension')

    parser.add_argument('time_to_predict', type=str,
                        help='The time that the user wants predict the electric field and electric potential for')

    args = parser.parse_args()

    model_filename = args.model_filename + '.pth'

    layers = model_filename.split('$')[0].split('-')
    values_to_use = model_filename.split('$')[1].split('-')

    # change this to be 183 when symh is involved
    if values_to_use[0] == 'All':
        NN_layout = np.array([123])
    else:
        NN_layout = np.array([60 * len(values_to_use) + 3])

    NN_layout = np.append(NN_layout, np.array(layers))
    NN_layout = np.append(NN_layout, np.array([3])).astype(int)
    number_of_layers = len(NN_layout) - 2

    NN_dict = {1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    try:
        NeuralNetwork = NN_dict[number_of_layers]
    except:
        raise KeyError("The amount of layers inputted is not available")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork(NN_layout).to(device)

    model.load_state_dict(torch.load(model_filename))

    time = dt.datetime.strptime(args.time_to_predict, '%Y-%m-%dT%H:%M:%S')

    imef_data, potential = db.predict_efield_and_potential(model, time=time, number_of_inputs=len(values_to_use))

    xrplot.plot_efield(imef_data, 'predicted_efield_polar_iLiMLT', mode='polar', count=False)

    xrplot.plot_potential(imef_data, potential)


if __name__ == '__main__':
    main()