import numpy as np
import Neural_Networks as NN

# change this to +5 once cos(MLAT) and sin(MLAT) are in NN
def get_predictor_counts(number_of_inputs):
    return 60*number_of_inputs+3


# given a list of layers, return a Neural network class
def get_NN(NN_layout, device='cpu'):
    NN_dict = {1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    try:
        NeuralNetwork = NN_dict[len(NN_layout)-2]
    except:
        raise KeyError("The amount of layers inputted is not available")

    model = NeuralNetwork(NN_layout).to(device)

    return model


def output_error(values_to_use, parameters, number_of_layers, final_test_error, file_to_output_to = 'test_errors.txt'):
    # This function outputs the results of kfold cross validation from create_neural_network to a text file

    counter = 0
    values_string = ''
    for value in values_to_use:
        values_string += value
    string = str('Inputs: ' + values_string + ' || Layers: ')
    for parameter in parameters:
        if counter % 2 == 0 and counter < 2 * number_of_layers - 2:
            string = string + str(len(parameter)) + '-'
        elif counter % 2 == 0 and counter == 2 * number_of_layers - 2:
            string = string + str(len(parameter))
        counter += 1

    # Output the properties and the test results of the NN to a file called test_errors.txt
    put_error_here = open(file_to_output_to, 'a')
    output = string + str(
        ' || ExMSE: ' + str(final_test_error[0]) + ' || EyMSE: ' + str(final_test_error[1]) + ' || EzMSE: '
        + str(final_test_error[2]) + ' || Total E MSE: ' + str(np.sum(final_test_error)) + '\n')
    put_error_here.write(output)

    return output