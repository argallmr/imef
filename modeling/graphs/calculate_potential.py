import xarray as xr
import numpy as np

# When printing out xr[variable].values, says print out every value
np.set_printoptions(threshold=np.inf)


def get_A(min_Lvalue, max_Lvalue):
    # E=AΦ
    # A=-⛛ (-1*Gradient)
    # In polar coordinates, A=(1/△r, 1/r△Θ), where r is the radius and Θ is the azimuthal angle
    # In this case, r=L, △r=△Θ=1

    # First we need to make the gradient operator. Since we have electric field data in bins labeled as 4.5, 5.5, ... and want potential values at integer values 1,2,...
    # We use the central difference operator to get potential values at integer values.

    # For example, E_r(L=6.5, MLT=12.5)=[Φ(7,13)+Φ(7,12)]/2 - [Φ(6,13)+Φ(6,12)]/2
    # Recall matrix multiplication rules to know how we can reverse engineer each row in A knowing the above
    # So for E_r, 1/2 should be where Φ(7,13)+Φ(7,12) is, and -1/2 should be where Φ(6,13)+Φ(6,12) is
    # For the example above, that row in A looks like [0.....-1/2,-1/2, 0....1/2, 1/2, 0...0].

    # For E_az, things are slightly different, E_az(L=6.5, MLT=12.5) = 1/L * 24/2π * [Φ(7,13)+Φ(6,13)]/2 - [Φ(7,12)+Φ(6,12)]/2
    # 1/L represents the 1/r△Θ in the gradient operator, and 24/2π is the conversion from radians to MLT
    # All of the rows follow the E_r and E_az examples, and as a result A has 4 values in each row

    # This runs assuming the E vector is organized like the following:
    # E=[E_r(L=0,MLT=0), E_az(0,0), E_r(0,1), E_az(0,1)...E_r(1,0), E_az(1,0)....]
    # This may be changed later, especially if a 3rd dimension is added

    # The edge case where MLT=23.5 must be treated separately, because it has to use MLT=23 and MLT=0 as its boundaries

    L_range = int(max_Lvalue - min_Lvalue + 1)
    A = np.zeros((2 * 24 * L_range, 24 * (L_range + 1)))

    # In order to index it nicely, we must subtract the minimum value from the max value, so we can start indexing at 0
    # As a result, L_counter does not always represent the actual L value
    # In this case, the real L value is calculated by adding L_counter by min_Lvalue
    matrix_value_r = .5
    for L_counter in range(L_range):
        # This only accounts for MLT values from 0.5 to 22.5. The value where MLT = 23.5 is an exception handled at the end
        for MLT_counter in range(0, 23):
            # Here is where we implement the A values that give E_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 1)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 24)] = matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 25)] = matrix_value_r

            # Here is where we implement the A values that give E_az at the same point that the above E_r was found
            matrix_value_az = .5 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter)] = -matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 24)] = -matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 1)] = matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 25)] = matrix_value_az

        # Where MLT=23.5 is implemented
        # E_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=23)] = -matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter)] = -matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=47)] = matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=24)] = matrix_value_r

        # E_az
        matrix_value_az = .5 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=23)] = -matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=47)] = -matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter)] = matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=24)] = matrix_value_az

    # Conversion factor between kV/Re and mV/m
    # The -1 comes from E=-⛛V. A=-⛛, therefore we need the -1 in front
    constant = -1 / 6.3712
    A *= constant

    return A


def get_A_row(L, MLT=0, other=0):
    return 48 * L + 2 * MLT + other


def get_A_col(L, MLT=0, other=0):
    return 24 * L + MLT + other


def get_C(min_Lvalue, max_Lvalue):
    # C is the laplacian, or the second derivative matrix. It is used to smooth the E=AΦ relation when solving the inverse problem
    # The overall procedure used to find A is used again here (reverse engineering the values of A), however there are more values per row
    # Also, the central difference operator is not used here, so there are no halving of values
    # For the example y=Cx: y(L=6, MLT=12) = x(L=5, 12 MLT)+x(L=7, 12 MLT)+x(L=6, 11 MLT)+x(L=6, 13 MLT)-4*x(L=6, 12 MLT)

    # Like A, the edge cases must be accounted for. While the MLT edge cases can be handled the same way as in A, there are now edge cases in L.
    # The L edge cases are handled by **ignoring the lower values apparently**
    # For example, if L=4 was the lowest L value measured, then y=x(L=5, 0 MLT)+x(L=4, 23 MLT)+x(L=4, 1 MLT)-4*x(L=4, 0 MLT)

    # But, because C is a square matrix, we can use a different, much easier method to create this matrix
    # From the above example, we know that every value down the diagonal is -4. So we can use np.diag(np.ones(dimension)) to make a square matrix with ones across the diagonal and 0 elsewhere
    # Multiplying that by -4 gives us all the -4 values we want.
    # We can use the same method to create a line of ones one above the diagonal by using np.diag(np.ones(dimension-1), 1)
    # The above method can be refactored to create a line of ones across any diagonal of the matrix
    # So we just create a couple lines of ones and add them all together to create the C matrix

    L_range = int(max_Lvalue - min_Lvalue + 1)
    MLT = 24

    # For the example y(L=6, MLT=12), this creates the -4*x(L=6, MLT=12)
    minusfour = -4 * np.diag(np.ones(MLT * (L_range + 1)))

    # These create the x(L=6, MLT=13) and x(L=6, MLT=11) respectively
    MLT_ones = np.diag(np.ones(MLT * (L_range + 1) - 1), 1)
    moreMLT_ones = np.diag(np.ones(MLT * (L_range + 1) - 1), -1)

    # These create the x(L=7, MLT=12) and x(L=5, MLT=12) respectively
    L_ones = np.diag(np.ones(MLT * (L_range + 1) - MLT), MLT)
    moreL_ones = np.diag(np.ones(MLT * (L_range + 1) - MLT), -MLT)

    # Add the ones matrices and create C
    C = minusfour + MLT_ones + moreMLT_ones + L_ones + moreL_ones

    # Nicely, this method handles the edge L cases for us, so we don't have to worry about those.
    # However we do need to handle the edge MLT cases, since both MLT=0 and MLT=23 are incorrect as is

    # This loop fixes all the edge cases except for the very first and very last row in C, as they are fixed differently than the rest
    for counter in range(1, L_range+1):
        # Fixes MLT=23
        C[MLT * counter - 1][MLT*counter] = 0
        C[MLT * counter - 1][MLT*(counter-1)] = 1

        # Fixes MLT=0 at the L value 1 higher than the above statement
        C[MLT * counter][MLT * counter - 1] = 0
        C[MLT * counter][MLT * (counter+1) - 1] = 1

    # Fixes the first row
    C[0][MLT-1] = 1
    # Fixes the last row
    C[MLT*(L_range+1)-1][MLT*L_range] = 1

    return C


def main():
    # Open data
    imef_data = xr.open_dataset('2years.nc')

    # Determine the L range that the data uses
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values

    # Find the number of bins relative to L and MLT
    # nL is the number of L values in E, not Φ. So there will be nL+1 in places. There are 6 L values in E, but 7 in Φ (As L is taken at values of 4.5, 5.5, etc in E, but 4, 5, etc in Φ)
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    # Get the electric field data and make them into vectors
    E_r = imef_data['E_mean'][:, :, 0].values.flatten()
    E_az = imef_data['E_mean'][:, :, 1].values.flatten()

    # Create the number of elements that the potential will have
    nElements = 24 * int(max_Lvalue - min_Lvalue + 1)
    E = np.zeros(2 * nElements)

    # Reformat E_r and E_az so that they are combined into 1 vector following the format
    # [E_r(L=4, MLT=0), E_az(L=4, MLT=0), E_r(L=4, MLT=1), E_az(L=4, MLT=1), ... E_r(L=5, MLT=0), E_az(L=5, MLT=0), ...]
    for index in range(0, nElements):
        E[2 * index] = E_r[index]
        E[2 * index + 1] = E_az[index]

    # Create the A matrix
    A = get_A(min_Lvalue, max_Lvalue)

    # Create the C matrix
    C = get_C(min_Lvalue, max_Lvalue)

    # Define the tradeoff parameter γ
    gamma = 2.51e-4

    # Solve the inverse problem according to the equation in Matsui 2004 and Korth 2002
    # V=(A^T * A + γ * C^T * C)^-1 * A^T * E
    V = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + gamma * np.dot(C.T, C)), A.T), E)
    V = V.reshape(nL+1, nMLT)


if __name__ == '__main__':
    main()
