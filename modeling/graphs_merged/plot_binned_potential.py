import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse

def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')

def get_A(min_Lvalue, max_Lvalue):
    # E=AΦ
    # A=-⛛ (-1*Gradient)
    # In polar coordinates, A=(1/△r, 1/r△Θ), where r is the radius and Θ is the azimuthal angle
    # In this case, r=L, △r=△Θ=1

    # First we need to make the gradient operator. Since we have electric field data in bins labeled as 4.5, 5.5, ... and want potential values at integer values 1,2,...
    # We use the central difference operator to get potential values at integer values.

    # For example, E_r(L=6.5, MLT=12.5)=Φ(7,13)+Φ(7,12) - Φ(6,13)+Φ(6,12)
    # Recall matrix multiplication rules to know how we can reverse engineer each row in A knowing the above
    # So for E_r, 1 should be where Φ(7,13)+Φ(7,12) is, and -1 should be where Φ(6,13)+Φ(6,12) is
    # For the example above, that row in A looks like [0.....-1,-1, 0....1, 1, 0...0].

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
    matrix_value_r = 1
    for L_counter in range(L_range):
        # This only accounts for MLT values from 0.5 to 22.5. The value where MLT = 23.5 is an exception handled at the end
        for MLT_counter in range(0, 23):
            # Here is where we implement the A values that give E_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 1)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 24)] = matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 25)] = matrix_value_r

            # Here is where we implement the A values that give E_az at the same point that the above E_r was found
            matrix_value_az = 1 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
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
        matrix_value_az = 1 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
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
    # C is the hessian, or the second derivative matrix. It is used to smooth the E=AΦ relation when solving the inverse problem
    # The overall procedure used to find A is used again here (reverse engineering the values of A), however there are more values per row
    # Also, the central difference operator is not used here, so there are no halving of values
    # For the example y=Cx: y(L=6, MLT=12) = x(L=5, 12 MLT)+x(L=7, 12 MLT)+x(L=6, 11 MLT)+x(L=6, 13 MLT)-4*x(L=6, 12 MLT)

    # Like A, the edge cases must be accounted for. While the MLT edge cases can be handled the same way as in A, there are now edge cases in L.
    # The L edge cases are handled by **ignoring the lower values apparently**
    # For example, if L=4 was the lowest L value measured, then y=x(L=5, 0 MLT)+x(L=4, 23 MLT)+x(L=4, 1 MLT)-4*x(L=4, 0 MLT)

    # But, because C is a square matrix, we can use a different, much easier method to create this matrix than we did with A
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


def E_corot(r):
    # E_corot = C_corot*R_E/r^2 * 𝜙_hat
    # C_corot is found here
    omega_E = 2 * np.pi / (24 * 3600)  # angular velocity of Earth (rad/sec)
    B_0 = 3.12e4  # Earth mean field at surface (nT)
    R_E = 6371.2  # Earth radius (km)
    C_corot = omega_E * B_0 * R_E ** 2 * 1e-3  # V (nT -> T 1e-9, km**2 -> (m 1e3)**2)

    # Corotation Electric Field
    #  - Azimuthal component in the equatorial plane
    E_cor = (-C_corot * R_E / r.loc[:, 'r'] ** 2)
    E_cor = np.stack([E_cor,
                      np.zeros(len(E_cor)),
                      np.zeros(len(E_cor))], axis=1)
    E_cor = xr.DataArray(E_cor,
                         dims=['time', 'cyl'],
                         coords={'time': r['time'],
                                 'cyl': ['r', 'phi', 'z']})

    return E_cor


def cyl2cart(r_cyl):
    x = r_cyl.loc[:, 'r'] * np.cos(r_cyl.loc[:, 'phi'])
    y = r_cyl.loc[:, 'r'] * np.sin(r_cyl.loc[:, 'phi'])
    z = r_cyl.loc[:, 'z'].drop('cyl')

    # Combine into a vector
    #   - Convert r from an ndarray to a DataArray
    r_cart = (xr.concat([x, y, z], dim='cyl')
              .T.rename({'cyl': 'cart'})
              .assign_coords({'time': r_cyl['time'],
                              'cart': ['x', 'y', 'z']})
              )

    return r_cart


def xform_cyl2cart(r):
    # Unit vectors
    x_hat = np.stack([np.cos(r.loc[:, 'phi']),
                      -np.sin(r.loc[:, 'phi']),
                      np.zeros(len(r))], axis=1)
    y_hat = np.stack([np.sin(r.loc[:, 'phi']),
                      np.cos(r.loc[:, 'phi']),
                      np.zeros(len(r))], axis=1)
    z_hat = np.repeat(np.array([[0, 0, 1]]), len(r), axis=0)

    xcyl2cart = xr.DataArray(np.stack([x_hat, y_hat, z_hat], axis=2),
                             dims=('time', 'cyl', 'cart'),
                             coords={'time': r['time'],
                                     'cyl': ['r', 'phi', 'z'],
                                     'cart': ['x', 'y', 'z']})

    return xcyl2cart.transpose('time', 'cart', 'cyl')


def xform_cart2cyl(r):
    # Unit vectors
    phi = np.arctan2(r.loc[:, 'y'], r.loc[:, 'x'])
    r_hat = np.stack([np.cos(phi), np.sin(phi), np.zeros(len(r))], axis=1)
    phi_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros(len(r))], axis=1)
    z_hat = np.repeat(np.array([[0, 0, 1]]), len(r), axis=0)

    xcart2cyl = xr.DataArray(np.stack([r_hat, phi_hat, z_hat], axis=2),
                             dims=('time', 'cart', 'cyl'),
                             coords={'time': r['time'],
                                     'cart': ['x', 'y', 'z'],
                                     'cyl': ['r', 'phi', 'z']})

    return xcart2cyl.transpose('time', 'cyl', 'cart')


def calculate_potential(imef_data, name_of_variable):
    # Determine the L range that the data uses
    min_Lvalue = 3
    max_Lvalue = 9

    # Find the number of bins relative to L and MLT
    # nL is the number of L values in E, not Φ. So there will be nL+1 in places. There are 6 L values in E, but 7 in Φ (As L is taken at values of 4.5, 5.5, etc in E, but 4, 5, etc in Φ)
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    # Get the electric field data and make them into vectors. MUST BE POLAR COORDINATES
    E_r = imef_data[name_of_variable][:, :, 0].values.flatten()
    E_az = imef_data[name_of_variable][:, :, 1].values.flatten()

    # Create the number of elements that the potential will have
    nElements = 24 * nL
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
    V = V.reshape(nL + 1, nMLT)

    return V


def plot_potential(V_data, otherL, otherMLT):
    # Note that it is expected that the electric field data is in polar coordinates. Otherwise the potential values are incorrect

    # find L and MLT range used in the data given
    min_Lvalue = 3
    max_Lvalue = 9
    nL = int(max_Lvalue - min_Lvalue + 1)

    min_MLTvalue = 0
    max_MLTvalue = 23
    nMLT = int(max_MLTvalue - min_MLTvalue + 1)

    # Create a coordinate grid
    new_values = otherMLT.values-1
    phi = (2 * np.pi * new_values / 24).reshape(nL, nMLT)
    r = otherL.values.reshape(nL, nMLT)-1

    extra_phi_value = phi[0][0]+2*np.pi

    # The plot comes out missing a section since the coordinates do not completely go around the circle.
    # So we have to copy/paste the first plot point to the end of each of the lists so that the plot is complete
    for counter in range(nL):
        add_to_r = np.append(r[counter], r[counter][0])
        add_to_phi = np.append(phi[0], extra_phi_value)
        add_to_V_data = np.append(V_data[counter], V_data[counter][0])
        if counter==0:
            new_r = [add_to_r]
            new_phi = [add_to_phi]
            new_V_data = [add_to_V_data]
        else:
            new_r = np.append(new_r, [add_to_r], axis=0)
            new_phi= np.append(new_phi, [add_to_phi], axis=0)
            new_V_data = np.append(new_V_data, [add_to_V_data], axis=0)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.set_xlabel("Potential")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    # Plot the data. Note that new_V_data is multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction of normal cartesian coordinates
    im = ax1.contourf(new_phi, new_r, new_V_data*-1, cmap='coolwarm', vmin=-5, vmax=5)
    # plt.clabel(im, inline=True, fontsize=8)
    # plt.imshow(new_V_data, extent=[-40, 12, 0, 10], cmap='RdGy', alpha=0.5)
    fig.colorbar(im, ax=ax1)
    # Draw the earth
    draw_earth(ax1)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='This program takes a file of binned electric field values, calculates the convection electric field, then calculates the electric potential of that electric field and plots it'
    )

    parser.add_argument('input_filename', type=str, help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                             'Do not include file extension')

    args = parser.parse_args()

    input_filename = args.input_filename + '.nc'
    # Open file
    data = xr.open_dataset(input_filename)

    # Size of the bins
    dr = 1
    dtheta = 1

    theta = np.arange(0, 24 + dtheta, dtheta) * 2 * np.pi / 24
    r = np.arange(3, 10 + dr, dr)

    # make a grid of values that will convert the electric field from cartesian to polar (cylindrical) coordinates
    ngrid = len(theta) * len(r)
    theta_grid, r_grid = np.meshgrid(theta, r)
    z_grid = np.zeros(ngrid)

    cyl_grid = xr.DataArray(np.stack([r_grid.flatten(), theta_grid.flatten(), z_grid], axis=1),
                            dims=('time', 'cyl'),
                            coords={'time': np.arange(ngrid),
                                    'cyl': ['r', 'phi', 'z']}
                            )
    cart_grid = cyl2cart(cyl_grid)
    xcart2cyl = xform_cart2cyl(cart_grid)

    # calculate the convective electric field
    E_convection = data['E_EDI'] - data['E_con'] - data['E_cor']

    # reshape the data so that it will work with the conversion function
    E_convection_reshaped = xr.DataArray(E_convection.values.reshape(168, 3), dims=['time', 'cart'],coords={'time': np.arange(168), 'cart': data['cart']})

    # convert to polar
    E_convection_polar = xcart2cyl.dot(E_convection_reshaped, dims='cart')

    # reshape and store the polar data in a new dataset
    imef_data = xr.Dataset()
    data_polar_split = xr.DataArray(E_convection_polar.values.reshape(7, 24, 3), dims=['r', 'theta', 'cart'],
                             coords={'r': data['r'], 'theta':data['theta']*(12/np.pi),  'cart': data['cart']})
    imef_data['E_convection_polar'] = data_polar_split
    imef_data = imef_data.rename({'r': 'L', 'theta': 'MLT'})
    variable_name = 'E_convection_polar'

    otherL, otherMLT = xr.broadcast(imef_data['L'], imef_data['MLT'])

    imef_data['E_convection_polar'] = imef_data['E_convection_polar'].fillna(0)
    # print(imef_data['E_convection_polar'].values[1])

    # calculate the electric potential and plot
    V = calculate_potential(imef_data, variable_name)
    plot_potential(V, otherL, otherMLT)


if __name__ == '__main__':
    main()