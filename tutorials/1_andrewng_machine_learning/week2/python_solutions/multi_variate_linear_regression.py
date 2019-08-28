import numpy as np
from matplotlib import pyplot
from input_file_operations import *
from linear_regression_operations import *
from plot_operations import *


def main():
    # read the input file
    ip_array = read_input_file("ex1data2.txt")

    # separate input file into training set input and output
    (ip_x_array, ip_y_array) = prepare_input_matrices(ip_array)
    (training_set_sample_count, training_set_feature_count) = ip_x_array.shape

    # perform feature normalization
    (norm_x_array, mean_x_array, range_x_array) = perform_feature_normalization(ip_x_array)

    # create the design array (X) and the model parameter array (theta)
    ip_design_array = create_design_array(norm_x_array)
    model_parameter_array = np.zeros((training_set_feature_count + 1, 1))

    # compute the model output and cost with initial guess for theta
    init_model_output_array = compute_model_output_array(ip_design_array, model_parameter_array)
    init_cost = compute_cost_function_square_mean_error(ip_design_array, model_parameter_array, ip_y_array)
    # print("initial theta: \n", model_parameter_array)
    # print("initial cost: ", init_cost)

    # compute the model parameters using gradient descent
    learning_rate = 0.01
    max_iter = 20000
    (model_parameter_array_gradient_descent, gradient_descent_cost_array, iter_count) = perform_gradient_descent(
        ip_design_array, model_parameter_array, ip_y_array, learning_rate, init_cost, max_iter)

    print("theta after gradient descent: \n", model_parameter_array_gradient_descent)
    print("cost: ", gradient_descent_cost_array[iter_count - 1])
    print("iter count: ", iter_count)

    # compute the model parameters using the normal equation
    # the normal equation does not require feature normalization
    ip_design_array = create_design_array(ip_x_array)
    (model_parameter_array_normal_equation, normal_equation_cost) = perform_normal_equation(ip_design_array, ip_y_array)

    print("theta after normal equation: \n", model_parameter_array_normal_equation)
    print("cost: ", normal_equation_cost)

    # create some plots
    # plot the input data as a scatter plot
    plot_3d_scatterplot(ip_x_array[:,0]/1000,ip_x_array[:,1],ip_y_array[:,0]/100000,"jet","input data","area (x1000 sq ft)","rooms","price(x100k)")

    # plot the variation of the cost function with each iteration
    plot_1d_scatterplot(range(0, max_iter), gradient_descent_cost_array[0:max_iter]*(10**-9), "Cost during gradient descent",
                        "gradient descent iteration", "mean square error cost (x 10^9)",
                        [0, 20000, 0, 35], "r-", 2)




if __name__ == "__main__":
    main()