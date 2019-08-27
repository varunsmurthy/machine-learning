import numpy as np
from matplotlib import pyplot
from input_file_operations import *
from linear_regression_operations import *
from plot_operations import *


def main():
    # read the input file
    ip_array = read_input_file("ex1data1.txt")

    # separate input file into training set input and output
    (ip_x_array, ip_y_array) = prepare_input_matrices(ip_array)
    (training_set_sample_count,training_set_feature_count) = ip_x_array.shape

    # perform feature normalization
    (norm_x_array, mean_x_array, range_x_array) = perform_feature_normalization(ip_x_array)

    # create the design array (X) and the model parameter array (theta)
    ip_design_array = create_design_array(norm_x_array)
    model_parameter_array = np.zeros((training_set_feature_count+1,1))

    # compute the model output and cost with initial guess for theta
    init_model_output_array = compute_model_output_array(ip_design_array, model_parameter_array)
    init_cost = compute_cost_function_square_mean_error(ip_design_array, model_parameter_array, ip_y_array)
    # print("initial theta: \n", model_parameter_array)
    # print("initial cost: ", init_cost)

    # compute the model parameters using gradient descent
    learning_rate = 0.01
    max_iter = 20000
    (model_parameter_array_gradient_descent,gradient_descent_cost_array,iter_count) = perform_gradient_descent(ip_design_array,model_parameter_array,ip_y_array,learning_rate,init_cost,max_iter)

    print("theta after gradient descent: \n", model_parameter_array_gradient_descent)
    print("cost: ", gradient_descent_cost_array[iter_count-1])
    print("iter count: ", iter_count)

    # compute the model parameters using the normal equation
    # the normal equation does not require feature normalization
    ip_design_array = create_design_array(ip_x_array)
    (model_parameter_array_normal_equation,normal_equation_cost) = perform_normal_equation(ip_design_array,ip_y_array)

    print("theta after normal equation: \n", model_parameter_array_normal_equation)
    print("cost: ", normal_equation_cost)

    # create some plots
    # plot the input data as a scatter plot
    plot_1d_scatterplot(ip_x_array[:, 0],ip_y_array[:, 0],"input data","population (x 10k)","profit (x 10k)",[4, 25, -5, 25],"rx",2)

    # plot scatter plot along with gradient descent fit
    x_vals = np.arange(0,25,0.1)
    (x_vals_count,) = x_vals.shape
    x_vals_array = np.ones((x_vals_count,2))
    x_vals_array[:,1] = x_vals

    x_vals_norm = (x_vals - mean_x_array[0,0])/range_x_array[0,0]
    x_vals_norm_array = np.ones((x_vals_count, 2))
    x_vals_norm_array[:, 1] = x_vals_norm

    y_vals_array = compute_model_output_array(x_vals_norm_array, model_parameter_array_gradient_descent)

    plot_1d_scatter_and_fit(ip_x_array[:, 0],ip_y_array[:, 0],x_vals_array[:,1], y_vals_array[:,0],
                            "input data and gradient descent fit","population (x 10k)","profit (x 10k)",[4, 25, -5, 25],
                            "rx",2,"b-",1,"input","gradient descent")

    # plot scatter plot along with normal equation fit
    y_vals_array = compute_model_output_array(x_vals_array, model_parameter_array_normal_equation)

    plot_1d_scatter_and_fit(ip_x_array[:, 0], ip_y_array[:, 0], x_vals_array[:, 1], y_vals_array[:, 0],
                            "input data and normal equation fit", "population (x 10k)", "profit (x 10k)",
                            [4, 25, -5, 25],
                            "rx", 2, "b-", 1, "input", "normal equation")

    # plot the variation of the cost function with each iteration
    plot_1d_scatterplot(range(0,max_iter), gradient_descent_cost_array[0:max_iter], "Cost during gradient descent", "gradient descent iteration", "mean square error cost",
                        [0, 20000, 0, 35], "r-", 2)

    # plot the contour plot of the cost function varying with the model parameters
    theta_0_vals = np.arange(-10,10,0.01)
    theta_1_vals = np.arange(-1,4,0.01)

    # reallocate ip_design_array to correspond to feature normalized values
    #ip_design_array = create_design_array(norm_x_array)

    cost_function_array = np.zeros((theta_1_vals.size,theta_0_vals.size))

    for i in range(theta_0_vals.size):
        for j in range(theta_1_vals.size):
            curr_parameter_array = np.array([[theta_0_vals[i]], [theta_1_vals[j]]])
            curr_cost = compute_cost_function_square_mean_error(ip_design_array, curr_parameter_array, ip_y_array)
            cost_function_array[j, i] = curr_cost[0,0]

    plot_contour_plot(cost_function_array,theta_0_vals,theta_1_vals,"Cost function","theta_0","theta_1")


if __name__ == "__main__":
    main()
