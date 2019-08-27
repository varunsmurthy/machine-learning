import numpy as np


def prepare_input_matrices(ip_array):
    (ip_row_count, ip_col_count) = ip_array.shape
    ip_x_array = ip_array[:,0:ip_col_count-1]
    ip_y_array = ip_array[:,ip_col_count-1:ip_col_count]

    return ip_x_array, ip_y_array


def perform_feature_normalization(ip_x_array):
    (ip_row_count, ip_col_count) = ip_x_array.shape
    norm_x_array = np.zeros(ip_x_array.shape)
    mean_x_array = np.zeros((1,ip_col_count))
    range_x_array = np.zeros((1, ip_col_count))

    for i in range(0,ip_col_count):
        curr_x_feature = ip_x_array[:,i]
        mean_x_array[0,i] = curr_x_feature.mean()
        range_x_array[0, i] = curr_x_feature.max() - curr_x_feature.min()
        norm_x_array[:,i] = (curr_x_feature - mean_x_array[0,i])/(range_x_array[0, i])

    return norm_x_array,mean_x_array,range_x_array


def create_design_array(ip_x_array):
    (ip_row_count, ip_col_count) = ip_x_array.shape
    design_array = np.ones((ip_row_count, ip_col_count+1))
    design_array[:,1::] = ip_x_array

    return design_array


def compute_model_output_array(ip_design_array, ip_parameter_array):
    ip_design_matrix = np.asmatrix(ip_design_array)
    ip_parameter_matrix = np.asmatrix(ip_parameter_array)
    model_output_matrix = ip_design_matrix*ip_parameter_matrix

    return np.asarray(model_output_matrix)


def compute_cost_function_square_mean_error(ip_design_array, ip_parameter_array, ip_y_array):
    (sample_size_count, feature_count) = ip_design_array.shape
    model_output_array = compute_model_output_array(ip_design_array, ip_parameter_array)
    error_array = model_output_array - ip_y_array
    error_matrix = np.asmatrix(error_array)

    cost_function = (error_matrix.transpose()*error_matrix)/(2*sample_size_count)
    return cost_function


def perform_gradient_descent(ip_design_array, ip_parameter_array, ip_y_array, learning_rate, init_cost, max_iter):
    (sample_size_count, feature_count) = ip_design_array.shape

    # initialize the array which holds the cost after each iteration and the number of iterations performed
    cost_array = np.zeros(max_iter)
    iter_count = 0

    # define the minimum acceptable cost
    acceptable_cost = (0.1/100)*init_cost
    curr_cost = init_cost

    while (curr_cost >= acceptable_cost) and (iter_count < max_iter):
        model_output_array = compute_model_output_array(ip_design_array, ip_parameter_array)
        error_array = model_output_array - ip_y_array

        error_matrix = np.asmatrix(error_array)
        ip_design_matrix = np.asmatrix(ip_design_array)

        ip_parameter_array = ip_parameter_array - (ip_design_matrix.transpose()*error_matrix)*(learning_rate/sample_size_count)
        curr_cost = compute_cost_function_square_mean_error(ip_design_array, ip_parameter_array, ip_y_array)
        cost_array[iter_count] = curr_cost
        iter_count += 1

    return ip_parameter_array, cost_array, iter_count


def perform_normal_equation(ip_design_array, ip_y_array):
    ip_design_matrix = np.asmatrix(ip_design_array)
    ip_y_matrix = np.asmatrix(ip_y_array)

    parameter_array = (np.linalg.pinv(ip_design_matrix.transpose()*ip_design_matrix))*ip_design_matrix.transpose()*ip_y_matrix
    normal_equation_cost = compute_cost_function_square_mean_error(ip_design_array, parameter_array, ip_y_array)

    return parameter_array,normal_equation_cost