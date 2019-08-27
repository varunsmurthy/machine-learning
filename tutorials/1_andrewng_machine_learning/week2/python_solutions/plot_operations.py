from matplotlib import pyplot


def plot_1d_scatterplot(x_array,y_array,title,xlabel,ylabel,lim_list,marker_str,linewidth_num):
    # there should be some error handling to check if the dimensions of x and y match, let's do that later
    pyplot.plot(x_array,y_array,marker_str,linewidth = linewidth_num)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.axis(lim_list)
    pyplot.grid(True)
    pyplot.show()

    return 0


def plot_1d_scatter_and_fit(x_array_scatter,y_array_scatter,x_array_fit,y_array_fit,title,xlabel,ylabel,lim_list,
                            marker_str_scatter,linewidth_num_scatter,marker_str_fit,linewidth_num_fit,legend_scatter,
                            legend_fit):
    # there should be some error handling to check if the dimensions of x and y match, let's do that later
    pyplot.plot(x_array_scatter,y_array_scatter,marker_str_scatter,linewidth = linewidth_num_scatter,label=legend_scatter)
    pyplot.plot(x_array_fit, y_array_fit, marker_str_fit, linewidth=linewidth_num_fit,label=legend_fit)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.axis(lim_list)
    pyplot.grid(True)
    pyplot.legend()
    pyplot.show()

    return 0


def plot_contour_plot(plot_array,x_vals,y_vals,title,xlabel,ylabel):
    pyplot.jet()
    cont = pyplot.contourf(x_vals, y_vals, plot_array, range(0,100,5))
    pyplot.colorbar(cont)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.show()