import numpy as np


def read_input_file(file_name):
    ip_array = np.loadtxt(file_name, delimiter=',')
    return ip_array # ip_array is a numpy array


def main():
    ip_array = read_input_file("ex1data1.txt")


if __name__ == '__main__':
    main()