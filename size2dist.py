import numpy as np
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt

# import conversion data
lut_path = './modules/size2dist.csv'
_size_data, _dist_data = [], []
with open(lut_path) as lut:
    lot_data = csv.reader(lut, delimiter=',')
    for row in lot_data:
        _size_data.append(float(row[0]))
        _dist_data.append(float(row[1]))


def Size2Dist(n_polynomials=10, interpolate=False):
    """
    Create headsize to distance conversion from lut table
    :return:
    """
    if interpolate:
        return interp1d(_size_data, _dist_data, kind='quadratic', fill_value='extrapolate')
    return np.poly1d(np.polyfit(_size_data, _dist_data, n_polynomials))


def plot_fit(n_p):
    size2dist = Size2Dist(n_p)
    size2dist_intp = Size2Dist(n_p, True)
    size_distr = np.linspace(0, np.max(_size_data), 100)
    dist_ploysolve = size2dist(size_distr)
    dist_interp = size2dist_intp(size_distr)

    data_point, fit, fit_intp= plt.plot(_size_data, _dist_data, '*',
                                                  size_distr, dist_ploysolve, '--',
                                                  size_distr, dist_interp, '-.')
    plt.grid()
    plt.legend((data_point, fit, fit_intp), ('data_point', 'poly fit', 'interpolated fit'))
    plt.show()


if __name__ == '__main__':
    plot_fit(5)
