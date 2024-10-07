import numpy as np
from scipy.interpolate import interp1d

def connect_and_interpolate_points(x_data, y_data):
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]
    
    x_interp = np.arange(min(x_sorted), max(x_sorted) + 1)
    
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    y_interp = interp_func(x_interp)
    
    return x_interp, y_interp

def find_intersection(x, y, m, b):
    y_line = m * x + b
    idx = np.argmin(np.abs(y - y_line))
    return x[idx], y[idx]

def find_line_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None
    
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

def distance_func(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def si_loss_amount(b, c):
    return [(b[i]-c[i])/2 for i in range(len(b))]

def sige_selectivity(a, b, c):
    return [(2 * a[i]) / (b[i]-c[i]) for i in range(len(b))]