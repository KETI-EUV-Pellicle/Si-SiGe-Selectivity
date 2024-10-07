import numpy as np
from sklearn.cluster import DBSCAN

def prepare_data_from_test_img(test_img, predicted_ranges, total_rows=2476):
    X = []
    y_coords = []

    for start, end in predicted_ranges:
        for y_coord in range(start, end + 1):
            rows = []
            for offset in [-2, -1, 0, 1, 2]:
                row_idx = y_coord + offset
                if row_idx < 0 or row_idx >= total_rows:
                    row = np.zeros(test_img.shape[1])
                else:
                    row = test_img[row_idx, :]
                rows.append(row)
            
            X.append(np.stack(rows))
            y_coords.append(y_coord)
    
    return np.array(X), y_coords

def remove_spatial_outliers(x_data, y_data, eps_value=10, min_samples_value=4):
    data_points = np.array(list(zip(x_data, y_data)))
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(data_points)
    labels = dbscan.labels_
    filtered_x = [x for x, label in zip(x_data, labels) if label != -1]
    filtered_y = [y for y, label in zip(y_data, labels) if label != -1]
    return filtered_x, filtered_y

def cluster_points(filtered_sorted_layers, degree=9):
    cluster_layers = {}

    for layer, data in filtered_sorted_layers.items():
        x_points, y_points = np.array(data['x']), np.array(data['y'])

        red_line = np.poly1d(np.polyfit(x_points, y_points, deg=degree))

        top = {'x': [x for x, y in zip(x_points, y_points) if y <= red_line(x)],
               'y': [y for x, y in zip(x_points, y_points) if y <= red_line(x)]}

        bot = {'x': [x for x, y in zip(x_points, y_points) if y > red_line(x)],
               'y': [y for x, y in zip(x_points, y_points) if y > red_line(x)]}

        cluster_layers[layer] = {'top': top, 'bot': bot}

    return cluster_layers