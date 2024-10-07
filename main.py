import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import cv2
import numpy as np
import torch
import glob
from scipy.stats import stats, linregress
from sklearn.cluster import KMeans
import threading
import pandas as pd

from models.unet import UNet1D
from models.tcn import SiGe_TCN, SiGe_X_Range_TCN
from utils.image_processing import read_img_with_korean_path, si_predict_layers, sige_predict_layers, scale_detection
from utils.data_preparation import prepare_data_from_test_img, remove_spatial_outliers, cluster_points
from utils.geometric_operations import connect_and_interpolate_points, find_intersection, find_line_intersection, distance_func, si_loss_amount, sige_selectivity
from utils.data_processing import sige_predict, find_consecutive_ones, merge_ranges, prepare_data_from_test_img, predict_layer_points, recess_point_predict_layers
from losses.focal_loss import FocalLoss

class SelectivityApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Selectivity Analysis App")
        self.master.geometry("2476x2476")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_ui()
        self.load_models()
        self.mode = 'view'

    def setup_ui(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)

        self.btn_load = ttk.Button(btn_frame, text="Load Image", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=5)

        self.btn_predict_recess = ttk.Button(btn_frame, text="Predict Recess Points", command=self.predict_recess_points_thread)
        self.btn_predict_recess.grid(row=0, column=1, padx=5)

        self.btn_process = ttk.Button(btn_frame, text="Calculate Selectivity", command=self.process_image_thread)
        self.btn_process.grid(row=0, column=2, padx=5)

        mode_frame = ttk.Frame(btn_frame)
        mode_frame.grid(row=0, column=3, padx=20)

        self.mode_var = tk.StringVar(value="view")
        ttk.Radiobutton(mode_frame, text="View", variable=self.mode_var, value="view", command=lambda: self.set_mode("view")).grid(row=0, column=0)
        ttk.Radiobutton(mode_frame, text="Add", variable=self.mode_var, value="add", command=lambda: self.set_mode("add")).grid(row=0, column=1)
        ttk.Radiobutton(mode_frame, text="Move", variable=self.mode_var, value="move", command=lambda: self.set_mode("move")).grid(row=0, column=2)
        ttk.Radiobutton(mode_frame, text="Delete", variable=self.mode_var, value="delete", command=lambda: self.set_mode("delete")).grid(row=0, column=3)

        self.mode_label = ttk.Label(main_frame, text="Current Mode: View")
        self.mode_label.grid(row=1, column=0, columnspan=2, pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, columnspan=2, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_models(self):
        self.model = UNet1D(in_channels=5, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('2.데이터/7. 이미지처리/project/model_parameter/edge_model_5_0.pth'))
        
        self.sige_model = SiGe_TCN(input_channels=5, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
        self.sige_model.load_state_dict(torch.load('2.데이터/7. 이미지처리/project/model_parameter/recess_range_model.pt'))
        
        self.sige_x_model = SiGe_X_Range_TCN(input_channels=5, num_channels=[16, 32, 128, 256], kernel_size=3, dropout=0.2)
        self.sige_x_model.load_state_dict(torch.load('2.데이터/7. 이미지처리/project/model_parameter/recess_point_model2.pt'))

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.test_img = read_img_with_korean_path(file_path)
            self.test_img = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
            self.test_img = self.test_img / 255.0  # 정규화
            if self.test_img.shape[0] == 2475:
                self.test_img = np.pad(self.test_img, ((0, 1), (0, 1)), mode='edge')
            
            self.display_image(self.test_img)

    def display_image(self, img):
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.canvas.draw()

    def predict_recess_points_thread(self):
        self.start_progress()
        threading.Thread(target=self.predict_recess_points).start()

    def predict_recess_points(self):
        try:
            self.model.to(self.device)
            self.model.eval()

            window_size = 5
            stride = 1
            predictions = sige_predict(self.sige_model, self.test_img, window_size, stride)
            binary_predictions = [1 if p > 0.7 else 0 for p in predictions]
            predicted_ranges = find_consecutive_ones(binary_predictions)
            range_threshold = 20
            merged_ranges = merge_ranges(predicted_ranges, range_threshold)

            self.sige_x_model.to(self.device)

            X, y_coords = prepare_data_from_test_img(self.test_img, merged_ranges)
            predicted_points = predict_layer_points(self.sige_x_model, X, self.device)

            result_layer_point = {}
            for y_coord, pred in zip(y_coords, predicted_points):
                result_layer_point[y_coord] = [int(pred)]

            x_min = min([i[0] for i in result_layer_point.values()])
            x_max = max([i[0] for i in result_layer_point.values()])

            center_points = [round((start + end)/2) for (start, end) in merged_ranges]
            recess_point = recess_point_predict_layers(self.model, self.test_img, center_points, self.device, x_min, x_max)

            recess_points = {k: points[0] for k, points in recess_point.items() if len(points) != 0}

            self.recess_points = dict()
            for i, point in enumerate(center_points):
                v = []
                for offset in range(-2, 3):
                    if point + offset in recess_points.keys():
                        v.append(recess_points[point + offset])
                if len(v) != 0:
                    self.recess_points[point] = round(sum(v) / len(v))
                else:
                    if i != 0:
                        self.recess_points[point] = self.recess_points[center_points[i-1]]
                    else:
                        self.recess_points[point] = 0

            self.sort_recess_points()
            self.master.after(0, self.plot_recess_points)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred while predicting recess points: {str(e)}"))
        finally:
            self.master.after(0, self.stop_progress)

    def process_image_thread(self):
        self.start_progress()
        threading.Thread(target=self.process_image).start()

    def process_image(self):
        if not hasattr(self, 'recess_points'):
            self.master.after(0, lambda: messagebox.showerror("Error", "Please predict recess points first"))
            self.master.after(0, self.stop_progress)
            return

        try:
            self.si_predicted_data = si_predict_layers(self.model, self.test_img, self.device)

            x_values = []
            y_values = []

            for x, y_list in self.si_predicted_data.items():
                for y in y_list:
                    x_values.append(x)
                    y_values.append(y)

            y_data_points = np.array(y_values).reshape(-1, 1)
            n_clusters = 9  # Layer 개수
            kmeans_y = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(y_data_points)

            labels_y = kmeans_y.labels_

            clustered_layers_y = {i: {'x': [], 'y': []} for i in range(n_clusters)}
            for label, (x, y) in zip(labels_y, zip(x_values, y_values)):
                clustered_layers_y[label]['x'].append(x)
                clustered_layers_y[label]['y'].append(y)

            cluster_medians = {i: np.median(clustered_layers_y[i]['y']) for i in range(n_clusters)}
            sorted_clusters = sorted(cluster_medians.items(), key=lambda item: item[1])  # Ascending order

            sorted_clustered_layers_y = {}
            for new_label, (old_label, _) in enumerate(sorted_clusters):
                sorted_clustered_layers_y[new_label] = clustered_layers_y[old_label]

            filtered_sorted_layers = {}
            for layer, data in sorted_clustered_layers_y.items():
                filtered_x, filtered_y = remove_spatial_outliers(data['x'], data['y'], eps_value=20, min_samples_value=9)
                filtered_sorted_layers[layer] = {'x': filtered_x, 'y': filtered_y}

            cluster_layers = cluster_points(filtered_sorted_layers)

            self.final_layer = dict()

            for layer, clusters in cluster_layers.items():
                top = dict()
                bot = dict()
                cluster_0_x_interp, cluster_0_y_interp = connect_and_interpolate_points(clusters['top']['x'], clusters['top']['y'])
                cluster_1_x_interp, cluster_1_y_interp = connect_and_interpolate_points(clusters['bot']['x'], clusters['bot']['y'])
                
                top['x'] = cluster_0_x_interp
                top['y'] = cluster_0_y_interp

                bot['x'] = cluster_1_x_interp
                bot['y'] = cluster_1_y_interp

                self.final_layer[layer] = {'top': top, 'bot': bot}

            self.si_points = {int((self.final_layer[layer_n]['top']['y'][0] + self.final_layer[layer_n]['bot']['y'][0]) // 2): 
                              (self.final_layer[layer_n]['top']['x'][0] + self.final_layer[layer_n]['bot']['x'][0]) // 2 
                              for layer_n in self.final_layer.keys()}
            image = (self.test_img * 255.0).astype(np.uint8) 
            self.pixel_to_nm = scale_detection(image, 100)

            self.thk_a = []
            self.thk_b = []
            self.thk_c = []

            self.master.after(0, self.plot_results)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred while processing the image: {str(e)}"))
        finally:
            self.master.after(0, self.stop_progress)

    def start_progress(self):
        self.progress_bar.start(10)

    def stop_progress(self):
        self.progress_bar.stop()

    def sort_recess_points(self):
        self.recess_points = dict(sorted(self.recess_points.items()))

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.mode_var.get() == 'add':
            self.recess_points[y] = x
        elif self.mode_var.get() == 'move':
            closest_point = min(self.recess_points.keys(), key=lambda k: abs(k - y))
            del self.recess_points[closest_point]
            self.recess_points[y] = x
        elif self.mode_var.get() == 'delete':
            closest_point = min(self.recess_points.keys(), key=lambda k: abs(k - y))
            del self.recess_points[closest_point]
        
        self.sort_recess_points()
        self.plot_recess_points()

    def set_mode(self, mode):
        self.mode_var.set(mode)
        self.mode_label.config(text=f"Current Mode: {mode.capitalize()}")

    def plot_recess_points(self):
        self.ax.clear()
        self.ax.imshow(self.test_img, cmap='gray')
        x_coords = list(self.recess_points.values())
        y_coords = list(self.recess_points.keys())
        self.ax.scatter(x_coords, y_coords, color='r', s=50, label='Recess Points')

        for y, x in self.recess_points.items():
            self.ax.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points')

        self.ax.set_ylim(0, 2476)
        self.ax.invert_yaxis()
        self.canvas.draw()

    def plot_results(self):
        self.ax.clear()
        # self.ax.figure.set_size_inches(20, 20)

        x_coords = list(self.recess_points.values())
        y_coords = list(self.recess_points.keys())

        si_x_coords = list(self.si_points.values())
        si_y_coords = list(self.si_points.keys())

        lines = []
        for i in range(len(x_coords) - 1):
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i + 1], y_coords[i + 1]
            slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            if np.isnan(slope):
                x2 = x2 + 1
                slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            lines.append((slope, intercept, x1, x2, y1, y2)) 

        self.ax.scatter(x_coords, y_coords, color='r', label='Points')

        for i, line in enumerate(lines):
            slope, intercept, x1, x2, y1, y2 = line
            if i == 0:
                x_values = np.linspace(2476, x2, 1000)
            else:
                x_values = np.linspace(x1, x2, 100)
            y_values = slope * x_values + intercept
            self.ax.plot(x_values, y_values, 'r-', alpha=0.5)

        shifted_lines = []
        shift = 50 / self.pixel_to_nm
        shift_xcoords = [x + shift for x in x_coords]

        for i in range(len(x_coords) - 1):
            x1, y1 = shift_xcoords[i], y_coords[i]
            x2, y2 = shift_xcoords[i + 1], y_coords[i + 1]
            slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            if np.isnan(slope):
                x2 = x2 + 1
                slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            shifted_lines.append((slope, intercept, x1, x2, y1, y2))

        self.ax.scatter(shift_xcoords, y_coords, color='r', label='Points')

        for i, line in enumerate(shifted_lines):
            slope, intercept, x1, x2, y1, y2 = line
            if i == 0:
                x_values = np.linspace(2476, x2, 1000)
            else:
                x_values = np.linspace(x1, x2, 100)
            y_values = slope * x_values + intercept
            self.ax.plot(x_values, y_values, 'r-', alpha=0.5)

        si_lines = []
        for i in range(len(si_x_coords) - 1):
            x1, y1 = si_x_coords[i], si_y_coords[i]
            x2, y2 = si_x_coords[i + 1], si_y_coords[i + 1]
            slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            if np.isnan(slope):
                x2 = x2 + 1
                slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            si_lines.append((slope, intercept, x1, x2, y1, y2)) 

        self.ax.scatter(si_x_coords, si_y_coords, color='r', label='Points')

        for i, line in enumerate(si_lines):
            if i == len(si_lines) - 1:
                x_values = np.linspace(0, x2, 100)
                y_values = slope * x_values + intercept
                self.ax.plot(x_values, y_values, 'r-', alpha=0.5)
            else:
                slope, intercept, x1, x2, y1, y2 = line
                x_values = np.linspace(x1, x2, 100)
                y_values = slope * x_values + intercept
                self.ax.plot(x_values, y_values, 'r-', alpha=0.5)

        si_shifted_lines = []
        si_shift = 10 / self.pixel_to_nm
        si_shift_xcoords = [x + si_shift for x in si_x_coords]

        for i in range(len(si_x_coords) - 1):
            x1, y1 = si_shift_xcoords[i], si_y_coords[i]
            x2, y2 = si_shift_xcoords[i + 1], si_y_coords[i + 1]
            slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            if np.isnan(slope):
                x2 = x2 + 1
                slope, intercept, _, _, _ = stats.linregress([x1, x2], [y1, y2])
            si_shifted_lines.append((slope, intercept, x1, x2, y1, y2))

        for i, line in enumerate(si_shifted_lines):
            if i == len(si_lines) - 1:
                x_values = np.linspace(0, x2, 100)
                y_values = slope * x_values + intercept
                self.ax.plot(x_values, y_values, 'r-', alpha=0.5)
            else:
                slope, intercept, x1, x2, y1, y2 = line
                x_values = np.linspace(x1, x2, 100)
                y_values = slope * x_values + intercept
                self.ax.plot(x_values, y_values, 'r-', alpha=0.5)

        for layer_n in range(len(self.final_layer) - 1):
            top_x = self.final_layer[layer_n]['top']['x']
            top_y = self.final_layer[layer_n]['top']['y']
            bot_x = self.final_layer[layer_n]['bot']['x']
            bot_y = self.final_layer[layer_n]['bot']['y']

            # SiGe 부분
            if layer_n == 0:
                intersection = find_intersection(top_x, top_y, shifted_lines[0][0], shifted_lines[0][1])
            else:
                intersection = find_intersection(top_x, top_y, shifted_lines[layer_n-1][0], shifted_lines[layer_n-1][1])

            contact_point = np.where(top_x == intersection[0])[0][0]
            tangent_range = 100

            top_slope, _, _, _, _ = stats.linregress(top_x[contact_point-tangent_range : contact_point + tangent_range],
                                                     top_y[contact_point-tangent_range : contact_point + tangent_range])

            if top_slope == 0:
                top_slope = 1e-6
            x_range = np.linspace(intersection[0] - 300, intersection[0] + 300, 100)

            b = intersection[1] - top_slope * intersection[0]
            y_range = top_slope * x_range + b

            self.ax.plot(x_range, y_range, 'r-', label='직선')

            perpendicular_slope = -1 / top_slope
            perpendicular_intercept = intersection[1] - perpendicular_slope * intersection[0]
            
            bot_intersection = find_intersection(bot_x, bot_y, perpendicular_slope, perpendicular_intercept)
            distance = np.sqrt((intersection[0] - bot_intersection[0])**2 + 
                            (intersection[1] - bot_intersection[1])**2) * self.pixel_to_nm
            self.thk_b.append(distance)

            x_perp = np.linspace(intersection[0], bot_intersection[0], 1000)
            y_perp = perpendicular_slope * x_perp + perpendicular_intercept
            self.ax.plot([intersection[0], bot_intersection[0]], [intersection[1], bot_intersection[1]], 'r-', label='Perpendicular Line')
            self.ax.text(intersection[0] + abs(intersection[0]-bot_intersection[0]) + 30,
                         intersection[1] + abs(intersection[1]-bot_intersection[1])/2,
                         f'layer {layer_n+1} : {round(distance,4)} nm', c='black')

            self.ax.scatter(*intersection, color='r', s=30, label='Top Intersection')
            self.ax.scatter(*bot_intersection, color='r', s=30, label='Bottom Intersection')

            # Si 부분
            if layer_n == 0:
                si_intersection = find_intersection(top_x, top_y, si_shifted_lines[0][0], si_shifted_lines[0][1])
            else:
                si_intersection = find_intersection(top_x, top_y, si_shifted_lines[layer_n-1][0], si_shifted_lines[layer_n-1][1])

            contact_point = np.where(top_x == si_intersection[0])[0][0]
            tangent_range = 300

            top_slope, _, _, _, _ = stats.linregress(top_x[: contact_point + tangent_range],
                                                     top_y[: contact_point + tangent_range])
            if top_slope == 0:
                top_slope = 1e-6

            x_range = np.linspace(si_intersection[0] - 300, si_intersection[0] + 300, 100)

            b = si_intersection[1] - top_slope * si_intersection[0]
            y_range = top_slope * x_range + b

            self.ax.plot(x_range, y_range, 'r-', label='직선')

            perpendicular_slope = -1 / top_slope
            perpendicular_intercept = si_intersection[1] - perpendicular_slope * si_intersection[0]
            
            si_bot_intersection = find_intersection(bot_x, bot_y, perpendicular_slope, perpendicular_intercept)
            distance = np.sqrt((si_intersection[0] - si_bot_intersection[0])**2 + 
                            (si_intersection[1] - si_bot_intersection[1])**2) * self.pixel_to_nm
            self.thk_c.append(distance)

            x_perp = np.linspace(si_intersection[0], si_bot_intersection[0], 1000)
            y_perp = perpendicular_slope * x_perp + perpendicular_intercept
            self.ax.plot([si_intersection[0], si_bot_intersection[0]], [si_intersection[1], si_bot_intersection[1]], 'r-', label='Perpendicular Line')
            self.ax.text(si_intersection[0] + abs(si_bot_intersection[0]-si_bot_intersection[0]) + 30,
                         si_intersection[1] + abs(si_intersection[1]-si_bot_intersection[1])/2 + 30,
                         f'layer {layer_n+1} : {round(distance,2)} nm', c='black')

            self.ax.scatter(*si_intersection, color='r', s=10, label='Top Intersection')
            self.ax.scatter(*si_bot_intersection, color='r', s=10, label='Bottom Intersection')

            recess_intercept = y_coords[layer_n] - top_slope * x_coords[layer_n]
            recess_line = (top_slope, recess_intercept)

            if layer_n == len(self.final_layer.keys()) - 2:
                sige_recess_intersection = find_line_intersection(si_lines[-2][0], si_lines[-2][1], recess_line[0], recess_line[1])
            else:
                sige_recess_intersection = find_line_intersection(si_lines[layer_n][0], si_lines[layer_n][1], recess_line[0], recess_line[1])
            recess_point = (x_coords[layer_n], y_coords[layer_n])
            distance = distance_func(sige_recess_intersection, recess_point) * self.pixel_to_nm
            self.thk_a.append(distance)
            self.ax.plot([recess_point[0], sige_recess_intersection[0]], [recess_point[1], sige_recess_intersection[1]], c='r')
            self.ax.text((recess_point[0] + sige_recess_intersection[0])/2,
                         (recess_point[1] + sige_recess_intersection[1])/2 + 30,
                         f'layer {layer_n+1} : {round(distance,2)} nm', c='black')
            
        self.ax.imshow(self.test_img, cmap='gray')
        self.ax.set_ylim(0, 2476)
        self.ax.invert_yaxis()
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.si_loss_values = si_loss_amount(self.thk_b,self.thk_c)
        self.sige_selectivity_values = sige_selectivity(self.thk_a,self.thk_b,self.thk_c)

        data = dict()
        # Print results
        for i in range(len(self.thk_a)):
            print(f"Layer {i+1}: A={self.thk_a[i]:.2f}, B={self.thk_b[i]:.2f}, C={self.thk_c[i]:.2f}, Si_Loss={self.si_loss_values[i]:.2f}, SiGe_Selectivity={self.sige_selectivity_values[i]:.2f}")
            data[f'{i+1} Layer'] = [self.thk_a[i], self.thk_b[i], self.thk_c[i], self.si_loss_values[i], self.sige_selectivity_values[i]]

        data = pd.DataFrame(data)
        data.index = ['A Thickness', 'B Thickness', 'C Thickness', 'Si Loss', 'SiGe Selectivity']
        data = data.transpose()
        data.to_csv('data.csv')
    def run(self):
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SelectivityApp(root)
    app.run()