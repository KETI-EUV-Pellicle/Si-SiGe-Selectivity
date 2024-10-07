import torch
import numpy as np

def sige_predict(model, image, window_size=5, stride=1):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, image.shape[0] - window_size + 1, stride):
            window = image[i:i+window_size, :]
            window_tensor = torch.FloatTensor(window).unsqueeze(0)
            output = model(window_tensor).round()
            predictions.append(output.item())
    return predictions

def find_consecutive_ones(binary_list):
    ranges = []
    start = None
    for i, value in enumerate(binary_list):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(binary_list) - 1))
    return ranges


def merge_ranges(ranges, threshold):
    if not ranges:
        return []

    merged = [ranges[0]]

    for current_start, current_end in ranges[1:]:
        last_start, last_end = merged[-1]

        if current_start - last_end <= threshold:
            merged[-1] = (last_start, current_end)
        else:
            # 새로운 범위 추가
            merged.append((current_start, current_end))

    return merged


def prepare_data_from_test_img(test_img, predicted_ranges, total_rows=2476):
    X = []
    y_coords = []  # 각 X 데이터에 해당하는 y 좌표를 저장

    for start, end in predicted_ranges:
        for y_coord in range(start, end + 1):
            # 현재 row와 전후 2개 row를 가져옴, 범위를 벗어나는 부분은 0으로 채움
            rows = []
            for offset in [-2, -1, 0, 1, 2]:
                row_idx = y_coord + offset
                if row_idx < 0 or row_idx >= total_rows:
                    row = np.zeros(test_img.shape[1])  # 범위를 벗어나는 경우 0으로 채움
                else:
                    row = test_img[row_idx, :]
                rows.append(row)
            
            # X 데이터에 추가 (5개 row를 쌓아올림)
            X.append(np.stack(rows))
            y_coords.append(y_coord)
    
    return np.array(X), y_coords  # X를 정규화

def predict_layer_points(model, X, device):
    model.eval()
    predictions = []
    model.to(device)
    
    with torch.no_grad():
        for x in X:
            x = torch.FloatTensor(x).unsqueeze(0).to(device)  # (1, 5, 2476) 형태로 변환
            output = model(x)
            predictions.append(output.item() * 2476)  # 0-1 범위의 예측값을 0-2476 범위로 변환
    
    return predictions


def recess_point_predict_layers(model, image, center_points ,device,x_min,x_max):
    model.eval()
    predicted_data = {}
    
    with torch.no_grad():
        for row in center_points:
            for gap in range(-2,3):
                rows_data = []
                for offset in range(-2, 3):
                    if 0 <= row + gap + offset < image.shape[0]:
                        rows_data.append(image[row + gap + offset, :])
                    else:
                        rows_data.append(np.zeros_like(image[0, :]))
                
                input_data = np.array(rows_data)
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
                
                output = model(input_tensor)
                output = output.squeeze().cpu().numpy()
                
                # 임계값 적용하여 레이어 위치 결정
                predicted_layers = np.where(output > 0.5)[0]
                predicted_layers = predicted_layers[(predicted_layers >= x_min - 200) & (predicted_layers <= x_max + 200)]
                predicted_data[row+gap] = predicted_layers
    
    return predicted_data