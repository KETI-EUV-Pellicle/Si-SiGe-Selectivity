import cv2
import numpy as np
import torch

def read_img_with_korean_path(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

def si_predict_layers(model, image, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predicted_data = {}
    
    with torch.no_grad():
        for col in range(image.shape[1]):
            cols_data = []
            for offset in range(-2, 3):
                if 0 <= col + offset < image.shape[1]:
                    cols_data.append(image[:, col + offset])
                else:
                    cols_data.append(np.zeros_like(image[:, 0]))
            
            input_data = np.array(cols_data)
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            output = output.squeeze().cpu().numpy()
            
            predicted_layers = np.where(output > 0.5)[0]
            predicted_data[col] = predicted_layers
    
    return predicted_data

def sige_predict_layers(model, image, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predicted_data = {}
    
    with torch.no_grad():
        for row in range(image.shape[0]):
            rows_data = []
            for offset in range(-2,3):
                if 0 <= row + offset < image.shape[0]:
                    rows_data.append(image[row + offset, :])
                else:
                    rows_data.append(np.zeros_like(image[0, :]))
            
            input_data = np.array(rows_data)
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            output = output.squeeze().cpu().numpy()
            
            predicted_layers = np.where(output > 0.5)[0]
            predicted_data[row] = predicted_layers
    
    return predicted_data


def scale_detection(image,scale_value = 100):
    # 이미지의 하단 부분만 크롭 (Scale Bar가 있는 영역)
    height, width = image.shape
    bottom_region = image[int(height*0.9):, :]
    # 이진화
    _, binary = cv2.threshold(bottom_region, 200, 255, cv2.THRESH_BINARY)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 긴 윤곽선 찾기
    scale_bar = max(contours, key=cv2.contourArea)

    # Scale Bar의 길이 계산
    x, y, w, h = cv2.boundingRect(scale_bar)
    scale_bar_length = w

    print(f"Scale Bar의 길이: {scale_bar_length} px")

    nm_per_pixel = scale_value / scale_bar_length
    print(f"1 px 당 {nm_per_pixel:.2f} nm")
    
    return nm_per_pixel