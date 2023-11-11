import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import json
import requests
from PIL import Image
from io import BytesIO
import numpy as np  
import os
from sklearn.metrics import average_precision_score
from myModel import create_model

all_error_list = ['Bánh không tròn | Distorted shape',
 'Cháy | Baking - Burnt',
 'Viền k đều | Edge - Uneven',
 'Bánh bé | Size - Too small',
 'Thiếu bóng | too few balls',
 'Không đốm | Baking - Does not have leopard-spotting',
 'Màu nhạt | Baking - Pale',
 'Viền nhỏ | Edge - Too small',
 'Viền to | Edge - Too big',
 'Nở viền không đủ | edge pizza is not enough swollen',
 'Viền thấp | Edge - Too low',
 'Phô mai cao | Topping - Cheese too high * with a core*',
 'Topping - Không đúng | incorrect portioning',
 'Không cân Topping - Not even half and half',
 'Sốt trên mép | Topping - Sauce covering on the edge',
 'Lên men thiếu | Fermentation - Lack fermentation',
 'Quá theo viền | Topping - Too strong shaping the edge',
 'Trộn lẫn | Topping - Bended',
 'Quá tập trung | Topping - Topping too centered',
 'Topping - Không đều | Not even',
 'Lên men quá nhiều | Fermentation - Over fermentation',
 'Không hình tròn | Topping - Not circled',
 'Bánh lớn | Size - Too big']

model = create_model()

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path)
    new_img = transform(img).unsqueeze(dim = 0)
    model.eval()
    with torch.inference_mode():
        output = model(new_img)
        prediction = torch.round(output)
        prediction = prediction.squeeze()
        error_list = []
        for i, element in enumerate(prediction):
            if element.item() == 1:
                error_list.append(all_error_list[i])
    if len(error_list) == 0:
        error_list.append('Không có lỗi')
    return error_list


if __name__=="__main__":
    print(predict_image('data/pizza_test_predict/64a00a1aa2f5430028b12f0f.jpg'))


