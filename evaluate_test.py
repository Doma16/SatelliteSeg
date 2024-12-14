import os
import cv2
import torch
import numpy as np
import pandas as pd
from model.our_model import WholeModel
from Transform import Transform

from config import config
from utils import get_save_name

COLUMN_NAMES = ["id", "prediction"]
MODEL_PATH = "model/trained_models/"
device = 'cpu'

def create_big_output(model, transformed_image, model_input_size=(400, 400), step=208):
    out_shape = transformed_image.shape
    big_output = torch.zeros(transformed_image.shape[-2:], dtype=torch.float32)
    weight_factor = torch.zeros_like(big_output)
    
    for start_h in range(0, out_shape[-2] - model_input_size[-2] + 1, step):
        for start_w in range(0, out_shape[-1] - model_input_size[-1] + 1, step):
            end_h = start_h + model_input_size[0]
            end_w = start_w + model_input_size[1]
            
            image_slice = transformed_image[:, start_h:end_h, start_w:end_w]
            image_slice = image_slice.unsqueeze(0)
            with torch.no_grad():        
                model_out = model(image_slice.to(device)).cpu()
                model_out = model_out.squeeze(0).squeeze(0)
                model_out = torch.clamp(model_out, min=0.0, max=1.0)
            
            # KADA DA SE KLAMPUJE ???
            big_output[start_h:end_h, start_w:end_w] += model_out
            weight_factor[start_h:end_h, start_w:end_w] += torch.ones(model_input_size)
    
    return big_output / weight_factor

def export_to_df(prediction: torch.Tensor, test_id):
    unfold = torch.nn.Unfold(kernel_size=(16, 16), stride=16)
    unfolded_pred = unfold(prediction.unsqueeze(0).unsqueeze(0))
    mean_by_kernel = unfolded_pred.mean(dim=1, keepdim=False).squeeze(0).reshape((38, 38))
    assert mean_by_kernel.shape == (38, 38)
    
    rows = []
    test_string = ("0" * (3 - len(test_id))) + test_id
    for row in range(38):
        for col in range(38):
            row_id = "_".join([test_string, str(row*16), str(col*16)])
            pred = round(mean_by_kernel[row, col].item())
            rows.append(pd.DataFrame({"id": [row_id], "prediction": [pred]}))
    
    return pd.concat(rows, ignore_index=True)

def generate_test_output(model, transform, test_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    test_dirs = os.listdir(test_path)
    dataframes = []
    
    for test_dir in test_dirs:
        print(test_dir)
        test_image_path = os.path.join(test_path, test_dir, test_dir + ".png")
        test_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)
        transformed_image = transform.image_transform(test_image)
        
        prediction = create_big_output(model, transformed_image)
        
        df = export_to_df(prediction, test_id=test_dir.split("test_")[1])
        dataframes.append(df)
        
        out_file = os.path.join(out_path, test_dir + ".npy")
        np.save(out_file, prediction.numpy())  # unsqueeze ? 
    
    whole_df = pd.concat(dataframes)
    file_name = "out.csv"
    whole_df.to_csv(os.path.join(out_path, file_name), index=False) 


if __name__ == '__main__':
    model = WholeModel().to(device, dtype=torch.float32)
    load_name = get_save_name(model, config) + '_end.pth'
    load_path = os.path.join(MODEL_PATH, load_name)
    model.load_state_dict(torch.load(load_path, map_location=device))
    
    generate_test_output(model, Transform(), "data/test", "./proba")