import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import os

def sort_csv(path_to_csv, out_path="send.csv"):
    df = pd.read_csv(path_to_csv)
    sorted_dfs = []
    for i in range(1, 51):
        id = ("0" * (3 - len(str(i)))) + str(i)
        current_df = df[df["id"].str.startswith(id)]
        sorted_dfs.append(current_df)
    
    result = pd.concat(sorted_dfs)
    result.to_csv(out_path, index=False)
    

def block_visualize(path_to_csv, test_data_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    df = pd.read_csv(path_to_csv)
    
    for i in range(1, 51):
        id = ("0" * (3 - len(str(i)))) + str(i)
        current_df = df[df["id"].str.startswith(id)]

        pred = np.zeros((608, 608))
        for _, row in current_df.iterrows():
            row_start, col_start = row["id"].split("_")[1:]
            row_start, col_start = int(row_start), int(col_start)
            if int(row["prediction"]):
                pred[row_start:row_start + 16, col_start:col_start + 16] = 1

        test_image_path = f"data/test/test_{i}/test_{i}.png"
        test_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

        fig = plt.figure(figsize=(16, 8))
        ax_pred = fig.add_subplot(1, 2, 1)
        ax_gt = fig.add_subplot(1, 2, 2)

        ax_pred.imshow(pred, cmap="gray")
        ax_gt.imshow(test_image)
        fig.savefig(os.path.join(out_path, f"{i}.png")) 
    

if __name__ == '__main__':
    block_visualize("./model/trained_models/UNet_2_0.0001_50_True_torch.float32/sub.csv", "data/test", "proba/slike")
    sort_csv("./model/trained_models/UNet_2_0.0001_50_True_torch.float32/sub.csv")  
    