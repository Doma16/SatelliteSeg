from Dataset import SatDataset
from Transform import Transform

from utils import read_json_variable, get_save_name
from model.our_model import WholeModel
from model.unet import UNet
from config import config
from eval import visualize

import numpy as np
import torch
import os
import cv2


threshold = 0.25
def patch_to_label(p):
    mean = np.mean(p)
    if mean > threshold:
        return 1
    else:
        return 0

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(num_classes=1).to(device)
    model = model.eval()

    load_model = read_json_variable('paths.json', 'save_path')
    load_path = os.path.join(load_model, get_save_name(model, config)+'_end.pth')
    model.load_state_dict(torch.load(load_path, map_location=device))

    transform = Transform()
    path = read_json_variable('paths.json', 'test')
    
    out_path = read_json_variable('paths.json', 'save_path')
    out_path = os.path.join(out_path, get_save_name(model, config))

    os.makedirs(out_path, exist_ok=True)
    test_dirs = os.listdir(path)

    for test_dir in test_dirs:
        print(test_dir)
        test_image_path = os.path.join(path, test_dir,  test_dir + ".png")
        test_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

        image = transform.image_transform(test_image).to(device)
        image = image.unsqueeze(0)
        out = None
        with torch.no_grad():
            out = model(image)

        out = torch.clamp(out, min=0.0, max=1.0)
        out = torch.round(out)

        visualize(out[0, 0], image[0])

        out_name = f'{int(test_dir.split("_")[-1]):03d}' + '.npy'
        out = out.cpu().numpy()[0]
        np.save(os.path.join(out_path, out_name), out)


    out_csv = os.path.join(out_path, 'sub.csv')
    csvf = open(out_csv, 'w')
    csvf.write('id,prediction\n')
    
    for test_dir in test_dirs:
        im_id = f'{int(test_dir.split("_")[-1]):03d}'
        out_name = im_id + '.npy'
        load_path = os.path.join(out_path, out_name)

        out_str = []
        im = np.load(load_path)[0]
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j+ patch_size]
                label = patch_to_label(patch)
                out_str.append(f'{int(im_id):03d}_{i}_{j},{label}\n')
        
        for line in out_str:
            csvf.write(line)

    csvf.close()