import os

import torch
import cv2
from model.unet import UNet
from Dataset import Dataset
from Transform import EvalTransform
from utils import read_json_variable
from gen_out_images import patch_to_label

device = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold = 0.25
load_paths = [...]
NUM_MODELS = len(load_paths)

def final_vote(models_output, im_id):
    out_str = []
    models_patch_label = []
    patch_size = 16
    for j in range(0, output.shape[1], patch_size):
        for i in range(0, output.shape[0], patch_size):
            for output in models_output:
                patch = output[i:i + patch_size, j:j+ patch_size]
                model_patch_label = patch_to_label(patch)
                models_patch_label.append(model_patch_label)
            final_vote = sum(models_patch_label) >= NUM_MODELS // 2
            out_str.append(f'{int(im_id):03d}_{i}_{j},{final_vote}\n')
    
    return out_str
            

def majority_vote(models):
    transform = EvalTransform()
    path = read_json_variable('paths.json', 'test')
    out_path = read_json_variable('paths.json', 'save_path')
    os.makedirs(out_path, exist_ok=True)
    test_dirs = os.listdir(path)
    
    out_csv = os.path.join(out_path, 'sub.csv')
    csvf = open(out_csv, 'w')
    csvf.write('id,prediction\n')

    for test_dir in test_dirs:
        im_id = f'{int(test_dir.split("_")[-1]):03d}'
        test_image_path = os.path.join(path, test_dir, test_dir+'.png')
        test_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

        image = transform.image_transform(test_image)
        image = image.unsqueeze(0)
        out = None
        out_str = []
        with torch.no_grad():
            models_out = []
            for model in models:
                out = model(image)
                out = torch.clamp(out, min=0.0, max=1.0)
                out = torch.round(out)
                models_out.append(out.cpu().numpy())
            
            
            out_str.append(final_vote(models_out, im_id))
            
    for line in out_str:
            csvf.write(line)
    csvf.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    
    for load_path in load_paths:
        model = UNet().to(device)
        model.load_state_dict(torch.load(load_path, map_location=device), strict=False)
        model = model.to(device)
        models.append(model)

    majority_vote(models=models)


if __name__ == '__main__':
    main()
