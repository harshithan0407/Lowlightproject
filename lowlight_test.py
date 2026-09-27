import torch, os, sys, argparse, time, glob
import torch.nn as nn
import torchvision, torch.optim
import torch.backends.cudnn as cudnn
import dataloader, model
import numpy as np
from torchvision import transforms
from PIL import Image

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(BASE_DIR, "snapshots", "Epoch99.pth")

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)

    DCE_net = model.enhance_net_nopool()
    DCE_net.load_state_dict(torch.load(SNAPSHOT_PATH, map_location='cpu'))
    DCE_net.eval()

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    print(f"Enhancement time: {time.time() - start:.3f}s")

    # Save output mirroring test_data → result folder structure
    result_path = image_path.replace('test_data', 'result')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"Saved: {result_path}")


def image_enhancement():
    with torch.no_grad():
        filePath = os.path.join(BASE_DIR, 'data', 'test_data', 'images')
        test_list = glob.glob(os.path.join(filePath, "*"))
        if not test_list:
            print("No test images found in data/test_data/images/")
            return
        for image in test_list:
            print(f"Processing: {image}")
            lowlight(image)


if __name__ == "__main__":
    image_enhancement()
