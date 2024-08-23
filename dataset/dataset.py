import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

class Shipwreck(Dataset):
    def __init__(self, camera_dir,sonar_dir,transform=None):
        # self.camera_dir = camera_dir
        self.transform = transform
        # self.camera_paths = [os.path.join(camera_dir, fname) for fname in os.listdir(camera_dir)]
        self.sonar_paths = [os.path.join(sonar_dir, fname) for fname in os.listdir(sonar_dir)]

    def __len__(self):
        return len(self.sonar_paths)

    def __getitem__(self, idx):
        sonar_path = self.sonar_paths[idx]
        cam_path = self.transform_filename(sonar_path).replace('sonar', 'camera')
        img4channel = self.concat_images(cam_path,sonar_path)
        return img4channel
    
    def concat_images(self,rgb_image_path, gray_image_path):
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        gray_image = Image.open(gray_image_path).convert('L')

        rgb_image = rgb_image.resize((256, 256))
        gray_image = gray_image.resize((256, 256))

        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(rgb_image) 
        gray_tensor = to_tensor(gray_image)  

        return torch.cat((rgb_tensor, gray_tensor), dim=0)

    def split_tensor(self,combined_tensor):
        assert combined_tensor.shape[1] == 4, "Input tensor must have 4 channels."

        rgb_tensor = combined_tensor[:,:3, :, :]
        gray_tensor = combined_tensor[:,3, :, :].unsqueeze(0)

        return rgb_tensor, gray_tensor

    def transform_filename(self,input_path):
        directory, filename = os.path.split(input_path)
        name, extension = os.path.splitext(filename)
        
        if name.startswith('s') and 'c' in name:
            parts = name.split('c')
            sonar_part = parts[0]  
            code_part = parts[1]  
        
            new_name = f'c{code_part}{sonar_part}'
            
            new_path = os.path.join(directory, new_name + extension)
            
            return new_path
        else:
            raise ValueError("Filename format is not as expected (should start with 's' and contain 'c').")

if __name__ == "__main__":
    dataset = Shipwreck(camera_dir='dataset/d1/camera',sonar_dir='dataset/d1/sonar')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for input_image in dataloader:
    #     print(input_image.shape)    

    print(len(dataset))
    