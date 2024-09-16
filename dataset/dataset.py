import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from Preprocess import pre_process

class Shipwreck(Dataset):
    def __init__(self,sonar_dir,transform=None):
        self.transform = transform
        self.sonar_paths = [os.path.join(sonar_dir, fname) for fname in os.listdir(sonar_dir)]
        pp = pre_process()

    def __len__(self):
        return len(self.sonar_paths)

    def __getitem__(self, idx):
        sonar_path = self.sonar_paths[idx]
        cam_path = self.transform_filename(sonar_path).replace('sonar', 'camera')
        sonarTensor = self.sonarPP(sonar_path)
        camTensor = self.camPP(cam_path)
        return camTensor,sonarTensor
    
    def sonarPP(self,sonar_path):
        pp = pre_process()
        img = pp.totalPipeline(sonar_path)
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        return img_tensor

    def camPP(self,rgb_image_path):
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        rgb_image = rgb_image.resize((256, 256))
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(rgb_image) 
        return rgb_tensor

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
    dataset = Shipwreck(sonar_dir='dataset/d1/sonar')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i,j in dataloader:
        print(i.shape)    
        print(j.shape)
        break

    print(len(dataset))
    