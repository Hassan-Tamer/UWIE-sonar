import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from net.Ushape_Trans import Generator
import copy

class VGG19FeatureExtractor(nn.Module):
    # input [1, 1, 507, 507]
    def __init__(self, output_dim=4096):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)    
        self.features = vgg19.features[0:30]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()        
        
    def forward(self, x):
        x = self.features(x) #[1, 512, 31, 31]
        x = self.maxpool(x) #[1, 512, 16, 16]
        x = self.conv(x)
        x = self.relu(x) #[1, 256, 16, 16]
        return x
    
    def generate_random_image(self):
        return torch.randn(1,1,507,507)


class ImageFeatureExtractor(nn.Module):
    # input [1, 3, 256, 256]
    def __init__(self, image_base_model):
        super(ImageFeatureExtractor, self).__init__()
        self.base_model = image_base_model
        self.base_model.use_encoder_only = True
        self.base_model.train()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.base_model(x)  #[1, 256, 16, 16]
        return x
    
    def generate_random_image(self):
        return torch.randn(1,3,256,256)
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Fusion(nn.Module):
    # input 2 vectors [1,256,16,16]
    def __init__(self):
        super(Fusion, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.swish = Swish()

        self.convTranspose1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.convTranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convTranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
    
    def forward(self, x, y):
        z = torch.cat((x,y),1)  #[1, 512, 16, 16]
        z = self.convTranspose1(z)  # [1, 128, 32, 32]
        z = self.swish(z)
        z = self.convTranspose2(z)  # [1, 64, 64, 64]
        z = self.swish(z)
        z = self.convTranspose3(z)  # [1, 3, 128, 128]
        z = self.swish(z)
        z = self.upsample(z)        
        return z
    
    def generate_random_image(self):
        return torch.randn(1,256,16,16), torch.randn(1,256,16,16)
    
    
class FinalModel(nn.Module):
    def __init__(self,generator):
        super(FinalModel,self).__init__()
        self.sonar_feature_extractor = VGG19FeatureExtractor()
        original_generator = generator
        self.originalModelCopy = copy.deepcopy(original_generator)
        self.img_feature_extractor = ImageFeatureExtractor(self.originalModelCopy)
        self.fusion = Fusion()

    def forward(self, cam, sonar):
        x = self.sonar_feature_extractor(sonar)
        y = self.img_feature_extractor(cam)
        z = self.fusion(x, y)
        return z
    
    def generate_random_image(self):
        return torch.randn(1,3,256,256) , torch.randn(1,1,507,507), 
