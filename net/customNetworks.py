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
        
        
        self.features = vgg19.features
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class ImageFeatureExtractor(nn.Module):
    # input [1, 3, 256, 256]
    def __init__(self, image_base_model):
        super(ImageFeatureExtractor, self).__init__()
        self.base_model = image_base_model
        self.base_model.use_decoder_only = True
        self.base_model.train()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.base_model(x)  #torch.Size([1, 256, 16, 16])
        x = self.maxpool(x) #torch.Size([1, 256, 8, 8])
        x = self.conv1(x) #torch.Size([1, 64, 8, 8])
        x = F.gelu(x)
        x = self.flatten(x) #torch.Size([1, 4096])
        return x

class Fusion(nn.Module):
    # input 2 vectors [1,4096]
    def __init__(self):
        super(Fusion, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.convTranspose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.convTranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convTranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
    
    def forward(self, x, y):
        x = x.view(-1,1,64,64)
        y = y.view(-1,1,64,64)
        z = torch.cat((x,y),1)  #[1, 2, 64, 64]
        z = self.maxpool(z)  #[1, 2, 32, 32]
        z = self.conv1(z) #[1, 128, 32, 32]
        z = self.bn1(z)
        z = F.gelu(z)
        z = self.maxpool(z) #[1,128,16,16]
        z = self.conv2(z) #[1,256,16,16]
        z = self.bn2(z)
        z = F.gelu(z)

        z = self.convTranspose1(z)  # [1, 128, 32, 32]
        z = F.gelu(z)
        z = self.convTranspose2(z)  # [1, 64, 64, 64]
        z = F.gelu(z)
        z = self.convTranspose3(z)  # [1, 3, 128, 128]
        z = F.gelu(z)
        z = self.upsample(z)
        return z
    
class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel,self).__init__()
        self.sonar_feature_extractor = VGG19FeatureExtractor()
        original_generator = Generator()
        self.originalModelCopy = copy.deepcopy(original_generator)
        self.img_feature_extractor = ImageFeatureExtractor(self.originalModelCopy)
        self.fusion = Fusion()

    def forward(self, cam, sonar):
        x = self.sonar_feature_extractor(sonar)
        y = self.img_feature_extractor(cam)
        z = self.fusion(x, y)
        return z

class DeiTAutoencoder(nn.Module):
    # input [1, 1, 507, 507]
    def __init__(self):
        super(DeiTAutoencoder, self).__init__()
        self.DeiT = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.fc = nn.Linear(1000, 4096)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=32, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=29, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

        self.convT1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)
        self.convT2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)
        self.convT3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=6)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.DeiT(x)
        x = self.fc(x) #[1,4096]

        x = x.view(-1,1,64,64)
        x = self.convT1(x)
        x = F.elu(x)
        x = self.convT2(x)
        x = F.elu(x)
        x = self.convT3(x)
        x = F.sigmoid(x)

        return x