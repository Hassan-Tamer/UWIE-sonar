import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class ResizeSonar(nn.Module):
    def __init__(self):
        super(ResizeSonar, self).__init__()
        self.convlayer1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.convlayer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=31, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(1)


    def forward(self, x):
        x = self.convlayer1(x)
        x = self.bn1(x)
        x = nn.functional.gelu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.convlayer2(x)
        x = self.bn2(x)
        x = nn.functional.gelu(x)        
        
        return x
    
class SonarExtractor(nn.Module):
    def __init__(self,sonar_base_model):
        super(SonarExtractor, self).__init__()
        self.base_model = sonar_base_model
        self.resizeModel = ResizeSonar()

    def forward(self,x):
        x = self.resizeModel(x)
        x = self.base_model(x)
        return x

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        print(vgg19)
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
    

class modifiedGenerator(nn.Module):
    def __init__(self,original_generator):
        super(modifiedGenerator, self).__init__()
        self.model = original_generator

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.model(x)
        return x