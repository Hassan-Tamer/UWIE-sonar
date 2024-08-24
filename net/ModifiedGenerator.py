import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)  # [B, 3, H, W] -> [B, 64, H/2, W/2]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # [B, 128, H/4, W/4] -> [B, 256, H/8, W/8]
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # [B, 256, H/8, W/8] -> [B, 128, H/4, W/4]
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # [B, 128, H/4, W/4] -> [B, 64, H/2, W/2]
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # [B, 64, H/2, W/2] -> [B, 3, H, W]
    
    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class modifiedGenerator(nn.Module):
    def __init__(self,original_generator):
        super(modifiedGenerator, self).__init__()
        self.model = original_generator
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.model(x)
        return x