import os
import torch
from net.Ushape_Trans import *
from net.utils import *
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from loss.LAB import *
from loss.LCH import *
from net.ModifiedGenerator import *
from matplotlib import pyplot as plt
from torchvision.utils import save_image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)

original_generator = Generator().cuda()
original_generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth"))

modified_generator = modifiedGenerator(original_generator).cuda()

modified_generator.eval()
modified_generator.cuda()
rando = torch.randn(1, 4, 256, 256).cuda()
output = modified_generator(rando)

out=output[3].data
save_image(out, "hassan.png")