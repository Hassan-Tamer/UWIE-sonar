from keras.models import load_model
import numpy as np
import tf2onnx
import torch
from onnx2pytorch import ConvertModel

def TFtoTorch(hdf5_path,save_path):
    model = load_model(hdf5_path,compile=True)
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    pytorch_model = ConvertModel(onnx_model)

    torch.save(pytorch_model.state_dict(), save_path)
    return pytorch_model

if __name__ == '__main__':
    model = TFtoTorch('densenet121SonarWeights.hdf5','densenet121SonarWeights.pth')
