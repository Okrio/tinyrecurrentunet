import torch
import torch.onnx
import torch.nn.functional as F
import onnx
import json
import argeparse
import os

from network import TRUNet



def load_model(model_path, network_config):
    model = TRUNet(**network_config)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt[model_state_dict])
    model.eval()
    return model



def export_onnx(model,
                export_path,
                time_step, 
                channels, 
                frequency):
    #Create dummy input for tracing
    x = torch.randn((time_step, channels, frequency), requires_grad = False)
    
    #export as onnx model
    torch.onnx.export(model,
                     export_path,
                     export_params = True,
                     opset_version = 15,
                     do_constant_folding=True)
    





if __name__ == "__main__":
    
    parser = argeparse.ArgumentParser()
    parser.add_argument('-c', '--config',    type=str, help = 'Path to config Json file')
    parser.add_argument('-i', '--ckpt_path', type=str, help = 'Path to trained model checkpoints')
    parser.add_argument('-o', '--exp_path',  type=str, help = 'Onnx model export path')
    args = parser.parse_args()
    
    #load jsoin file
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    #get configs
    model_config = config["network"]
    onnx_config = config["onnx_config"]
    
    #load pre-trained model
    trained_model = load_model(args.ckpt_path, 
                               model_config)
    
    #export onnx model
    export(trained_model, 
           args.exp_path, 
           **onnx_config)
