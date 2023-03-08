import torch
import torch.onnx
import torch.nn.functional as F
import onnx
import onnxruntime as rt
import json
import argparse
import os

from network import TRUNet



def load_model(model_path, network_config):
    model = TRUNet(**network_config)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model



def export_onnx(model,
                export_path):
    #Create dummy input for tracing
    x = torch.randn(751, 4, 257)
    print(x.shape)
    #export as onnx model
    torch.onnx.export(model,
                      x,
                     export_path,
                     export_params = True,
                     opset_version = 10,
                     do_constant_folding= True,
                     input_names = ['input'],
                     output_names = ['output'])
    
def optim_onnx(onnx_model_path,
              onnx_opt_path):
    sess_options = rt.SessionOptions()
    
    #set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    #enable model serialization after graph optimization
    sess_options.optimized_model_filepath = onnx_opt_path
    session = rt.InferenceSession(onnx_model_path,  sess_options)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',    type=str, help = 'Path to config Json file')
    parser.add_argument('-i', '--ckpt_path', type=str, help = 'Path to trained model checkpoints')
    parser.add_argument('-o', '--exp_path',  type=str, help = 'Onnx model export path')
    parser.add_argument('-g', '--graph_opt', type=boolean, default = False, help = 'Set to True if onnx graph optimization is required')
    parser.add_argument('-x', '--graph_opt_path', type=str, default = 'content/content/opt_onnx_file.onnx')
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
    export_onnx(trained_model, 
           args.exp_path)
    print('ONNX model exported successfully to {}.'.format(args.exp_path))
    
    if args.graph_opt == True:
         optim_onnx(args.exp_path,
                    args.graph_opt_path)
        print('ONNX graph optimized successfully')
        
