#!/usr/bin/env python3
"""Pass input directly to output.

https://github.com/PortAudio/portaudio/blob/master/test/patest_wire.c

This script allows to test the trained model in a streaming scenario in Python
"""
import argparse
import json
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
import torch
import torchaudio
assert numpy  # avoid "imported but unused" message (W0611)


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
parser.add_argument('-i', '--input-device', type=int_or_str, help='input device (numeric ID or substring)')
parser.add_argument('-o', '--output-device', type=int_or_str, help='output device (numeric ID or substring)')
parser.add_argument('-c', '--channels', type=int, default=2, help='number of channels')
parser.add_argument('--dtype', help='audio data type')
parser.add_argument('--samplerate', type=float, help='sampling rate')
parser.add_argument('--blocksize', type=int, help='block size')
parser.add_argument('--latency', type=float, help='latency in seconds')
parser.add_argument('--model_path', type=str, help='path to model')
parser.add_argument('--model_cofig', type=str, help='JSON file encompassing model config')
args = parser.parse_args(remaining)

#load model function
def load_model(model_path, network_config):
    model = TRUNet(**network_config)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def preprocess():
    """
    convert time-domain audio to features of shape(freq, 4, time)
    """
    pass

def postprocess():
    """
    convert features back to time-domain audio
    """
    pass


#get model configs from JSON file
with open(args.model_config) as f:
    data = f.read()
config = json.loads()
model_config = config["network"]

#Instantiate model
model = load_model(args.model_path, model_config)

#stream audio through model
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    #convert numpy audio-streams to features
    features = preprocess(indata)
    
    #feed-forward audio to model
    denoise_features = model(features)
    
    #reconstruct denoised audio from features
    denoised_audio = posprocess(denoised_features).numpy()
    
    #out to stream
    outdata[:] = denoise_audio

try:    
    with sd.Stream(device=(args.input_device, 
                           args.output_device),
                           samplerate=args.samplerate, 
                           blocksize=args.blocksize,
                           dtype=args.dtype, 
                           latency=args.latency,
                           channels=args.channels,
                           callback=callback):
        
        print('press Return to quit')
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
