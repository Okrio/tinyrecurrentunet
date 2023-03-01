# REAL-TIME DENOISING AND DEREVERBERATION WITH TINY RECURRENT U-NET

Unofficial implementation of [REAL-TIME DENOISING AND DEREVERBERATION WTIH TINY RECURRENT U-NET](https://arxiv.org/pdf/2102.03207.pdf) in PyTorch. Tiny Recurrent U-Net (TRU-Net) is a lightweight online inference model that matches the performance of current state-of-the-art models. The size of the quantized version of TRU-Net is 362 kilobytes (~300k parameters), which is small enough to be deployed on edge devices. In addition, the small-sized model with a new masking method called phase-aware β-sigmoid mask enables simultaneous denoising and dereverberation.

## Requirements

Create and activate a virtual environment and install dependencies.

```
pip install -r requirements.txt
```

## Dataset

- The code uses [Microsoft DNS 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.13981.pdf) dataset. The dataset, pre-processing codes, and instruction to generate training data can be found in [this link](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master). Assume the dataset is stored under ```./dns```. Prior to generating clean-noisy data pairs, to comply with the paper's configurations, the following parameters in their ```noisyspeech_synthesizer.cfg``` file: 
```
total_hours: 300, 
snr_lower: -5, 
snr_upper: 25, 
total_snrlevels: 30
```


```
Training set directory: 
./training_dataset/clean/fileid_{0..10000}.wav
./training_dataset/noisy/fileid_{0..10000}.wav
./training_dataset/noise/fileid_{0..10000}.wav
```

Generate training data: 
```
python noisyspeech_synthesizer_singleprocess.py
```
## Training



## Denoising


## Evaluation


## 

