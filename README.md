# REAL-TIME DENOISING AND DEREVERBERATION WTIH TINY RECURRENT U-NET

Unofficial implementation of [REAL-TIME DENOISING AND DEREVERBERATION WTIH TINY RECURRENT U-NET](https://arxiv.org/pdf/2102.03207.pdf) in PyTorch. Tiny Recurrent U-Net (TRU-Net) is a lightweight online inference model that matches the performance of current state-ofthe-art models. The size of the quantized version of TRU-Net is 362
kilobytes, which is small enough to be deployed on edge devices. In addition, the small-sized model with a new masking method called phase-aware β-sigmoid mask enables simultaneous denoising and dereverberation.

## Training

Assuming the dataset is stored in the waveform formate with the structure below:
```
Training sets: 
./dns/training_dataset/clean/fileid_{0..59999}.wav
./dns/training_dataset/noisy/fileid_{0..59999}.wav
./dns/training_dataset/noise/fileid_{0..59999}.wav
```
