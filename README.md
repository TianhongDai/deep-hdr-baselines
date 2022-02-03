# Deep High Dynamic Range Imaging Benchmark
This repository is the pytorch implementation of various High Dynamic Range (HDR) Imaging algorithms. Please find the details below.

## Maintenance and Contributors
[@TianhongDai](https://github.com/TianhongDai) and [@WeiLi-THU](https://github.com/WeiLi-THU)

## Requirements
- pytorch==1.4.0
- opencv-python
- scikit-image==0.17.2

## ToDo List
- [ ] adaptive padding
- [ ] add more baselines

## Supported Algorthms
- [x] DeepHDR [1]
- [x] NHDRRNet [2]
- [x] AHDR [3]
- [x] DAHDR [4]

## Instruction
1. download the Kalantari dataset via: [[link]](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/), and organize the dataset as follows:
```
dataset
│
└───Traning
│   │  001
│   │  002
│   │  003
│   |  ...
│   
└───Test
    │  001
    │  002
    |  003
    |  ...   
```
2. train the network [unet|nhdrrnet|ahdr|dahdr]:
```bash
python train.py --net-type unet --cuda --batch-size 8 --lr 0.0002
```
3. continue training using the pre-saved checkpoint:
```bash
python train.py --net-type unet --cuda --resume --last-ckpt-path <the saved ckpt path> 
```
4. test the model and save HDR images:
```bash
python eval_metric.py --net-type unet --model-path <the saved ckpt path> --cuda --save-image
```

## Pre-trained Models
The pre-trained models can be downloaded from the [released page](https://github.com/TianhongDai/deep-hdr-baselines/releases/tag/v1.0.0).

## Performance
|            | DeepHDR[1] | NHDRRNet[2] | AHDR[3] | DAHDR[4] |
|:------------:|------------:|-------------:|:-------:|:----------:|
| PSNR-$\mu$ |   42.2695        |    42.4769    |   43.5742    |   43.5240     |
| SSIM-$\mu$ |   0.9941         |    0.9942     |   0.9956     |   0.9956      |
| PSNR-L     |   40.0627        |    40.1978    |   41.1551    |   40.7534     |
| SSIM-L     |   0.9892         |    0.9889     |   0.9903     |   0.9905      |

## Acknowledgements
[@elliottwu](https://github.com/elliottwu) for [DeepHDR](https://github.com/elliottwu/DeepHDR)   
[@qingsenyangit](https://github.com/qingsenyangit) for [AHDRNet](https://github.com/qingsenyangit/AHDRNet)   
[@Galaxies99](https://github.com/Galaxies99) for [NHDRRNet details](https://github.com/Galaxies99/NHDRRNet-pytorch)

## References
[1] [Deep High Dynamic Range Imaging with Large Foreground Motions](https://arxiv.org/abs/1711.08937)  
[2] [Deep HDR Imaging via A Non-Local Network](https://ieeexplore.ieee.org/document/8989959)   
[3] [Attention-guided Network for Ghost-free High Dynamic Range Imaging](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf)  
[4] [Dual-Attention-Guided Network for Ghost-Free High Dynamic Range Imaging](https://link.springer.com/article/10.1007/s11263-021-01535-y)