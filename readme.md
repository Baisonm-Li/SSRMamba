# SSRMamba

Code of KSSANet: "SSRMamba: Efficient Visual State Space Model for Spectral Super-Resolution"

Spectral super-resolution, which reconstructs hy-
perspectral images (HSI) from a single RGB image, has gar-
nered increasing attention. Due to the limitations of CNN
structures in spectral modeling and the high computational
cost of Transformer structures, existing deep learning (DL)-
based methods struggle to balance spectral reconstruction quality
and computational efficiency. Recently, Mamba methods base
on state-space models (SSM) show great potential in model-
ing long-range dependencies with linear complexity. Therefore,
we introduce the Mamba model into spectral super-resolution
(SSR) task. Specifically, we propose a three-stage SSR network
base on Mamba, called SSRMamba. We design SpaMamba,
SSMamba, and SpeMamba modules for shallow spatial infor-
mation extraction, mixed information encoding, and spectral
information reconstruction, respectively. Extensive experimental
results demonstrate that SSRMamba not only surpasses existing
methods in terms of quantification and quality, achieving state-of-
the-art (SOTA) performance, but also significantly reduces model
size and computational cost.  

## Requirements
- Python 3.8+
- PyTorch 1.4+
- CUDA 10.1+
- torchvision 0.5+
- h5py 2.10+
- matplotlib 3.2+

## Datasets
- CAVE: https://www.cs.columbia.edu/CAVE/databases/multispectral/
- Pavia: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University
- Havard https://vision.seas.harvard.edu/hyperspec/download.html
## Usage
```
python train.py --dataset CAVE --batch_size 32 
```

## Acknowledgement
Our code references [efficient-kan](https://github.com/Blealtan/efficient-kan.git) and [pykan](https://github.com/KindXiaoming/pykan.git). Thanks for their greak work!