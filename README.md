## Gaussian MAE (WIP)

<img src="./assets/gmae.png" width="650px"></img>

<a href="https://arxiv.org/abs/2501.03229v1">Gaussian MAE</a> explores masked autoencoders (MAE) with gaussian splatting. It enables some zero shot capabilities via mid level gaussian based representations

To use the vanilla mae use the following
```python
from src.mae import MAE

dtype = torch.bfloat16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gmae = MAE(
    image_size = 256,
    channels = 3,
    patch_size = 4,
    masking_ratio = 0.75,

    # encoder configs
    encoder_dim = 512,
    encoder_depth = 8,
    encoder_heads = 8,
    encoder_dim_head = 64,

    # decoder configs
    decoder_dim = 512,
    decoder_depth = 8,
    decoder_heads = 8,
    decoder_dim_head = 64
).to(device, dtype)

imgs = torch.randn(7, 3, 256, 256).to(device, dtype)

loss = gmae(imgs)
```

To download and extract the dataset used for training run `source scripts/download.sh`

### Installation

```bash
# create conda env and install torch
conda create --name gmae -y python=3.10
conda activate gmae
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git
```