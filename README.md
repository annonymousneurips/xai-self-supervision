PyTorch code for **Visualizing and Understanding Self-Supervised Vision Learning**

## Gradio Web-Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Annon/xai-self-supervised)
**Note**: Hugging Face Spaces only provide CPUs so some methods are slow in the web-demo. We recomend using Google Colab below with a GPU:
## Google Colab [![Open In Colab](https://github.com/amrzv/awesome-colab-notebooks/blob/main/images/colab.svg)](https://colab.research.google.com/drive/1C3io30vzdGhxywhapJYE-lsITYLofhAe?usp=sharing)
## Locally
To experiment with this code, clone this repo and then open the jupyter notebook `xai.ipynb`.

Either download [ImageNet](https://image-net.org/download.php) validation set (in that case you will have the images in `val/images/`) or use an image of your interest. In that case, please change the path in jupyter file to point to your image of interest. 

## Models
You may download some self-supervised models to experiment with:

| Model | link | location |
| --- | --- | --- |
| SimCLRv2 (1x) | [link](https://drive.google.com/file/d/1c2Hl_uutm9IssG8TdpI0b3d2PqB5VHyQ/view?usp=sharing)| `pretrained_models/simclr2_models/` |
| SimCLRv2 (2x) | [link](https://drive.google.com/file/d/1028oGnbdFg-SzYetrGPFb9g6mSVA5dRL/view?usp=sharing) | `pretrained_models/simclr2_models/` |
| Barlow Twins | [link](https://drive.google.com/file/d/18l3Z-OHMD-b5Eo8_dCXOu_hjLNZQf5he/view?usp=sharing) | `pretrained_models/barlow_models/` |
| SimSiam | [link](https://drive.google.com/file/d/1u5xsaitKtQXMiD4Wg9hItei8y0DBGEXP/view?usp=sharing) |`pretrained_models/simsiam_models/` |

In `xai.ipynb`, simply set `network` to one of the following: `simclrv2`, `barlow_twins`, `simsiam`

For all models other than SimCLRv2, please change the hyperparameters in `xai.ipynb` according to below:

| Method | Hyperparameters |
| --- | --- |
| Feature Visualization | `up_until = 5`, `reg_l2 = False` |
| Pixel Invariance | `epochs = 10` |
