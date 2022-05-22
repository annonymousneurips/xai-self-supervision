PyTorch code for **Visualizing and Understanding Self-Supervised Vision Learning**

## Gradio Web-Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Annon/xai-self-supervised)
**Note**: Hugging Face Spaces only provide CPUs so some methods are slow in the web-demo. We recomend using Google Colab below with a GPU:
## Google Colab [![Open In Colab](https://github.com/amrzv/awesome-colab-notebooks/blob/main/images/colab.svg)](https://colab.research.google.com/drive/1C3io30vzdGhxywhapJYE-lsITYLofhAe?usp=sharing)
## Locally
To experiment with this code, clone this repo and then open the jupyter notebook `xai.ipynb`.

Either download [ImageNet](https://image-net.org/download.php) validation set (in that case you will have the images in `val/images/`) or use an image of your interest. In that case, please change the path in jupyter file to point to your image of interest. 

For SimCLRv2 models, please download from [here](https://drive.google.com/drive/folders/1mw5o_6kzYNnI-IJAUgNYDGFNV8ig3Rer?usp=sharing) and place them in `pretrained_models/simclr2_models/`
