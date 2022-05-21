PyTorch code for **Visualizing and Understanding Self-Supervised Vision Learning**

## Gradio Web-Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Annon/xai-self-supervised)
Note: For methods which require several runs of the model (such as Perturbation methods and Pixel Invariance), the [Gradio](https://gradio.app/) web-demo is a bit slow (~ 2 min) since the [HuggingFace Spaces](https://huggingface.co/spaces) only provide CPUs. For ultimate speed, we recomend using the jupyter notebook provided here with a GPU. 

To experiment with this code, simply open the jupyter notebook `xai.ipynb`.

Either download [ImageNet](https://image-net.org/download.php) validation set (in that case you will have the images in `val/images/`) or use an image of your interest. In that case, please change the path in jupyter file to point to your image of interest. 

For SimCLRv2 models, please download from [here](https://drive.google.com/drive/folders/1mw5o_6kzYNnI-IJAUgNYDGFNV8ig3Rer?usp=sharing) and place them in `pretrained_models/simclr2_models/`
