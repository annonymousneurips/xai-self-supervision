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

## Results for SimCLRv2 (2X)
The difference between SimCLRv2 (1X) (which is evaluated in Table 1 in the paper) and SimCLR (2X) is that SimCLR (2X) doubles the dimension of the CNN

|                            |  SI↑  |  SD↓  |  CI↑  |  CD↓  |  SAD↓ |  SIC↑ |  CAD↓ |  CID↑ |  MS↓  |
|:--------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|    Conditional Occlusion   | 0.795 | 0.714 | 0.644 | 0.215 | 0.166 | 0.223 | 0.282 | 0.095 |   NA  |
| Context-Agnostic Occlusion | 0.793 | 0.724 | 0.639 | 0.207 | 0.167 | 0.208 | 0.298 | 0.082 |   NA  |
|     Pairwise Occlusion     | 0.763 | 0.643 | 0.636 | 0.276 | 0.380 | 0.068 | 0.583 | 0.013 |   NA  |
|          Grad-CAM          | 0.798 | 0.736 | 0.663 | 0.213 | 0.170 | 0.212 | 0.303 | 0.059 | 2.419 |
|   Interaction-CAM (Mean)   | 0.795 | 0.719 | 0.653 | 0.215 | 0.203 | 0.180 | 0.373 | 0.049 | 3.748 |
|    Interaction-CAM (Max)   | 0.805 | 0.730 | 0.663 | 0.214 | 0.195 | 0.195 | 0.441 | 0.034 | 2.716 |
|    Interaction-CAM (SA)    | 0.802 | 0.736 | 0.654 | 0.221 | 0.168 | 0.248 | 0.611 | 0.021 | 18.81 |
|      Pixel Invariance      |   NA  |   NA  | 0.692 | 0.194 |   NA  |   NA  | 0.000 | 0.800 |   NA  |
|      Input x Gradient      | 0.889 | 0.844 | 0.578 | 0.102 | 0.087 | 0.173 | 0.940 | 0.000 | 1.891 |
|    Input x Gradient (G)    | 0.905 | 0.868 | 0.724 | 0.069 | 0.076 | 0.175 | 0.798 | 0.001 | 2.277 |
|         Smooth-Grad        | 0.884 | 0.858 | 0.637 | 0.070 | 0.117 | 0.147 | 0.837 | 0.000 | 1.252 |
|       Smooth-Grad (G)      | 0.882 | 0.848 | 0.691 | 0.114 | 0.104 | 0.155 | 0.734 | 0.002 | 1.838 |
|       Avg. Transforms      | 0.892 | 0.853 | 0.578 | 0.100 | 0.077 | 0.200 | 0.938 | 0.000 | 1.925 |
|     Avg. Transforms (G)    | 0.907 | 0.878 | 0.726 | 0.069 | 0.064 | 0.218 | 0.794 | 0.001 | 2.407 |
|     Avg. Transforms (N)    | 0.844 | 0.779 | 0.601 | 0.091 | 0.153 | 0.167 | 0.916 | 0.000 | 0.976 |
|   Avg. Transforms (G + N)  | 0.868 | 0.823 | 0.711 | 0.078 | 0.153 | 0.193 | 0.782 | 0.001 | 1.007 |

## Results for Barlow Twins
|                            |  SI↑  |  SD↓  |  CI↑  |  CD↓  |  SAD↓ |  SIC↑ |  CAD↓ |  CID↑ |   MS↓  |
|:--------------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|
|    Conditional Occlusion   | 0.657 | 0.424 | 0.535 | 0.176 | 0.508 | 0.140 | 0.547 | 0.121 |   NA   |
| Context-Agnostic Occlusion | 0.651 | 0.429 | 0.515 | 0.172 | 0.441 | 0.170 | 0.475 | 0.133 |   NA   |
|     Pairwise Occlusion     | 0.648 | 0.501 | 0.528 | 0.186 | 0.557 | 0.092 | 0.693 | 0.028 |   NA   |
|          Grad-CAM          | 0.683 | 0.498 | 0.561 | 0.179 | 0.385 | 0.178 | 0.431 | 0.114 | 143.50 |
|   Interaction-CAM (Mean)   | 0.673 | 0.475 | 0.556 | 0.169 | 0.413 | 0.163 | 0.446 | 0.099 | 110.16 |
|    Interaction-CAM (Max)   | 0.689 | 0.483 | 0.564 | 0.169 | 0.425 | 0.158 | 0.531 | 0.081 |  79.06 |
|    Interaction-CAM (SA)    | 0.686 | 0.496 | 0.556 | 0.173 | 0.452 | 0.155 | 0.704 | 0.045 |  94.67 |
|      Pixel Invariance      |   NA  |   NA  | 0.453 | 0.375 |   NA  |   NA  | 0.000 | 0.558 |   NA   |
|      Input x Gradient      | 0.724 | 0.587 | 0.489 | 0.119 | 0.471 | 0.038 | 0.979 | 0.001 | 16.554 |
|    Input x Gradient (G)    | 0.822 | 0.602 | 0.699 | 0.068 | 0.465 | 0.047 | 0.940 | 0.001 | 578.03 |
|         Smooth-Grad        | 0.750 | 0.576 | 0.533 | 0.089 | 0.459 | 0.035 | 0.972 | 0.002 |  12.82 |
|       Smooth-Grad (G)      | 0.829 | 0.607 | 0.709 | 0.076 | 0.449 | 0.057 | 0.933 | 0.001 |  10.02 |
|       Avg. Transforms      | 0.728 | 0.593 | 0.488 | 0.117 | 0.431 | 0.047 | 0.980 | 0.002 |  14.53 |
|     Avg. Transforms (G)    | 0.826 | 0.619 | 0.702 | 0.069 | 0.450 | 0.050 | 0.939 | 0.002 |  81.72 |
|     Avg. Transforms (N)    | 0.716 | 0.554 | 0.500 | 0.112 | 0.513 | 0.042 | 0.974 | 0.002 |  12.46 |
|   Avg. Transforms (G + N)  | 0.828 | 0.597 | 0.705 | 0.070 | 0.447 | 0.047 | 0.936 | 0.002 |  25.16 |

