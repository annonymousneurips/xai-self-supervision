import torch
import numpy as np
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
from scipy.ndimage.filters import gaussian_filter1d
import scipy.ndimage as nd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, nn.ReLU):
        return (F.relu(grad_in[0]),)
    
def blur_sailency(input_image):
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def occlusion(img1, img2, model, w_size = 64, stride = 8, batch_size = 32):
    
    measure = nn.CosineSimilarity(dim=-1) 
    output_size = int(((img2.size(-1) - w_size) / stride) + 1)
    out1_condition, out2_condition = model(img1), model(img2)
    images1 = []
    images2 = []

    for i in range(output_size):
        for j in range(output_size):
            start_i, start_j = i * stride, j * stride
            image1 = img1.clone().detach()
            image2 = img2.clone().detach()
            image1[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            image2[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            images1.append(image1)
            images2.append(image2)

    images1 = torch.cat(images1, dim=0).to(device)
    images2 = torch.cat(images2, dim=0).to(device)

    score_map1 = []
    score_map2 = []

    assert images1.shape[0] == images2.shape[0]

    for b in range(0, images2.shape[0], batch_size):

        with torch.no_grad():
            out1 = model(images1[b : b + batch_size, :])
            out2 = model(images2[b : b + batch_size, :])

        score_map1.append(measure(out1, out2_condition))  # try torch.mm(out2_condition, out1.t())[0]
        score_map2.append(measure(out1_condition, out2))  # try torch.mm(out1_condition, out2.t())[0]

    score_map1 = torch.cat(score_map1, dim = 0)   
    score_map2 = torch.cat(score_map2, dim = 0)    
    assert images2.shape[0] == score_map2.shape[0] == score_map1.shape[0]

    heatmap1 = score_map1.view(output_size, output_size).cpu().detach().numpy()
    heatmap2 = score_map2.view(output_size, output_size).cpu().detach().numpy()
    base_score = measure(out1_condition, out2_condition)

    heatmap1 = (heatmap1 - base_score.item()) * -1   # or base_score.item() - heatmap1. The higher the drop, the better
    heatmap2 = (heatmap2 - base_score.item()) * -1   # or base_score.item() - heatmap2. The higher the drop, the better
    
    return heatmap1, heatmap2

def occlusion_context_agnositc(img1, img2, model, w_size = 64, stride = 8, batch_size = 32):
    
    measure = nn.CosineSimilarity(dim=-1) 
    output_size = int(((img2.size(-1) - w_size) / stride) + 1)
    out1_condition, out2_condition = model(img1), model(img2)

    images1_occlude_mask = []
    images2_occlude_mask  = []

    for i in range(output_size):
        for j in range(output_size):
            start_i, start_j = i * stride, j * stride
            image1 = img1.clone().detach()
            image2 = img2.clone().detach()
            image1[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            image2[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = 0  
            images1_occlude_mask.append(image1)
            images2_occlude_mask.append(image2)

    images1_occlude_mask = torch.cat(images1_occlude_mask, dim=0).to(device)
    images2_occlude_mask = torch.cat(images2_occlude_mask, dim=0).to(device)

    images1_occlude_backround = []
    images2_occlude_backround = []

    copy_img1 = img1.clone().detach()
    copy_img2 = img2.clone().detach()

    for i in range(output_size):
        for j in range(output_size):
            start_i, start_j = i * stride, j * stride

            image1 = torch.zeros_like(img1)
            image2 = torch.zeros_like(img2)

            image1[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = copy_img1[:, :, start_i : start_i + w_size, start_j : start_j + w_size]
            image2[:, :, start_i : start_i + w_size, start_j : start_j + w_size] = copy_img2[:, :, start_i : start_i + w_size, start_j : start_j + w_size]

            images1_occlude_backround.append(image1)
            images2_occlude_backround.append(image2)

    images1_occlude_backround = torch.cat(images1_occlude_backround, dim=0).to(device)
    images2_occlude_backround = torch.cat(images2_occlude_backround, dim=0).to(device)

    score_map1 = []
    score_map2 = []

    assert images1_occlude_mask.shape[0] == images2_occlude_mask.shape[0]

    for b in range(0, images1_occlude_mask.shape[0], batch_size):

        with torch.no_grad():
            out1_mask = model(images1_occlude_mask[b : b + batch_size, :])
            out2_mask = model(images2_occlude_mask[b : b + batch_size, :])
            out1_backround = model(images1_occlude_backround[b : b + batch_size, :])
            out2_backround = model(images2_occlude_backround[b : b + batch_size, :])

        out1 = out1_backround - out1_mask
        out2 = out2_backround - out2_mask
        score_map1.append(measure(out1, out2_condition))  # or torch.mm(out2_condition, out1.t())[0]
        score_map2.append(measure(out1_condition, out2))  # or torch.mm(out1_condition, out2.t())[0]

    score_map1 = torch.cat(score_map1, dim = 0)   
    score_map2 = torch.cat(score_map2, dim = 0)    
    assert images1_occlude_mask.shape[0] == images2_occlude_mask.shape[0] == score_map2.shape[0] == score_map1.shape[0]

    heatmap1 = score_map1.view(output_size, output_size).cpu().detach().numpy()
    heatmap2 = score_map2.view(output_size, output_size).cpu().detach().numpy()

    heatmap1 = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())
    heatmap2 = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min())
    
    return heatmap1, heatmap2

def pairwise_occlusion(img1, img2, model, batch_size, erase_scale, erase_ratio, num_erases):

    measure = nn.CosineSimilarity(dim=-1) 
    out1_condition, out2_condition = model(img1), model(img2)
    baseline = measure(out1_condition, out2_condition).detach()
    # a bit sensitive to scale and ratio. erase_scale is from (scale[0] * 100) % to (scale[1] * 100) %
    random_erase = transforms.RandomErasing(p=1.0, scale=erase_scale, ratio=erase_ratio)  
    
    image1 = img1.clone().detach()
    image2 = img2.clone().detach()
    images1 = []
    images2 = []

    for _ in range(num_erases):
        images1.append(random_erase(image1))
        images2.append(random_erase(image2))

    images1 = torch.cat(images1, dim=0).to(device)
    images2 = torch.cat(images2, dim=0).to(device)
    
    sims = []
    weights1 = []
    weights2 = []

    for b in range(0, images2.shape[0], batch_size):

        with torch.no_grad():
            out1 = model(images1[b : b + batch_size, :])
            out2 = model(images2[b : b + batch_size, :])
            sims.append(measure(out1, out2))
            weights1.append(out1.norm(dim=-1))
            weights2.append(out2.norm(dim=-1))

    sims = torch.cat(sims, dim = 0)       
    weights1, weights2 = torch.cat(weights1, dim = 0).cpu().numpy(), torch.cat(weights2, dim = 0).cpu().numpy()
    weights = list(zip(weights1, weights2))
    sims = baseline - sims   # the higher the drop, the better
    sims = F.softmax(sims, dim = -1)
    sims = sims.cpu().numpy()

    assert sims.shape[0] == images1.shape[0] == images2.shape[0]
    A1 = np.zeros((224, 224))
    A2 = np.zeros((224, 224))

    for n in range(images1.shape[0]):

        im1_2d = images1[n].cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)
        im2_2d = images2[n].cpu().numpy().transpose((1, 2, 0)).sum(axis=-1)

        joint_similarity = sims[n]
        weight = weights[n]

        if weight[0] < weight[1]:
            A1[im1_2d == 0] += joint_similarity
        else:
            A2[im2_2d == 0] += joint_similarity

    A1 = A1 / (np.max(A1) + 1e-9)  
    A2 = A2 / (np.max(A2) + 1e-9)

    return A1, A2

def tv_reg(img, l1 = True):
    
    diff_i = (img[:, :, :, 1:] - img[:, :, :, :-1])
    diff_j = (img[:, :, 1:, :] - img[:, :, :-1, :])
    
    if l1:
        return diff_i.abs().sum() + diff_j.abs().sum()
    else:
        return diff_i.pow(2).sum() + diff_j.pow(2).sum()
    
def dream(image1, image2, model, iterations, lr, optimize_score, ema, reg_l2, reg_l2_weight, blur, lr_norm, use_tv, tv_weight, 
          minmax_weight):

    smoothing_coefficient = 0.5
    measure = nn.CosineSimilarity(dim=-1)
    image1 = torch.from_numpy(image1).to(device).requires_grad_()
    image2 = torch.from_numpy(image2).to(device)

    for i in range(iterations):
        
        model.zero_grad()
        out1, out2 = model(image1), model(image2)  
        
        # use out[:,channel].norm() to optimize an image for a specific channel.
        if optimize_score:
            loss = measure(out1.view(1, -1), out2.view(1, -1))
            loss += minmax_weight * ((out1.view(1, -1) - out2.view(1, -1))**2).mean()
        else:
            ema_output = ema * out1.data + (1 - ema) * out2.data
            out1.data.copy_(ema_output)
            loss = out1.norm()     
           
        if reg_l2:
            loss += - (reg_l2_weight * torch.norm(image1))

        if use_tv:
            loss += - (tv_weight * tv_reg(image1, l1 = False))
            
        loss.backward()
        
        if blur:   
            # you can also apply scipy.ndimage.gaussian_filter for each channel seperately and then concatenate --> same thing
            X_np = image1.grad.data.cpu().numpy()
            sigma = ((i + 1) / iterations) * 2.0 + smoothing_coefficient
            X_np = gaussian_filter1d(X_np, sigma, axis=2)
            X_np = gaussian_filter1d(X_np, sigma, axis=3)
            image1.grad.data.copy_(torch.from_numpy(X_np).type_as(image1.grad.data))
        
        norm_lr = lr
        if lr_norm:
            avg_grad = np.abs(image1.grad.data.cpu().numpy()).mean() 
            norm_lr = lr / avg_grad
            
        image1.data += norm_lr * image1.grad.data
        
        image1.data = image1.data.clip(0, 1) 
        image1.grad.data.zero_()
        
    return image1.cpu().data.numpy()

def dissect_resnet(ssl_model, up_until):
    
    resnet = ssl_model.encoder.net
    resnet = list(resnet.children())[:up_until]
    resnet = nn.Sequential(*resnet)
    return resnet

def deepdream(img1, img2, ssl_model, optimize_score, up_until, ema, reg_l2, reg_l2_weight, use_tv, tv_weight, minmax_weight, 
              blur, iterations, lr, lr_norm, octave_scale, num_octaves, init_scale):
    """
    optimize_score: whether to optimize the constrastive score of the features of the two images
    up_until: dissect the resnet up until that layer
    ema: exponentially moving average from other image features when using maximizing features. This applied when 
    optimize_score = False. The lower, the more to take from the other image
    reg_l2: whether to use l2 reg to penalize high values
    reg_l2_weight: reg_l2 weight if using it
    blur: whether to apply blurring at each optimization step
    iterations: number of iterations 
    lr: the learning rate
    lr_norm: whether to scale the learning rate (like Adam, but scales simply by the average)
    num_octaves: how many different scaled to take in the pyramid. For each scale, we run the optimization. 
    Algorithm is from deep dream and showed much better results
    use_tv: whether to us total variation regularizer to encourage smoothness
    tv_weight: the weight of total variation if applied
    minmax_weight: the weight to maximize the distances of the generated image and the other image
    init_scale: multiply this value with the randomly initialized image from the uniform distribution (only applies when mask = False)
    """
    
    dd_model = dissect_resnet(ssl_model, up_until)
    image = init_scale * torch.rand(1,3,224,224)  
    image = image.cpu().numpy()
    image2 = img2.cpu().numpy() 

    octaves = [image]
    octaves_images2 = [image2]

    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))
        octaves_images2.append(nd.zoom(octaves_images2[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    octaves_images2 = octaves_images2[::-1]
    detail = np.zeros_like(octaves[-1])

    for octave, octave_base in enumerate(octaves[::-1]):

        if octave > 0:
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)

        input_image = octave_base + detail

        dreamed_image = dream(image1 = input_image, 
                              image2 = octaves_images2[octave], 
                              model = dd_model, 
                              iterations = iterations, 
                              lr = lr, 
                              optimize_score = optimize_score, 
                              ema = ema, 
                              reg_l2 = reg_l2, 
                              reg_l2_weight = reg_l2_weight,
                              blur = blur, 
                              lr_norm = lr_norm, 
                              use_tv = use_tv,   
                              tv_weight = tv_weight, 
                              minmax_weight = minmax_weight)   

        detail = dreamed_image - octave_base
        
    return dreamed_image, detail

def synthesize(ssl_model, model_type, img1, img_cls_layer, lr, l2_weight, alpha_weight, alpha_power, tv_weight, init_scale, network):
    
    if model_type == 'imagenet':
        reduce_lr = False  
        model = torchvision.models.resnet50(pretrained=True)
        model = list(model.children())[:img_cls_layer]    
        model = nn.Sequential(*model).to(device)
        model.eval()
    else:
        reduce_lr = True
        shift_layer = 3 if network == 'simclrv2' else 0
        equivalent_layer = img_cls_layer - shift_layer  
        model = list(ssl_model.encoder.net.children())[:equivalent_layer]
        model = nn.Sequential(*model).to(device)
        model.eval()

    opt_img = (init_scale * torch.randn(1, 3, 224, 224)).to(device).requires_grad_()
    target_feats = model(img1).detach()
    optimizer = torch.optim.SGD([opt_img], lr=lr, momentum=0.9)

    for i in range(201):
        opt_img.data = opt_img.data.clip(0,1)
        optimizer.zero_grad()
        output = model(opt_img)
        l2_loss = l2_weight * ((output - target_feats) ** 2).sum() / (target_feats ** 2).sum()
        reg_alpha = alpha_weight * (opt_img ** alpha_power).sum()
        reg_total_variation = tv_weight * tv_reg(opt_img, l1 = False)
        loss = l2_loss + reg_alpha + reg_total_variation
        loss.backward()
        optimizer.step()

        if reduce_lr and i % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1/10
                
    return opt_img

def get_difference(ssl_model, baseline, image, lr, l2_weight, alpha_weight, alpha_power, tv_weight, init_scale, network):

    imagenet_images = []
    ssl_images = []

    for lay in range(4,7):
        image_net_image = synthesize(ssl_model, baseline, image, lay, lr, l2_weight, alpha_weight, alpha_power, tv_weight, init_scale, network).detach().clone()
        ssl_image = synthesize(ssl_model, 'ssl', image, lay, lr, l2_weight, alpha_weight, alpha_power, tv_weight, init_scale, network).detach().clone()
        imagenet_images.append(image_net_image)
        ssl_images.append(ssl_image)
        
    return imagenet_images, ssl_images

def create_mixed_images(transform_type, ig_transforms, step, img_path, add_noise):

    img = Image.open(img_path).convert('RGB')
    img1 = ig_transforms['pure'](img).unsqueeze(0).to(device)
    img2 = ig_transforms[transform_type](img).unsqueeze(0).to(device)

    lambdas = np.arange(1,0,-step)
    mixed_images = []
    for l,lam in enumerate(lambdas):
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_images.append(mixed_img)
        
    if add_noise:
        sigma = 0.15 / (torch.max(img1) - torch.min(img1)).item()
        mixed_images = [im + torch.zeros_like(im).normal_(0, sigma) if (n>0) and (n<len(mixed_images)-1) else im for n,im in enumerate(mixed_images)]
        
    return mixed_images

def averaged_transforms(guided, ssl_model, mixed_images, blur_output):

    measure = nn.CosineSimilarity(dim=-1)

    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    grads1 = []
    grads2 = []

    for xbar_image in mixed_images[1:]:  
        input_image1 = mixed_images[0].clone().requires_grad_()
        input_image2 = xbar_image.clone().requires_grad_()

        if input_image1.grad is not None:
            input_image1.grad.data.zero_()
            input_image2.grad.data.zero_()

        score = measure(ssl_model(input_image1), ssl_model(input_image2))
        score.backward()
        grads1.append(input_image1.grad.data)
        grads2.append(input_image2.grad.data)

    grads1 = torch.cat(grads1).mean(0).unsqueeze(0)
    grads2 = torch.cat(grads2).mean(0).unsqueeze(0)

    sailency1, _ = torch.max((mixed_images[0] * grads1).abs(), dim=1)
    sailency2, _ = torch.max((mixed_images[-1] * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

def sailency(guided, ssl_model, img1, img2, blur_output):
    
    measure = nn.CosineSimilarity(dim=-1)
    
    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    input_image1 = img1.clone().requires_grad_()
    input_image2 = img2.clone().requires_grad_()
    score = measure(ssl_model(input_image1), ssl_model(input_image2))
    score.backward()
    grads1 = input_image1.grad.data
    grads2 = input_image2.grad.data   
    sailency1, _ = torch.max((img1 * grads1).abs(), dim=1)
    sailency2, _ = torch.max((img2 * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

def smooth_grad(guided, ssl_model, img1, img2, blur_output, steps = 50):
    
    measure = nn.CosineSimilarity(dim=-1)
    sigma = 0.15 / (torch.max(img1) - torch.min(img1)).item()
    
    if guided:
        handles = []
        for i, module in enumerate(ssl_model.modules()):
            if isinstance(module, nn.ReLU):
                handles.append(module.register_backward_hook(relu_hook_function))
                
    noise_images1 = []
    noise_images2 = []
    
    for _ in range(steps):
        noise = torch.zeros_like(img1).normal_(0, sigma)
        noise_images1.append(img1 + noise)   
        noise_images2.append(img2 + noise)
                
    grads1 = []
    grads2 = []

    for n1, n2 in zip(noise_images1, noise_images2):  
        input_image1 = n1.clone().requires_grad_()
        input_image2 = n2.clone().requires_grad_()

        if input_image1.grad is not None:
            input_image1.grad.data.zero_()
            input_image2.grad.data.zero_()

        score = measure(ssl_model(input_image1), ssl_model(input_image2))
        score.backward()
        grads1.append(input_image1.grad.data)
        grads2.append(input_image2.grad.data)

    grads1 = torch.cat(grads1).mean(0).unsqueeze(0)
    grads2 = torch.cat(grads2).mean(0).unsqueeze(0)   
    sailency1, _ = torch.max((img1 * grads1 ).abs(), dim=1)
    sailency2, _ = torch.max((img2 * grads2).abs(), dim=1)

    if guided:     # remove handles after finishing
        for handle in handles:
            handle.remove()
            
    if blur_output:
        sailency1 = blur_sailency(sailency1)
        sailency2 = blur_sailency(sailency2)
            
    return sailency1, sailency2

def get_pixel_invariance_dataset(img_path, num_augments, batch_size, no_shift_transforms, ssl_model):
    
    measure = nn.CosineSimilarity(dim=-1)
    img = Image.open(img_path).convert('RGB')
    no_shift_aug = transforms.Compose([no_shift_transforms['aug'], 
                                       transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))]) 

    augments2 =  []
    labels = []

    for _ in range(num_augments):
        augments2.append(no_shift_aug(img).unsqueeze(0)) 

    data_samples1 = no_shift_transforms['pure'](img).unsqueeze(0).expand(num_augments, -1, -1, -1).to(device) 
    data_samples2 = torch.cat(augments2).to(device) 

    labels = []

    for b in range(0, data_samples1.shape[0], batch_size):

        with torch.no_grad():
            out1 = ssl_model(data_samples1[b : b + batch_size, :])
            out2 = ssl_model(data_samples2[b : b + batch_size, :])
            labels.append(measure(out1, out2))

    data_labels = torch.cat(labels).unsqueeze(-1).to(device)

    return data_samples1, data_samples2, data_labels

def pixel_invariance(data_samples1, data_samples2, data_labels, resize_transform, size, epochs, learning_rate, l1_weight, zero_small_values, blur_output):
    
    """
    size: resize the image to that when training the surrogate. Later we upsize
    epochs: number of epochs to train the surrogate model
    learning_rate: learning rate to train the surrogate model
    l1_weight: if not None, enables l1 regularization (sparsity)
    """
    x1 = resize_transform((size, size))(data_samples1)      # (num_samples, 3, size, size)
    x2 = resize_transform((size, size))(data_samples2)      # (num_samples, 3, size, size)

    x1 = x1.reshape(x1.size(0), -1).to(device)
    x2 = x2.reshape(x2.size(0), -1).to(device)
    
    surrogate = nn.Linear(size * size * 3, 1).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
    optimizer = torch.optim.SGD(surrogate.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        pred1, pred2 = surrogate(x1), surrogate(x2) 
        preds = (pred1 + pred2) / 2
        loss = criterion(preds, data_labels)

        if l1_weight is not None:
            loss += l1_weight * sum(p.abs().sum() for p in surrogate.parameters())

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
    
    heatmap = surrogate.weight.reshape(3, size, size)
    heatmap, _ = torch.max(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    if zero_small_values:
        heatmap[heatmap < 0.5] = 0
        
    if blur_output:
        heatmap = blur_sailency(heatmap.unsqueeze(0)).squeeze(0)
    
    return heatmap

class GradCAM(nn.Module):
    
    def __init__(self, ssl_model):
        super(GradCAM, self).__init__()
        
        self.gradients = {}
        self.features = {}
        
        self.feature_extractor = ssl_model.encoder.net
        self.contrastive_head = ssl_model.contrastive_head
        self.measure = nn.CosineSimilarity(dim=-1)
        
    def save_grads(self, img_index):
    
        def hook(grad):
            self.gradients[img_index] = grad.detach()

        return hook
    
    def save_features(self, img_index, feats):
        self.features[img_index] = feats.detach()
    
    def forward(self, img1, img2):
        
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        self.save_features('1', features1)
        self.save_features('2', features2)
        
        h1 = features1.register_hook(self.save_grads('1'))
        h2 = features2.register_hook(self.save_grads('2'))
        
        out1, out2 = features1.mean(dim=[2, 3]), features2.mean(dim=[2, 3])
        out1, out2 = self.contrastive_head(out1), self.contrastive_head(out2)
        score = self.measure(out1, out2)
        
        return score
    
def weight_activation(feats, grads):
    cam =  feats * F.relu(grads)
    cam = torch.sum(cam, dim=1).squeeze().cpu().detach().numpy()
    return cam

def get_gradcam(ssl_model, img1, img2):
    
    grad_cam = GradCAM(ssl_model).to(device)
    score = grad_cam(img1, img2)
    grad_cam.zero_grad()
    score.backward()

    cam1 = weight_activation(grad_cam.features['1'], grad_cam.gradients['1'])
    cam2 = weight_activation(grad_cam.features['2'], grad_cam.gradients['2'])
    return cam1, cam2

def get_interactioncam(ssl_model, img1, img2, reduction):
    
    grad_cam = GradCAM(ssl_model).to(device)
    score = grad_cam(img1, img2)
    grad_cam.zero_grad()
    score.backward()
    
    with torch.no_grad():
        base_head_feats, second_head_feats = ssl_model(img1), ssl_model(img2)
        
    base_head_feats = (base_head_feats - base_head_feats.min()) / (base_head_feats.max() - base_head_feats.min())
    second_head_feats = (second_head_feats - second_head_feats.min()) / (second_head_feats.max() - second_head_feats.min())
    invar_weight = 1 - ((base_head_feats - second_head_feats).abs()).mean()
    
    if reduction == 'mean':
        joint_weight = grad_cam.features['1'].mean([2,3]) * grad_cam.features['2'].mean([2,3])
    elif reduction == 'max':
        max_pooled1 = F.max_pool2d(grad_cam.features['1'], kernel_size=grad_cam.features['1'].size()[2:]).squeeze(-1).squeeze(-1)
        max_pooled2 = F.max_pool2d(grad_cam.features['2'], kernel_size=grad_cam.features['2'].size()[2:]).squeeze(-1).squeeze(-1)
        joint_weight = max_pooled1 * max_pooled2
    else:
        B, D, H, W = grad_cam.features['1'].size()
        reshaped1 = grad_cam.features['1'].permute(0,2,3,1).reshape(B, H * W, D)
        reshaped2 = grad_cam.features['2'].permute(0,2,3,1).reshape(B, H * W, D)
        features1_query, features2_query = reshaped1.mean(1).unsqueeze(1), reshaped2.mean(1).unsqueeze(1)
        attn1 = (features1_query @ reshaped1.transpose(-2, -1)).softmax(dim=-1)
        attn2 = (features2_query @ reshaped2.transpose(-2, -1)).softmax(dim=-1)
        att_reduced1 = (attn1 @ reshaped1).squeeze(1)
        att_reduced2 = (attn2 @ reshaped2).squeeze(1)
        joint_weight = att_reduced1 * att_reduced2
        
    joint_weight = joint_weight.unsqueeze(-1).unsqueeze(-1).expand_as(grad_cam.features['1'])
    
    feats1 = grad_cam.features['1'] * joint_weight * invar_weight
    feats2 = grad_cam.features['2'] * joint_weight * invar_weight

    cam1 = weight_activation(feats1, grad_cam.gradients['1'])
    cam2 = weight_activation(feats2, grad_cam.gradients['2'])
    
    return cam1, cam2

