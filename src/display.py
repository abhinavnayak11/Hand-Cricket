from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import math

import cv2
import torch



def display_images(dataloader, title, batch_size = None, 
                    denorm = False, 
                    denorm_mean = (0.485, 0.456, 0.406),   
                    denorm_std = (0.229, 0.224, 0.225)):
    
    if batch_size is None:
        batch_size = math.ceil(len(dataloader.dataset)/len(dataloader))
        
    samples, labels = iter(dataloader).next()
    new = []    # to store new tensors after editing numpy array to add the target in the image
    for img, y in zip(samples, labels):

        
        if denorm == False:
            # convert tensor to numpy, shape to (H,W,C), dtype to (uint8/uint16/int16/int32)
            # copy() will make it a contigous array 
            img = img.numpy().transpose(1,2,0).astype(np.int16).copy()
        
        if denorm == True:
            # convert tensor to numpy, shape to (H,W,C)
            img = img.numpy().transpose(1,2,0)

            # denormalize images
            img[:,:,0] = (img[:,:,0] * 255 * denorm_std[0]) + (255 * denorm_mean[0])
            img[:,:,1] = (img[:,:,1] * 255 * denorm_std[1]) + (255 * denorm_mean[1])
            img[:,:,2] = (img[:,:,2] * 255 * denorm_std[2]) + (255 * denorm_mean[2])

            # convert dtype to  (uint8/uint16/int16/int32)
            # copy() will make it a contigous array 
            img = img.astype(np.int16).copy()
        

        # put target value on the image
        cv2.putText(img, f"{y.int()}", (10,20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0),2)

        # convert to tensor (c,h,w)
        img = torch.tensor(img.transpose(2,0,1).astype(np.float32))

        # create list of tensors
        new.append(img)

    # convert list of 3d tensors to 4d tensor
    samples = torch.stack(new)
    plt.figure(figsize=(18,18))
    grid_imgs = make_grid(samples[:batch_size], nrow = 6)
    np_grid_imgs = grid_imgs.numpy()

    # pytorch requires torch tensor of shape (c,h,w) & dtype float
    # plt requires numpy array of shape (h,w,c) & dtype int
    plt.imshow(np.transpose(np_grid_imgs, (1,2,0)).astype(int))
    plt.title(title)
    plt.show()


def sigmoid(x):
    ex =  np.exp(x)
    return (ex/sum(ex))*100

def display_inference_result(samples, labels, outputs, denorm = False, 
                             denorm_mean = (0.485, 0.456, 0.406), 
                             denorm_std = (0.229, 0.224, 0.225)):
    
        
    new = []    # to store new tensors after editing numpy array to add the target in the image

    for img, y, output in zip(samples, labels, outputs):

        probabilities = sigmoid(output.tolist())
    
        if denorm == False:
            # convert tensor to numpy, shape to (H,W,C), dtype to (uint8/uint16/int16/int32)
            # copy() will make it a contigous array 
            img = img.numpy().transpose(1,2,0).astype(np.int16).copy()
        
        if denorm == True:
            # convert tensor to numpy, shape to (H,W,C)
            img = img.numpy().transpose(1,2,0)

            # denormalize images
            img[:,:,0] = (img[:,:,0] * 255 * denorm_std[0]) + (255 * denorm_mean[0])
            img[:,:,1] = (img[:,:,1] * 255 * denorm_std[1]) + (255 * denorm_mean[1])
            img[:,:,2] = (img[:,:,2] * 255 * denorm_std[2]) + (255 * denorm_mean[2])

            # convert dtype to  (uint8/uint16/int16/int32)
            # copy() will make it a contigous array 
            img = img.astype(np.int16).copy()
        

        # put target value on the image
        cv2.putText(img, f"{y.int()}", (10,20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0),2)
        
        h = 30
        for i, p in enumerate(probabilities):
            if p<0.1:
                p = 0
            else:
                p = round(p,2)
            cv2.putText(img, f"{i}:{p}", (10,h), 1, 0.7, (255,255,0),1)
            h+=10

        # convert to tensor (c,h,w)
        img = torch.tensor(img.transpose(2,0,1).astype(np.float32))

        # create list of tensors
        new.append(img)

    # convert list of 3d tensors to 4d tensor
    samples = torch.stack(new)
    plt.figure(figsize=(18,18))
    grid_imgs = make_grid(samples, nrow = 6)
    np_grid_imgs = grid_imgs.numpy()

    # pytorch requires torch tensor of shape (c,h,w) & dtype float
    # plt requires numpy array of shape (h,w,c) & dtype int
    plt.imshow(np.transpose(np_grid_imgs, (1,2,0)).astype(int))
    plt.title('Inference results')
    plt.show()




