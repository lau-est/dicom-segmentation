from dicom_utils import reader, display
from scipy import ndimage
from skimage import morphology
from skimage import measure
from sklearn import cluster
import numpy as np
from matplotlib import pyplot as plt

def body_mask(img, threshold = -400):
    in_img    = img.copy()
    img_shape = img.shape
    img       = ndimage.zoom(img, 256/np.asarray(img.shape), order=0)
    bodymask  = img > threshold
    bodymask  = ndimage.binary_closing(bodymask)
    bodymask  = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask  = ndimage.binary_erosion(bodymask, iterations=1)
    bodymask  = measure.label(bodymask.astype(int), connectivity=1)
    regions   = measure.regionprops(bodymask.astype(int))

    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask   = bodymask == max_region
        bodymask   = ndimage.binary_dilation(bodymask, iterations=1)

    real_scaling = np.asarray(img_shape)/256
    logical_mask = ndimage.zoom(bodymask, real_scaling, order=0)
    int_mask     = logical_mask.astype(int)
    masked_img   = (in_img + abs(in_img.min())) * int_mask

    return int_mask, masked_img 


def lung_mask(img):
    
    img_shape = img.shape
    
    mean      = np.mean(img)
    std       = np.std(img)
    img       = img - mean
    img       = img / std
    
    middle    = img[int(img_shape[1]/5):int(img_shape[1]/5*4),int(img_shape[0] /5):int(img_shape[0] /5*4)] 
    mean      = np.mean(middle)  
    max       = np.max(img)
    min       = np.min(img)
    
    img[img==max] = mean
    img[img==min] = mean

    kmeans        = cluster.KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers       = sorted(kmeans.cluster_centers_.flatten())
    threshold     = np.mean(centers)
    thresh_img    = np.where(img<threshold,1.0,0.0)  

    eroded        = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation      = morphology.dilation(eroded,np.ones([8,8]))

    labels        = measure.label(dilation) 
    label_vals    = np.unique(labels)
    regions       = measure.regionprops(labels)
    good_labels   = []

    for prop in regions:
        img_shape = img.shape
        B         = prop.bbox

        if B[2]-B[0]<img_shape[0] /10*9 and B[3]-B[1]<img_shape[1]/10*9 and B[0]>img_shape[0] /5 and B[2]<img_shape[1]/5*4:
            img_shape = img.shape
            good_labels.append(prop.label)

    mask    = np.ndarray([img_shape[0] ,img_shape[1]],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) 

    return labels,mask, mask*img