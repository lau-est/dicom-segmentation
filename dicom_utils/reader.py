import pydicom as dcm
import numpy as np
import scipy.ndimage
from skimage import measure
from skimage import morphology
from matplotlib import pyplot as plt
import os
import warnings

def read_ct(ct_dir):
    file_list = os.listdir(ct_dir)

    if len(file_list)   < 2:
        print(ct_dir, "-> Invalid CT directory address...")
        return None
    
    elif len(file_list) > 2000:
        warnings.warn("Too many slices. It might cause memory issues...")

    
    file_list.sort()

    while file_list[0][0] == '.':
        del file_list[0]
    
    slices = [dcm.read_file(ct_dir + '/' + s, force=True) for s in file_list]
    

    return slices


def read_slice(slice_dir):
    try:
        ds = dcm.read_file(slice_dir)
        return ds
    
    except:
        print("File reading error")
        return None


def get_image(dicom_data):
    img   = dicom_data.pixel_array
    min_v = img.min()
    max_v = img.max()

    img   = (img - min_v)/(max_v - min_v + 1e-6)
    img   = img * 255

    return img.astype(np.uint8)


def get_image_hu(dicom_data):
    hu = dcm.pixel_data_handlers.util.apply_modality_lut(dicom_data.pixel_array,dicom_data)
    return hu.astype(np.int16)


def get_windowed_image(dicom_data, wtype=None):
    """
    Rescales a CT scan Slice image to a specific windowing
    inputs:
        dicom_dataset: a dicom DataSet
        wtype: a string or tuple, tuple consists (WindowCenter, WindowWidth) or string can be
            'lung', 'brain', 'bone', 'liver', 'tissues', 'mediastinum'
    output:
        a numpy 2-d array of result image
    """

    if wtype is None:
        wc = -600
        ww = 1500
    
    elif type(wtype) == str:
        tmp = get_window_values(wtype)
        wc  = tmp[0]
        ww  = tmp[1]
    
    elif type(wtype) == tuple:
        wc = wtype[0]
        ww = wtype[1]
    
    else:
        print ("Error -> Invalid wtype argument...")
    
    hf_img = dcm.pixel_data_handlers.util.apply_modality_lut(dicom_data.pixel_array,dicom_data)
    dicom_data.WindowCenter = wc
    dicom_data.WindowWidth  = ww
    res    = dcm.pixel_data_handlers.util.apply_voi_lut(hf_img, dicom_data, index=0)
    
    return res


def get_window_values(wtype="lung"):
    hf_values = {
        "lung"       : (-600, 1500),
        "mediastinum": (50, 350)   ,
        "tissues"    : (50, 400)   ,
        "liver"      : (30, 150)   ,
        "brain"      : (40, 80)    ,
        "bone"       : (400, 1800)
    }

    return hf_values[wtype]
