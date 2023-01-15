import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import morphology
import nibabel as nib
import math
from scipy.ndimage import zoom
from imresize import imresize_in

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = (drange_out[0] - drange_in[0] * scale)
        data = data * scale + bias
    return data

def denorm(x):
    return adjust_dynamic_range(x, [-1,1],[0,1])

def norm(x):
    return adjust_dynamic_range(x, [0,1],[-1,1])

def adjust_scales2image(real_, cfg):
    '''Sets num_scales and scale_factor to respect respect min_size'''
    # real_: [batch_size, channels, width, height, depth] (new)
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    original_scale_factor = cfg.scale_factor
    cfg.num_scales = math.ceil(
        (math.log(
            cfg.min_size / (min(real_.shape[2], real_.shape[3], real_.shape[4])),
            original_scale_factor
        ))
    ) + 1
    cfg.scale_factor = math.pow(
        cfg.min_size/(min(real_.shape[2],real_.shape[3],real_.shape[4])),
        1/(cfg.num_scales)
    )

def imresize3D(im,scale):
    # [1,c,x,y,z]
    im = im[0]
    # [channel,w,h,d]
    im = im.transpose((1,2,3,0))
    # [w,h,d,channel]
    im = imresize_in(im, scale_factor=scale)
    # [batch,w,h,d,channels]
    im = im[None]
    # [batch,channels,w,h,d]
    im = im.transpose((0,4,1,2,3))
    return im

def load_data_fromNifti(path: str, cfg):
    """load multi-scale 3D shape data from h5 file

    Args:
        path (str): file path
        smooth (bool, optional): use gaussian blur. Defaults to True.
        only_finest (bool, optional): load only the finest(highest scale) shape. Defaults to False.

    Returns:
        np.ndarray or list[np.ndarray]: 3D shape(s)
    """
    shape_list = []
    real = nib.load(path).get_fdata()[None] # [1,x,y,z,c]
    real[:,:,:,:,0] /= 0.172 # place ct in [0 - 1 range]
    real = real.transpose((0,4,1,2,3)) # [1,c,x,y,z]
    adjust_scales2image(real, cfg)
    for i in range(0,cfg.num_scales+1):
        scale = math.pow(cfg.scale_factor,cfg.num_scales-i)
        curr_real = imresize3D(real, scale)
        shape_list.append(norm(curr_real))
    return shape_list