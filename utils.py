import numpy as np
from PIL import Image
import numpy as np
from os import stat
import WaveletImageCoder

# Compute video PSNR
def psnr(ref, meas, maxVal=255):
    assert np.shape(ref) == np.shape(meas), "Reference image must match measured image dimensions"

    dif = (ref.astype(float)-meas.astype(float)).ravel()
    mse = np.linalg.norm(dif)**2/np.prod(np.shape(ref))
    psnr = 10*np.log10(maxVal**2.0/mse)
    return psnr

def bpp(filename):
    size = stat(filename).st_size
    with open(filename, 'rb') as fh:
        soi = fh.read(2)
        if soi != WaveletImageCoder.SOI_MARKER:
            raise Exception("Start of Image marker not found!")
        
        M = int.from_bytes(fh.read(2), "big")
        N = int.from_bytes(fh.read(2), "big")

        return size * 8 / (M * N)

def comp_ratio(reference, measured):
    return stat(reference).st_size / stat(measured).st_size


def resize(img, M, N):
    return np.array(Image.fromarray(img).resize((N, M), resample=Image.BILINEAR))

CONV_MAT = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.4188688, -0.081312]]).T
INV_CONV_MAT = np.linalg.inv(CONV_MAT)

def RGB2YCbCr(im_rgb):
    """
    Input:  a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]

    Output: a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
    """
    
    im_ycbcr = np.array([-128, 0, 0]) + im_rgb @ CONV_MAT
    im_ycbcr = np.where(im_ycbcr > 127, 127, im_ycbcr)
    im_ycbcr = np.where(im_ycbcr < -128, -128, im_ycbcr)
    
    return im_ycbcr

def YCbCr2RGB(im_ycbcr):
    """
    Input:  a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]

    Output: a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
    """
    
    im_rgb = (np.array([128, 0, 0]) + im_ycbcr) @ INV_CONV_MAT
    im_rgb = np.where(im_rgb > 255, 255, im_rgb)
    im_rgb = np.where(im_rgb < 0, 0, im_rgb)
    return im_rgb