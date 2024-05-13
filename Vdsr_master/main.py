import os

import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
import Vdsr_master.vdsr as vdsr
import sys
sys.path.append("Vdsr_master/.")


# Define the colorization function
# We'll reuse the Cb and Cr channels from bicubic interpolation
def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def vdsr_main():
    # Load the pretrained model
    index = 0
    model = torch.load("Vdsr_master/model/model_epoch_50.pth")["model"]
    model = model.cuda()

    os.makedirs("temp1", exist_ok=True)

    for im_l_path in os.listdir("temp"):
        # print(im_l_path)

        im_l = Image.open("temp/" + im_l_path)
        # Load the groundtruth image and the low-resolution image (downscaled with a factor of 4)

        im_b = im_l
        # im_gt = Image.open("Set5/butterfly_GT.bmp").convert("RGB")

        # im_b = Image.open("Set5/butterfly_GT_scale_4.bmp").convert("RGB")

        # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
        # im_gt_ycbcr = np.array(im_gt.convert("YCbCr"))
        im_b_ycbcr = np.array(im_b.convert("YCbCr"))
        # im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
        im_b_y = im_b_ycbcr[:,:,0].astype(float)

        # Prepare for the input, a pytorch tensor
        im_input = im_b_y/255.
        im_input = Variable(torch.from_numpy(im_input).float()).\
            view(1, -1, im_input.shape[0], im_input.shape[1])

        im_input = im_input.cuda()

        out = model(im_input)

        out = out.cpu()
        im_h_y = out.data[0].numpy().astype(np.float32)
        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0, :, :]

        im_h = colorize(im_h_y, im_b_ycbcr)
        predix = im_l_path.split(".")[1]
        imgname = im_l_path.split(".")[0][5:]
        print(imgname)
        save_path = os.path.join("temp1", f'{imgname}(vdsr).{predix}')

        # cv2.imwrite(save_path, im_h)
        im_h.save(save_path)

        # return im_h
        # im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
        # im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
