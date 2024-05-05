import os
from PIL import Image
import shutil
import PIL.Image as pil_image

from SwinIR_master.main_test_swinir import swinir_main
from Real_ESRGAN_master.inference_realesrgan import real_esrgan_main
from Real_ESRGAN_master.inference_realesrgan_anime_6B import real_esrgan_anime_6B_main
from ESRGAN_master.test import esrgan_main

def crop_images(img, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = "image.png"
    # 创建输出目录
    # 为了保证不爆gpu，裁剪到最大512*512的小块若干个
    idx = 1  # 衡量划分的块数
    while True:
        if img.height / idx <= 512:
            break
        idx += 1
    height_count = idx
    idx = 1

    while True:
        if img.width / idx <= 512:
            break
        idx += 1
    width_count = idx

    batch_height = img.height // height_count
    batch_width = img.width // width_count

    if img.height / height_count > batch_height:
        batch_height += 1
    if img.width / width_count > batch_width:
        batch_width += 1

    cropped_img = []

    for i in range(0, width_count):
        for j in range(0, height_count):
            print(i*batch_width,
                  j*batch_height,
                  min((i+1) * batch_width, img.width),
                  min((j+1) * batch_height, img.height),
                  )
            cropped_img.append(img.crop((i*batch_width,
                                         j*batch_height,
                                         min((i+1)*batch_width, img.width),
                                         min((j+1)*batch_height, img.height))
                                        )
                               )
    # 保存裁剪后的图像
    prefix, suffix = filename.split(".")
    print("output", output_dir)
    print(os.getcwd())
    for i in range(len(cropped_img)):
        output_path = os.path.join(output_dir, prefix + "{}.".format(i+1001) + suffix)
        print(output_path)
        cropped_img[i].save(output_path)
        print(f"Image {filename} cropped and saved to {output_path}")

    return (img.height, img.width, height_count, width_count), filename

def stitch_images(input_dir, batch_size):
    # 创建输出目录
    h, w, hc, wc = batch_size

    # 存储所有图像对象
    images = []

    # 遍历输入目录中的所有PNG图片
    total_height, total_width = 0, 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # 打开图像并添加到列表中
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            images.append(img)

    total_width, total_height = w * 4, h * 4

    # 创建一个新的空白图像
    stitched_img = Image.new("RGBA", (total_width, total_height))

    # 拼接图像
    y_offset = 0
    x_offset = 0
    print("总大小({},{})".format(total_height, total_width))

    for img in images:
        stitched_img.paste(img, (x_offset, y_offset))
        print("当前左上角:{},{},  子块大小:{},{}".format(x_offset, y_offset, img.height, img.width))
        y_offset += img.height
        if y_offset >= total_height:
            y_offset = 0
            x_offset += img.width

    return stitched_img

    # 保存拼接后的图像
    # prefix, suffix = name.split(".")
    # output_path = os.path.join(output_dir, "{}({}).png".format(prefix, model))
    # stitched_img.save(output_path)

    # print(f"Stitched image saved to {output_path}")

def nearest_interpolation(image):
    # 最近邻插值
    w, h = image.size
    rw, rh = w * 4, h * 4
    return image.resize((rw, rh), pil_image.NEAREST)

def bilinear_interpolation(image):
    # 双线性插值
    w, h = image.size
    rw, rh = w * 4, h * 4
    return image.resize((rw, rh), pil_image.BILINEAR)

def bicubic_interpolation(image):
    # 双三次插值
    w, h = image.size
    rw, rh = w * 4, h * 4
    return image.resize((rw, rh), pil_image.BICUBIC)

def lanczos_interpolation(image):
    # lanczos插值
    w, h = image.size
    rw, rh = w * 4, h * 4
    return image.resize((rw, rh), pil_image.LANCZOS)

def swinir_extension(image):
    batch_size, _ = crop_images(image, "temp")
    swinir_main()
    out = stitch_images("temp1", batch_size)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def real_esrgan(image):
    batch_size, _ = crop_images(image, "temp")
    real_esrgan_main()
    out = stitch_images("temp1", batch_size)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def real_esrgan_anime_6B(image):
    batch_size, _ = crop_images(image, "temp")
    real_esrgan_anime_6B_main()
    out = stitch_images("temp1", batch_size)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def esrgan(image):
    batch_size, _ = crop_images(image, "temp")
    esrgan_main()
    out = stitch_images("temp1", batch_size)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out