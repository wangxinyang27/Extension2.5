import os
from PIL import Image
import shutil
import PIL.Image as pil_image

from SwinIR_master.main_test_swinir import swinir_main
from Real_ESRGAN_master.inference_realesrgan import real_esrgan_main
from Real_ESRGAN_master.inference_realesrgan_anime_6B import real_esrgan_anime_6B_main
from ESRGAN_master.test import esrgan_main
from Vdsr_master.main import vdsr_main
from EDSR_master.src.main import edsr_main


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

    # 如果有不完整的子块，要加上一
    if img.height / height_count > batch_height:
        batch_height += 1
    if img.width / width_count > batch_width:
        batch_width += 1

    # print(batch_width,batch_height,width_count, height_count)

    cropped_img = []
    cropped_value = []  # 保存图像的左上角位置和高宽
    cropped_key = []

    i_d, j_d = 0, 0
    # i,j表示左上角的坐标
    while i_d < img.width:
        while j_d < img.height:
            label = ""
            cropped_img.append(img.crop((i_d, j_d,
                                         min(i_d + batch_width, img.width),
                                         min(j_d + batch_height, img.height))
                                        )
                               )
            message = [i_d, j_d, min(i_d + batch_width, img.width), min(j_d + batch_height, img.height)]

            label += "t" if message[1] == 0 else ""  # 子块在顶部
            label += "b" if message[3] == img.height else ""  # 子块在底部
            label += "l" if message[0] == 0 else ""  # 子块在左侧
            label += "r" if message[2] == img.width else ""  # 子块在右侧

            message.append(label)
            print(message)
            cropped_value.append(message)

            if j_d + batch_height >= img.height:
                j_d = 0
                break
            j_d += batch_height // 2
        else:
            j_d = 0
        if i_d + batch_width >= img.width:
            break

        i_d += batch_width // 2

    # for i in range(0, width_count):
    #     for j in range(0, height_count):
    #         if (i*batch_width+batch_width//2 <= img.width and
    #                 j*batch_height+batch_height//2 <= img.height):
    #             cropped_img.append(img.crop((i*batch_width,
    #                                          j*batch_height,
    #                                          min((i+1)*batch_width, img.width),
    #                                          min((j+1)*batch_height, img.height))
    #                                         )
    #                                )  # 左闭右开
    #             o_message = [  # 四个角
    #                     i*batch_width,
    #                     j*batch_height,
    #                     min((i+1)*batch_width, img.width),
    #                     min((j+1)*batch_height, img.height),
    #                 ]
    #
    #             label = ""
    #             if o_message[1] == 0:
    #                 label += "t"  # 子块在顶部
    #             if o_message[3] == img.height:
    #                 label += "b"  # 子块在底部
    #
    #             if o_message[0] == 0:
    #                 label += "l"  # 子块在左侧
    #             if o_message[2] == img.width:
    #                 label += "r"  # 子块在右侧
    #
    #             o_message.append(label)
    #             print(o_message)
    #             cropped_value.append(o_message)
    #
    #         if (i+1) * batch_width < img.width:
    #             cropped_img.append(img.crop((i*batch_width+batch_width//2,  # 宽短补
    #                                          j*batch_height,
    #                                          min((i+1)*batch_width+batch_width//2, img.width),
    #                                          min((j+1)*batch_height, img.height))
    #                                         )
    #                                )
    #             w_message = [
    #                     i*batch_width+batch_width//2,
    #                     j*batch_height,
    #                     min((i+1)*batch_width+batch_width//2, img.width),
    #                     min((j+1)*batch_height, img.height),
    #                 ]
    #
    #             label = ""
    #             if w_message[1] == 0:
    #                 label += "t"  # 子块在顶部
    #             if w_message[3] == img.height:
    #                 label += "b"  # 子块在底部
    #
    #             if w_message[0] == 0:
    #                 label += "l"  # 子块在左侧
    #             if w_message[2] == img.width:
    #                 label += "r"  # 子块在右侧
    #
    #             w_message.append(label)
    #             print(w_message)
    #             cropped_value.append(w_message)
    #
    #         if (j + 1) * batch_height < img.height:
    #             cropped_img.append(img.crop((i * batch_width,
    #                                          j * batch_height + batch_height // 2,  # 高短补
    #                                          min((i + 1) * batch_width, img.width),
    #                                          min((j + 1) * batch_height + batch_height // 2, img.height))
    #                                         )
    #                                )
    #             h_message = [
    #                     i * batch_width,
    #                     j * batch_height + batch_height // 2,
    #                     min((i + 1) * batch_width, img.width),
    #                     min((j + 1) * batch_height + batch_height // 2, img.height),
    #                 ]
    #
    #             label = ""
    #             if h_message[1] == 0:
    #                 label += "t"  # 子块在顶部
    #             if h_message[3] == img.height:
    #                 label += "b"  # 子块在底部
    #
    #             if h_message[0] == 0:
    #                 label += "l"  # 子块在左侧
    #             if h_message[2] == img.width:
    #                 label += "r"  # 子块在右侧
    #
    #             h_message.append(label)
    #             print(h_message)
    #             cropped_value.append(h_message)

    # 保存裁剪后的图像
    prefix, suffix = filename.split(".")
    # print("output", output_dir)
    # print(os.getcwd())
    for i in range(len(cropped_img)):
        output_path = os.path.join(output_dir, prefix + "{}.".format(i+1000001) + suffix)
        # print(output_path)
        cropped_key.append(str(i+1000001))
        cropped_img[i].save(output_path)
        # print(f"Image {filename} cropped and saved to {output_path}")

    cropped_info = dict(zip(cropped_key, cropped_value))
    # print(cropped_info)

    return (img.height, img.width, height_count, width_count), cropped_info


def stitch_images(input_dir, batch_size, cropped_info=None, enlarged=False):
    # 创建输出目录
    # input()
    h, w, hc, wc = batch_size

    # 存储所有图像对象
    images = {}

    # 遍历输入目录中的所有PNG图片
    total_height, total_width = 0, 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # 打开图像并添加到列表中
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            fn = filename.split("(")[0].replace("image", "")
            images[str(fn)] = img

    print(images)

    if enlarged:
        total_width, total_height = w, h
    else:
        total_width, total_height = w * 4, h * 4

    # 创建一个新的空白图像
    stitched_img = Image.new("RGBA", (total_width, total_height))

    # 拼接图像
    y_offset = 0
    x_offset = 0
    print("总大小({},{})".format(total_height, total_width))

    # for img in images:
    #     stitched_img.paste(img, (x_offset, y_offset))
    #     # print("当前左上角:{},{},  子块大小:{},{}".format(x_offset, y_offset, img.height, img.width))
    #     y_offset += img.height
    #     print(x_offset, y_offset)
    #     if y_offset >= total_height:
    #         y_offset = 0
    #         x_offset += img.width

    for lab, direction in cropped_info.items():  # key, value
        lt_w, lt_h, rb_w, rb_h, label = direction  # 子块在原图像中的左上角和右下角(向右，向下....)
        # 子块的左上角在放大后的图像中应该的位置
        lt_w_ = (lt_w + 1) * 4 - 1 if not enlarged else lt_w
        lt_h_ = (lt_h + 1) * 4 - 1 if not enlarged else lt_h

        # 放大后子块的宽和高
        w = (rb_w - lt_w) * 4 if not enlarged else rb_w - lt_w
        h = (rb_h - lt_h) * 4 if not enlarged else rb_h - lt_h

        # 为了解决边缘不连续的问题，我们对子块进行裁剪
        # 处在边缘部分的不需要裁剪，这里默认四个边都不是边缘，各裁剪1/6
        # 如果给定的标识指出子块某个方向是边缘，就恢复
        left = 0 if "l" in label else int(w // 6)
        top = 0 if "t" in label else int(h // 6)
        right = w if "r" in label else int(w // 6 * 5)
        bottom = h if "b" in label else int(h // 6 * 5)

        # 裁剪后子块在放大后图像的相对位置不变，因此引入下面的变量来弥补裁剪后的偏移
        # 如果图像左边裁剪了六分之一，那么左上角位置应该往左加上这个六分之一的量
        left_offset = left if "l" not in label else 0
        top_offset = top if "t" not in label else 0

        # 裁剪时不依照子块在原图像或放大后图像的位置
        image = images[lab].crop((left, top, right, bottom))
        stitched_img.paste(image, (lt_w_ + left_offset, lt_h_ + top_offset))

        # if label == "o":
        #     image = images[lab]
        #     stitched_img.paste(image, (w_, h_))
        # elif label == "h":
        #     w, h = images[lab].size
        #     image = images[lab].crop((0, int(h * 0.25), w, int(h * 0.75)))
        #     stitched_img.paste(image, (w_, h_ + int(h * 0.25)))
        # elif label == "w":
        #     w, h = images[lab].size
        #     image = images[lab].crop((int(w * 0.25), 0, int(w * 0.75), h))
        #     stitched_img.paste(image, (w_ + int(w * 0.25), h_))

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
    batch_size, cropped_info = crop_images(image, "temp")
    swinir_main()
    out = stitch_images("temp1", batch_size, cropped_info)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def real_esrgan(image):
    batch_size, cropped_info = crop_images(image, "temp")
    real_esrgan_main()
    out = stitch_images("temp1", batch_size, cropped_info)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def real_esrgan_anime_6B(image):
    batch_size, cropped_info = crop_images(image, "temp")
    real_esrgan_anime_6B_main()
    out = stitch_images("temp1", batch_size, cropped_info)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def esrgan(image):
    batch_size, cropped_info = crop_images(image, "temp")
    esrgan_main()
    out = stitch_images("temp1", batch_size, cropped_info)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def vdsr(image):
    # 因为vdsr的very deep，所以还是会爆显存
    w, h = image.size
    image = image.resize((w * 4, h * 4), Image.BICUBIC)   # 先放大后裁剪
    batch_size, cropped_info = crop_images(image, "temp")
    # print(batch_size)
    vdsr_main()
    out = stitch_images("temp1", batch_size, cropped_info, enlarged=True)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out

def edsr(image):
    w, h = image.size
    image = image.resize((w * 4, h * 4), Image.BICUBIC)  # 先放大后裁剪
    batch_size, cropped_info = crop_images(image, "temp")
    # print(batch_size)
    edsr_main()
    out = stitch_images("temp1", batch_size, cropped_info, enlarged=True)
    shutil.rmtree('temp')
    shutil.rmtree('temp1')
    return out