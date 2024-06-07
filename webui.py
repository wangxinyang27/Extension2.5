import gradio as gr
from utils import (nearest_interpolation,
                   bilinear_interpolation,
                   bicubic_interpolation,
                   lanczos_interpolation,
                   swinir_extension,
                   real_esrgan,
                   real_esrgan_anime_6B,
                   esrgan,
                   vdsr,
                   edsr,
                   )

def fn_image(x):
    # 根据放大模型执行图像增强
    if mode == "Nearest":
        return nearest_interpolation(x)
    elif mode == "Bilinear":
        return bilinear_interpolation(x)
    elif mode == "Bicubic":
        return bicubic_interpolation(x)
    elif mode == "Lanczos":
        return lanczos_interpolation(x)
    elif mode == "SwinIR_x4":
        return swinir_extension(x)
    elif mode == "Real_ESRGAN_x4":
        return real_esrgan(x)
    elif mode == "Real_ESRGAN_x4_Anime_6B":
        return real_esrgan_anime_6B(x)
    elif mode == "ESRGAN":
        return esrgan(x)
    elif mode == "VDSR":
        return vdsr(x)
    # elif mode == "EDSR":
    #     return edsr(x)
    else:
        pass

mode = "Nearest"
def fn_mode(choice):
    # 放大模式的选择
    global mode
    mode = choice
    # print(mode)

def fn_upload_width(image):
    # 上传图像时更新滑块表示图像的宽
    return image.size[0]

def fn_upload_height(image):
    # 上传图像时更新滑块表示图像的高
    return image.size[1]

def fn_clear_width():
    # 清空图像时把滑块表示的宽的值置零
    return 0

def fn_clear_height():
    # 清空图像时把滑块表示的高的值置零
    return 0

with gr.Blocks() as demo:
    # 输入输出图像框
    with gr.Row():
        image_input = gr.Image(label="input", type='pil', sources=["upload"])
        image_output = gr.Image(label="output", format="png")
    # 滑块，用来表示图像的宽和高
    with gr.Column():
        image_width = gr.Slider(
            min_width=0, maximum=8192, label="width", step=1, interactive=False
        )
        image_height = gr.Slider(
            min_width=0, maximum=8192, label="height", step=1, interactive=False
        )

    # 上传图像时更新滑块的位置，表示图像的宽和高
    image_input.upload(fn_upload_width, inputs=image_input, outputs=image_width)
    image_input.upload(fn_upload_height, inputs=image_input, outputs=image_height)
    # 清空图像时把滑块的位置都置零
    image_input.clear(fn_clear_width, outputs=image_width)
    image_input.clear(fn_clear_height, outputs=image_height)
    # 选择放大模式的单选框
    radio = gr.Radio(
        ["Nearest",
         "Bilinear",
         "Bicubic",
         "Lanczos",
         "SwinIR_x4",
         "Real_ESRGAN_x4",
         "Real_ESRGAN_x4_Anime_6B",
         "ESRGAN",
         "VDSR",
         ],
        label="Extension Mode", value="Nearest"
    )
    # 单选框的值改变时，随之更改放大模式
    radio.change(fn_mode, inputs=radio)
    # 执行按键
    image_button = gr.Button("submit", size="lg")
    # 执行图像放大
    image_button.click(fn=fn_image, inputs=image_input, outputs=image_output)

demo.launch()
