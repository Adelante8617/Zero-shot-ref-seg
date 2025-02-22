import base64

def image_to_base64(image_path):
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        # 将图片内容编码为base64
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string