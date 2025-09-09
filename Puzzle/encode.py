from PIL import Image

def encrypt_message_to_image(image_path, message, output_path):
    # 打开图像
    img = Image.open(image_path)
    pixels = img.load()

    # 获取图像的宽度和高度
    width, height = img.size

    # 将消息转换为二进制字符串，并添加结束标志
    binary_message = ''.join([format(ord(c), '08b') for c in message]) + '00000000'

    # 确保消息能嵌入图片
    if len(binary_message) > width * height:
        raise ValueError("Message is too long to be embedded in the image.")

    # 嵌入消息到图像中
    index = 0
    for y in range(height):
        for x in range(width):
            if index < len(binary_message):
                r, g, b, n = pixels[x, y]
                # 修改红色通道的最后一位
                new_r = (r & 0xFE) | int(binary_message[index])
                pixels[x, y] = (new_r, g, b, n)
                index += 1

    # 保存嵌入了消息的图像
    img.save(output_path)
    print(f"Message embedded and saved to {output_path}")

# 示例使用
image_path = 'input_image.png'  # 输入图像路径
message = 'Hello, World!'  # 要嵌入的消息
output_path = 'output_image.png'  # 输出图像路径
encrypt_message_to_image(image_path, message, output_path)
