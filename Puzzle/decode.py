from PIL import Image

def decrypt_message_from_image(image_path):
    # 打开图像
    img = Image.open(image_path)
    pixels = img.load()

    # 获取图像的宽度和高度
    width, height = img.size

    # 提取图像中嵌入的二进制字符串
    binary_message = ''
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            binary_message += str(r & 1)

    # 将二进制字符串转换为消息
    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        if byte == '00000000':
            break  # 遇到结束标志，停止解码
        message += chr(int(byte, 2))

    return message

# 示例使用
image_path = 'output_image.png'  # 嵌入了消息的图像路径
message = decrypt_message_from_image(image_path)
print("Decrypted message:", message)
