from PIL import Image

def encode(input_path, message, output_path):
    img = Image.open(input_path)
    pixels = img.load()
    width, height = img.size
    message = ''.join([format(ord(c), '08b') for c in message]) + '00000000' # 遍历被加密信息，将其通过Unicode编码为8位二进制数

    if len(message) > width * height: # 如果信息过长，抛出错误
        raise ValueError("Message is too long to be embedded in the image.")

    index = 0
    for y in range(height):
        for x in range(width):
            if index < len(message):
                r, g, b = pixels[x, y]
                new_r = (r & 0xFE) | int(message[index]) # 先将原通道值最低位清零，再将其替换为message对应位
                pixels[x, y] = (new_r, g, b)
                index += 1

    img.save(output_path)
    print(f"Message embedded and saved to {output_path}")

input_path = 'test.png'
message = 'Love Lulu'
output_path = 'test_res.png'
encode(input_path, message, output_path)
