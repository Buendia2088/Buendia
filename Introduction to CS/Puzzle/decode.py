from PIL import Image

def decode(path):
    img = Image.open(path)
    pixels = img.load()
    width, height = img.size # 读取图像宽、高
    res = ''
    for y in range(height): 
        for x in range(width):
            r = pixels[x, y][0] # 读取当前像素r通道的值
            res += str(r & 1) # 取出r通道最后一位，并连接到结果字符串中

    message = ''
    for i in range(0, len(res), 8): # 间隔8位读取，满足ASCII码编码长度要求
        byte = res[i:i+8]
        if byte == '00000000': # 读取到终止符，结束遍历
            break
        message += chr(int(byte, 2)) # 将编码转为字符

    return message

path = 'test_res.png'
message = decode(path)
print("Hidden message:", message)
