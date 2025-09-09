from PIL import Image

def solve(path):
    img = Image.open(path)
    pixels = img.load()
    width = img.size[0] # 由于密文很短，只有第一行有信息，所以不需要读取图像的高度
    res = ''
    
    for x in range(width):
        r = pixels[x, 0][0] # 读取当前像素r通道的值
        res += str(r & 1) # 取出r通道最后一位，并连接到结果字符串中

    message = ''
    for i in range(0, len(res), 8): # 间隔8位读取，满足ASCII码编码长度要求
        byte = res[i:i+8]
        if len(byte) == 8:
            message += chr(int(byte, 2)) # 将编码转为字符

    return message

path = '/home/chengyuan/Buendia/Introduction to CS/Puzzle/b.png'
message = solve(path)
print("Hidden message:", message)
