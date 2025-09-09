import pytesseract
from PIL import Image

# 指定Tesseract的安装路径，如果Tesseract已经添加到环境变量中，可以省略这一步

# 打开图片文件
image_path = 't4.png'
img = Image.open(image_path)

# 使用Tesseract进行文字识别
text = pytesseract.image_to_string(img, lang='chi_sim')  # 'eng'表示英文识别，你也可以根据需要选择其他语言

# 输出识别结果
print(text)

