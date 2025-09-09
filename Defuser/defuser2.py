import cv2
import pytesseract

# 加载图片
image_path = 'test.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Tesseract进行文字识别
text = pytesseract.image_to_string(gray, lang='chi_sim')  # 识别英文文字
print("识别结果：", text)

# 如果指定内容出现在识别结果中，则将对应区域涂黑
specified_text = "法"
if specified_text in text:
    # 通过调用 Tesseract 的 detect方法来获取每个字符的位置信息
    boxes = pytesseract.image_to_boxes(gray)
    for b in boxes.splitlines():
        b = b.split(' ')
        if b[0] == specified_text[0]:  # 第一个字符的位置信息
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            expanded_x = max(x - 4 * w, 0)
            expanded_w = min(w + 8 * w, img.shape[1] - expanded_x)
            cv2.rectangle(img, (expanded_x, img.shape[0] - y), (expanded_x + expanded_w, img.shape[0] - h), (0, 0, 0), -1)  # 涂黑对应区域
# 保存处理后的图片
output_image_path = 'path_to_your_output_image.jpg'
cv2.imwrite(output_image_path, img)

print("处理后的图片已保存为:", output_image_path)

