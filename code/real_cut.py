from ultralytics import YOLO
import cv2
import os

def real_cut(image_file,model_path,crop_dir_name):
    # 加载模型，根据实际使用的模型进行替换
    model = YOLO(model_path)

    # image_file = r"multi.jpg"
    # if not os.path.exists(image_file):
    #     print(f"文件不存在: {image_file}")

    # 读取图像
    im0 = cv2.imread(image_file)
    if im0 is None:
        print(f"无法读取图像文件: {image_file}")

    # 使用模型进行预测
    results = model(im0)

    # 裁剪并保存结果
    border_size = 5  # 设置边框宽度
    border_color = [179, 186, 185]  # 灰色 (BGR 格式)

    for i, result in enumerate(results):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = im0[y1:y2, x1:x2]

            # 添加灰色边框到裁剪图像
            cropped_img_with_border = cv2.copyMakeBorder(
                cropped_img,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=border_color
            )

            # 保存带边框的裁剪图像
            crop_file_name = os.path.join(crop_dir_name, f"{i}.jpg")
            cv2.imwrite(crop_file_name, cropped_img_with_border)
            print(f"保存裁剪图像: {crop_file_name}")
            i += 1
            
    return i
