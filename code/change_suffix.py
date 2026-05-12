import os
from PIL import Image

def convert_to_jpg(input_path, output_path=None):
    """
    将图像转换为 JPG 格式
    :param input_path: 输入图像文件路径
    :param output_path: 输出图像文件路径（如果为 None，则自动生成）
    """
    # 打开图像
    img = Image.open(input_path)
    
    # 自动生成输出路径
    if output_path is None:
        # 替换后缀为 .jpg
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.jpg"
    
    # 将图像保存为 JPG 格式
    img.convert("RGB").save(output_path, "JPEG")
    print(f"图像已转换并保存为: {output_path}")

    # 删除原始图像文件
    os.remove(input_path)
    print(f"原始图像已删除: {input_path}")

# 示例使用
if __name__ == "__main__":
    input_file = r"multi.png"  # 替换为实际的输入图像路径
    convert_to_jpg(input_file)