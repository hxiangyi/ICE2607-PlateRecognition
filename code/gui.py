import sys
import os
import cv2
import shutil
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QGroupBox, QTextEdit, QStatusBar, QLayoutItem
)
from PyQt5.QtGui import QPixmap, QFont, QFontMetrics, QPalette, QBrush
from PyQt5.QtCore import Qt
import detect_aug
import change_suffix
import real_cut

# 将资源文件打包到可执行文件中
def get_resource_path(relative_path):
    try:
        # PyInstaller 创建一个临时文件夹并将资源存储在_MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# def get_resource_path(relative_path):
#     return relative_path

class LicensePlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    # 初始化界面
    def init_ui(self):
        self.setWindowTitle("智能车牌识别系统")
        self.setGeometry(100, 50, 1200, 1000)  # 更大的窗口尺寸
        
        # 设置背景图片
        self.set_background_image()
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 10px;  /* 保证按钮的内边距相同 */
                font-size: 25px;
                font-weight: bold;
                border-radius: 8px;
                width: 60px;  /* 设置固定的宽度 */
                height: 60px;  /* 设置固定的高度，确保按钮方形 */
                margin: 10px;  /* 设置按钮之间的间距 */
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 8px;
                font-size: 25px;
                font-weight: bold;
            }
            QLabel {
                font-size: 100px;
            }
        """)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 顶部标题
        title_label = QLabel("智能车牌识别系统")
        title_label.setAlignment(Qt.AlignCenter)
        # 设置标题样式
        title_label.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #003366;  /* 设置字体颜色为深蓝色 */
            border:1px solid #003366;  /* 为标题添加深蓝色边框 */
            padding: 4px;  /* 给标题添加内边距，确保文字和边框之间有空间 */
            border-radius: 4px;  /* 为边框添加圆角效果 */
        """)
        main_layout.addWidget(title_label)
        
        # 图片区域
        image_layout = QHBoxLayout()
        self.original_image = QLabel("原始图像")
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setStyleSheet("border: 1px solid #ccc; background: #e0e0e0; font-size: 30px;")
        self.original_image.setFixedSize(500, 400)  # 扩大显示区域
        
        self.result_image = QLabel("处理结果")
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setStyleSheet("border: 1px solid #ccc; background: #e0e0e0; font-size: 30px;")
        self.result_image.setFixedSize(500, 400)  # 扩大显示区域
        
        image_layout.addWidget(self.original_image)
        image_layout.addWidget(self.result_image)
        main_layout.addLayout(image_layout)
        
        # 结果输出区域
        result_box = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_box.setLayout(result_layout)
        result_box.setStyleSheet("""
            QGroupBox {
                font-size: 22px;
            }
            QGroupBox::title {
                color: white;  /* 设置标题文字颜色为白色 */
            }
        """)
        main_layout.addWidget(result_box)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("上传图片")
        self.process_button = QPushButton("开始处理")
        self.clear_button = QPushButton("清空内容")
        
        self.upload_button.clicked.connect(self.upload_image)
        self.process_button.clicked.connect(self.process_image)
        self.clear_button.clicked.connect(self.clear_content)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.clear_button)
        main_layout.addLayout(button_layout)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("准备就绪")
        self.status_bar.setStyleSheet("font-size: 16px;")
        main_layout.addWidget(self.status_bar)
        
        self.setLayout(main_layout)

    # 设置背景图像并确保其按窗口大小自适应
    def set_background_image(self):
        palette = self.palette()
        pixmap = QPixmap(get_resource_path("background.jpg"))
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding)
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
    
    # 窗口大小改变时调整字体大小
    def resizeEvent(self, event):
        self.adjust_text_font()
        self.set_background_image()
        super().resizeEvent(event)

    # 根据文本框大小调整字体大小
    def adjust_text_font(self):
        text_edit_width = self.result_text.width()
        text_edit_height = self.result_text.height()

        # 计算字体大小，确保字体适应文本框
        font = self.result_text.font()
        font_metrics = QFontMetrics(font)
        
        # 使用文本框宽高计算字体大小
        max_font_size = min(text_edit_width // font_metrics.averageCharWidth(), text_edit_height // font_metrics.height())
        
        # 设置字体大小
        font.setPointSize(max_font_size)
        self.result_text.setFont(font)

    # 上传图片
    def upload_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                # 检查文件格式
                if not file_path.lower().endswith('.jpg'):
                    # 如果不是 JPG 格式，使用 change_suffix 转换
                    new_file_path = change_suffix.convert_to_jpg(file_path)
                    if new_file_path:
                        file_path = new_file_path
                    else:
                        self.result_text.append("转换格式失败！")
                        self.status_bar.showMessage("转换格式失败")
                        return
                pixmap = QPixmap(file_path)
                self.original_image.setPixmap(pixmap.scaled(500, 400, Qt.KeepAspectRatio))
                self.result_text.append(f"图片已加载：{file_path}")
                self.image_path = file_path
                self.status_bar.showMessage("图片加载成功")
            else:
                self.status_bar.showMessage("未选择图片")
        except Exception as e:
            self.result_text.append(f"文件上传出现错误: {str(e)}")
            self.status_bar.showMessage("文件上传失败")
    
    # 清空布局
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()

    # 处理图片
    def process_image(self):
        try:
            if hasattr(self, 'image_path'):
                self.result_image.setPixmap(QPixmap())
                if self.result_image.layout():
                    # 清除所有子部件
                    while self.result_image.layout().count():
                        item = self.result_image.layout().takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    # 删除布局
                    QWidget().setLayout(self.result_image.layout())
                self.result_image.setText("处理结果")  # 恢复默认文本
                # 使用 real_cut 识别并裁剪车牌
                folder_name = "cut_temp"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                n = real_cut.real_cut(self.image_path, get_resource_path("best.pt"), "cut_temp")
                cropped_images = [f"cut_temp/{i}.jpg" for i in range(n)]
                if cropped_images:
                    # 清空 result_image 和 result_text
                    self.clear_layout(self.result_image.layout())
                    self.result_text.clear()
                    
                    # 创建一个新的 QWidget 作为容器
                    container_widget = QWidget()
                    layout = QVBoxLayout(container_widget)
                    
                    for i, cropped_img in enumerate(cropped_images):
                        # 创建一个新的 QLabel 来显示裁剪后的车牌图片
                        label = QLabel()
                        pixmap = QPixmap(cropped_img)
                        if pixmap.isNull():
                            self.result_text.append(f"无法加载图片：{cropped_img}")
                            continue
                        label.setPixmap(pixmap.scaled(500, 400, Qt.KeepAspectRatio))
                        layout.addWidget(label)
                        
                        # 使用 detect_license_plate 识别车牌类型和字符
                        plate_type, recognized_chars = detect_aug.detect_license_plate(cropped_img, get_resource_path("cnn_epoch_35.pth"))
                        self.result_text.append(f"处理完成！\n车牌号：{recognized_chars}\n车牌类型：{plate_type}")
                        self.status_bar.showMessage("处理完成")
                    
                    # 将容器 widget 设置到 result_image
                    self.result_image.setLayout(layout)
                else:
                    self.result_text.append("未检测到车牌！")
                    self.status_bar.showMessage("处理失败：未检测到车牌")
            else:
                self.result_text.append("请先上传图片！")
                self.status_bar.showMessage("处理失败：未上传图片")
        except Exception as e:
            self.result_text.append(f"图像处理出现错误: {str(e)}")
            self.status_bar.showMessage("处理失败")
        finally:
            # 删除临时文件夹
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)

    # 清空内容
    def clear_content(self):
        try:
            # if self.original_image.pixmap():
            #     self.original_image.setPixmap(QPixmap())  # 清空图片而不是清除文本

            # # 清空 result_image 的布局
            # layout = self.result_image.layout()
            # if layout is not None:
            #     while layout.count():
            #         child = layout.takeAt(0)
            #         if child.widget() is not None:
            #             child.widget().deleteLater()
            # self.result_image.setLayout(QVBoxLayout())  # 重新设置布局

            # self.result_text.clear()
            # self.status_bar.showMessage("内容已清空")
             # 清空原始图像
            if self.original_image.pixmap():
                self.original_image.setPixmap(QPixmap())
                self.original_image.setText("原始图像")  # 恢复默认文本

            # 清空结果图像
            self.result_image.setPixmap(QPixmap())
            if self.result_image.layout():
                # 清除所有子部件
                while self.result_image.layout().count():
                    item = self.result_image.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                # 删除布局
                QWidget().setLayout(self.result_image.layout())
            self.result_image.setText("处理结果")  # 恢复默认文本

            # 清空文本结果
            self.result_text.clear()
            
            # 更新状态栏
            self.status_bar.showMessage("内容已清空")
        except Exception as e:
            self.result_text.append(f"内容清除出现错误: {str(e)}")
            self.status_bar.showMessage("内容清除失败")

if __name__ == "__main__":
    # 创建应用程序和窗口对象
    app = QApplication(sys.argv)
    window = LicensePlateApp()
    # 显示窗口
    window.show()
    # 运行应用程序，并监听事件
    sys.exit(app.exec_())
