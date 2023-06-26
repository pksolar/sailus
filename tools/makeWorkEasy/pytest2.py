import sys
import random
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Random Numbers')
        self.setGeometry(100, 100, 400, 200)

        # 创建三个标签用于显示随机数
        self.num1 = QLabel(str(random.randint(1, 100)))
        self.num2 = QLabel(str(random.randint(1, 100)))
        self.num3 = QLabel(str(random.randint(1, 100)))

        # 创建一个按钮，用于更新随机数
        self.update_button = QPushButton('更新')
        self.update_button.clicked.connect(self.update_numbers)

        # 创建一个垂直布局，并将标签和按钮添加到其中
        vbox = QVBoxLayout()
        vbox.addWidget(self.num1)
        vbox.addWidget(self.num2)
        vbox.addWidget(self.num3)
        vbox.addWidget(self.update_button)

        # 创建一个QWidget并将垂直布局添加到其中
        widget = QWidget()
        widget.setLayout(vbox)

        # 将QWidget设置为主窗口的中心部件
        self.setCentralWidget(widget)

    def update_numbers(self):
        # 更新三个随机数并将其显示在标签上
        self.num1.setText(str(random.randint(1, 100)))
        self.num2.setText(str(random.randint(1, 100)))
        self.num3.setText(str(random.randint(1, 100)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
