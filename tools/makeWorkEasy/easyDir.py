import sys
import os
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
import win32com.client
from PyQt5.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self, folders):
        super().__init__()
        self.folders = folders
        self.initUI()
        self.setWindowIcon(QIcon('easyDir.jpg'))
    def initUI(self):
        # 添加重新运行按钮
        btn = QPushButton('重新运行', self)
        # btn.setGeometry(300, 250, 80, 30)
        btn.clicked.connect(self.restart_program)
        widget = QWidget()
        layout = QVBoxLayout()
        for folder in self.folders:
            label = QLabel(folder)
            label.setStyleSheet("border: 1px solid black; padding: 5px;")
            label.setOpenExternalLinks(True)
            label.mousePressEvent = lambda event, label=label: self.handle_click(event, label)
            layout.addWidget(label)
        layout.addWidget(btn)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle('Folders')
        self.setGeometry(100, 100, 400, 300)
        self.show()

    def handle_click(self, event, label):
        path = label.text()
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def restart_program(self):
        shell = win32com.client.Dispatch("Shell.Application")
        windows = shell.Windows()
        folder_paths = []
        for window in windows:
            if window.Document:
                folder = window.Document.Folder
                if folder:

                    folder_path = window.LocationURL.replace("file:///", "").replace("/", "\\")
                    if folder_path not in folder_paths:
                        folder_paths.append(folder_path)
        self.folders = sorted(folder_paths)
        self.initUI()


if __name__ == '__main__':
    shell = win32com.client.Dispatch("Shell.Application")
    windows = shell.Windows()
    folder_paths = []
    for window in windows:
        if window.Document:
            folder = window.Document.Folder
            if folder:

                folder_path = window.LocationURL.replace("file:///", "").replace("/", "\\")
                if folder_path not in folder_paths:
                    folder_paths.append(folder_path)
    # folders = [r'D:\SalusCall_Fast\offline_UI\platforms', r'C:\Users\user\Desktop\folder2', r'C:\Users\user\Desktop\folder3']
    app = QApplication(sys.argv)

    ex = MainWindow(sorted(folder_paths))
    ex.show()
    sys.exit(app.exec_())

