from PyQt5.QtCore import QDir, QFileSystemWatcher, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication


class FolderWatcher(QObject):
    folder_opened = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self.on_folder_opened)

        # 添加所有文件夹路径到监视列表中
        for drive in QDir.drives():
            self.watcher.addPath(drive.absolutePath())

    def on_folder_opened(self, folder_path):
        self.folder_opened.emit()


if __name__ == '__main__':
    app = QApplication([])
    watcher = FolderWatcher()

    def on_folder_opened():
        print("open")

    watcher.folder_opened.connect(on_folder_opened)
    app.exec_()
