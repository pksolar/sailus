import win32gui
import win32com.client

def enum_windows_callback(hwnd, windows):
    if win32gui.IsWindowVisible(hwnd):
        windows.append(hwnd)
    return True
windows = []
win32gui.EnumWindows(enum_windows_callback, windows)
cabinet_windows = []
for hwnd in windows:
    class_name = win32gui.GetClassName(hwnd)
    if class_name == "CabinetWClass":
        cabinet_windows.append(hwnd)

shell = win32com.client.Dispatch("Shell.Application")

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
for path in folder_paths:
    print(path)

