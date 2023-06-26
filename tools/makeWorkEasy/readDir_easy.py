import win32com.client

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
