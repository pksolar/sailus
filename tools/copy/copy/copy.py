import shutil
import os





def copy_all_py_files(src_folder, dst_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.py'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_folder, os.path.relpath(src_file, src_folder))
                dst_dir = os.path.dirname(dst_file)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy2(src_file, dst_file)



source_folder = r'E:\code\python_PK\tools'
destination_folder = r'E:\code\python_PK\tools\copy'

copy_all_py_files(source_folder, destination_folder)
