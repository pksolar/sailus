import SimpleITK as sitk
import skimage.io as io


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


# 显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        print(i)
        io.show()


# 单张显示
def show_img(ori_img):
    io.imshow(ori_img[100], cmap='gray')
    print("single")
    io.show()


# window下的文件夹路径
path = 'E:\\ProgramCode\\python\\voxelmorph\\VoxelMorph-torch-master\\LPBA40\\label\\S03.delineation.structure.label.nii.gz'
data = read_img(path)
show_img(data)