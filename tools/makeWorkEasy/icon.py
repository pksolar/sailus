from PIL import Image
import PythonMagick


def gen_ico_pill(image_path, resize=28, suffix='.png'):
    save_path = image_path.replace(suffix, '_pil.ico')
    image = Image.open(image_path)
    image_resize = image.resize((resize, resize), Image.LANCZOS)
    image_resize.save(save_path)


def gen_ico_magick(image_path, resize=28, suffix='.png'):
    save_path = image_path.replace(suffix, '_magick.ico')
    image = PythonMagick.Image(image_path)
    image.sample(f'{resize}x{resize}')  # 报错：RuntimeError: Magick: negative or zero image size `' @ error/image.c/CloneImage/811  # 是因为我没有给Image() 传入路径参数，laugh cry
    image.write(save_path)


if __name__ == '__main__':
    image_path = 'images/tree.png'
    gen_ico_pill(image_path, resize=28, suffix='.png')
    gen_ico_magick(image_path, resize=28, suffix='.png')
