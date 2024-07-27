from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torchvision
import math
import pandas
import matplotlib.pyplot as plt

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        print('make path:', path)
        os.makedirs(path)

def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_str_data(data, path):
    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")

def draw_log_csv(column_to_draw:str, data, logy=False):
    plot_df = pandas.DataFrame(index=list(range(256)))

    for name, log_csv in data:
        df = pandas.read_csv(log_csv)
        df = df.sort_values(by='epoch')
        df.set_index('epoch', inplace=True)

        # print(df)
        plot_df[name] = df[column_to_draw]

    print(plot_df)
    plot_df.plot(title=column_to_draw, logy=logy)
    plt.show()


if __name__ == '__main__':

    draw_log_csv('LR_exp_acc', [
        ('RAF_res', '/home/liangcong/dataset/Privacycheckpoints/RAF_res/results.csv'),
        ('RAF_res_Gu', '/home/liangcong/dataset/Privacycheckpoints/RAF_res_Gu/results.csv'),
        # ('MUG_adv_0.5', '/home/liangcong/dataset/Privacycheckpoints/new_MUG16_adv_L0.5/results.csv'),
        # ('MUG_adv_0.1', '/home/liangcong/dataset/Privacycheckpoints/new_MUG16_adv_L0.1/results.csv'),
        # ('MUG_adv_2', '/home/liangcong/dataset/Privacycheckpoints/new_MUG16_adv_L2/results.csv'),
    ], logy=False)