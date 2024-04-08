import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset

test_path = Path('datasets') / 'co3d' / 'extracted_chair' / ' imgs_cropped' / 'val'
rows = 4
cols = 4

curr_r = 0
curr_c = 0

axs = None


def plot_image(idx, dataset):
    global axs, curr_r, curr_c
    if curr_c == 0 and curr_r == 0:
        plt.show()
        _, axs = plt.subplots(rows, cols, figsize=(rows*4, cols*4))
    img = dataset.__getitem__(idx)['input_im']
    axs[curr_c][curr_r].imshow(img.permute(1, 2, 0))
    axs[curr_c][curr_r].title.set_text(str(idx))

    curr_c += 1
    if curr_c >= cols:
        curr_c = 0
        curr_r += 1
    if curr_r >= rows:
        curr_r = 0


def main():
    print('Loading dataset')
    dataset = ImageDataset(str(test_path), image_size=256, crop=None, is_validation=True)

    indices = list(range(0, len(dataset), 1))

    for idx in indices:
        plot_image(idx, dataset)
    plt.show()


if __name__ == '__main__':
    main()