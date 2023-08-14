from astropy.io import fits
import numpy as np
from torchvision import transforms
import torch


def load_fits_as_np(input_filename):
    orig_img = fits.getdata(input_filename)
    print(orig_img.shape, orig_img.dtype)
    print(np.min(orig_img), np.mean(orig_img), np.max(orig_img))
    # plt.imshow(orig_img, cmap="gray")

    img = orig_img.copy()

    print("original data range (min,mean,max):", np.min(img), np.mean(img), np.max(img))  # 0-16k

    img = np.asarray([img, img, img])  # fake rgb
    img = np.transpose(img, (1, 2, 0)).astype(np.int16)

    # normalise to [0, 255]
    img = (img - img.min()) / (
            img.max() - img.min()
    )
    img = img.astype(np.float32)
    print("normalised data range (min,mean,max):", np.min(img), np.mean(img), np.max(img))  # 0-1

    x = transforms.ToTensor()(img)
    x = x.unsqueeze(0).to('cpu')
    print("x data range (min,mean,max):", torch.min(x), torch.mean(x), torch.max(x))  # 0-1

    x_np = x.cpu().detach().numpy()
    print(x_np.shape)

    return x_np