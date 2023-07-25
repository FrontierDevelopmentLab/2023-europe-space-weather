# File where we reconstruct the compressed image

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from torchvision import transforms
import torch

input_filename = "../payload/secchi_l0_a_seq_cor1_20120306_20120306_230000_s4c1a.fts"
latent_name = "../results/latent_32_32"


from compressai.zoo import bmshj2018_factorized
device = 'cpu'
net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)


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

### Now load the saved files:

with open(latent_name+".bytes", "rb") as f:
    strings_loaded = f.read()
strings_loaded = [[strings_loaded]]

a, b = int(latent_name.split("_")[1]), int(latent_name.split("_")[2])
shape_loaded = ([a,b])

with torch.no_grad():
    out_net = net.decompress(strings_loaded, shape_loaded)
    #(is already called inside) out_net['x_hat'].clamp_(0, 1)

x_hat = out_net['x_hat']
print("x_hat data range (min,mean,max):", torch.min(x_hat), torch.mean(x_hat), torch.max(x_hat)) # 0-1

print(out_net.keys())

rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())



### And compare

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].title.set_text('Input')
ax[0].imshow(orig_img, cmap="gray")
ax[1].title.set_text('Decompressed from onxx model')
ax[1].imshow(rec_net, cmap="gray")
plt.show()



