import torch
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def vae_kl_loss(mu, log_var):
    return -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum()


def show_images(images):
    images = torch.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # print(img)
        plt.imshow(img.reshape([sqrtimg,sqrtimg]), cmap='gray')
    plt.show()