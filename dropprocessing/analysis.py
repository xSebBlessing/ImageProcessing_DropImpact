from matplotlib import pyplot as plt
import numpy as np

def show_images(image_list, image_id, n_images_before, n_images_after):
    fig, axs = plt.subplots(n_images_before + n_images_after + 1, 1)

    idxs = np.arange(-n_images_before, n_images_after+1)

    for idx, ax in zip(idxs, axs):
        ax.imshow(image_list[image_id + idx], cmap="gray")
    
    plt.show()
    