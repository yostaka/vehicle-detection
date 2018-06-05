import matplotlib.pyplot as plt


def visualize(imgs, titles, cmaps=None, fname=None, ncols=None):
    num_images = len(imgs)

    if ncols is None:

        if num_images % 2 == 0:
            ncols = 2
        else:
            ncols = 3

    if num_images % ncols == 0:
        nrows = num_images // ncols
    else:
        nrows = num_images // ncols + 1

    plt.figure()
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 9))
    f.tight_layout()

    for idx in range(num_images):
        if cmaps is None:
            ax[idx // ncols, idx % ncols].imshow(imgs[idx])
            ax[idx // ncols, idx % ncols].set_title(titles[idx], fontsize=30)
        else:
            ax[idx // ncols, idx % ncols].imshow(imgs[idx], cmap=cmaps[idx])
            ax[idx // ncols, idx % ncols].set_title(titles[idx], fontsize=30)

    # TODO: adjust spacing around figures
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

