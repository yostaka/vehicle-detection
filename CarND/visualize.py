import matplotlib.pyplot as plt


def visualize(imgs, titles, cmaps=None, fname=None, ncols=None):
    num_images = len(imgs)

    if len(imgs) == 1:
        plt.figure()
        if cmaps is not None:
            plt.imshow(imgs[0], cmap=cmaps[0])
        else:
            plt.imshow(imgs[0])
    else:
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

        if nrows == 1:
            for idx in range(num_images):
                if cmaps is None:
                    ax[idx].imshow(imgs[idx])
                    ax[idx].set_title(titles[idx], fontsize=30)
                else:
                    ax[idx].imshow(imgs[idx], cmap=cmaps[idx])
                    ax[idx].set_title(titles[idx], fontsize=30)
        else:
            for idx in range(num_images):
                if cmaps is None:
                    ax[idx // ncols, idx % ncols].imshow(imgs[idx])
                    ax[idx // ncols, idx % ncols].set_title(titles[idx], fontsize=30)
                else:
                    ax[idx // ncols, idx % ncols].imshow(imgs[idx], cmap=cmaps[idx])
                    ax[idx // ncols, idx % ncols].set_title(titles[idx], fontsize=30)


    # TODO: adjust spacing around figures
    plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

