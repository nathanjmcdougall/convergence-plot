"""A script to create nice animations of Julia sets for families of functions.
"""
from matplotlib.animation import ArtistAnimation

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_recursions_convergence

if __name__ == '__main__':
    xlim = [-2, 2]
    ylim = [-2, 2]
    extent = xlim + ylim
    PIXELS_PER_AXIS = 2000
    THRESHOLD = 2
    MAX_IT = 20

    def julia_func(theta):
        """Returns a julia function for the given parameter a.
        """
        constant = 0.2*np.exp(theta*1j)
        return lambda z: z**4 - 1.3*z + constant

    # TODO(NAMC) incorporate the more efficient method in this SO answer:
    # https://stackoverflow.com/a/15883620/10931340

    fig, ax = plt.subplots(figsize=(10, 10))
    frames = tqdm(np.linspace(0, 2*np.pi, num=100))
    ims = []
    for frame in frames:
        im = plot_recursions_convergence(
            ax,
            func=julia_func(frame),
            threshold=THRESHOLD,
            max_it=MAX_IT,
            extent=extent,
            pixels_per_axis=PIXELS_PER_AXIS,
            cmap='inferno'
            )
        ax.axis('off')
        ims.append([im])

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(19.20, 10.80, True)

    ani = ArtistAnimation(fig, ims, interval=30, blit=True)
    ani.save("animation.mp4", dpi=100)

    plt.show()
