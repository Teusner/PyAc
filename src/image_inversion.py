from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import acoustics as ac
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import PowerNorm


def amplitude(dt, signal):
    print(np.mean(ac.signal.amplitude_envelope(signal, dt)))
    return np.mean(ac.signal.amplitude_envelope(signal, dt))

if __name__ == "__main__":
    n, m = 100, 100
    dx, dy = 0.25, 0.25
    a_emitter = 100
    # t_ret = np.load("output/time.npy")
    t_ret = np.linspace(0, 50/1000, 50)
    dt = t_ret[1] - t_ret[0]
    A = np.memmap("output/Pressure_Field.npy", dtype='float32', mode='r', shape=(int(n/dx), int(m/dy), 50))

    image = np.mean(ac.signal.amplitude_envelope(A[:, :, int(0.25*t_ret.size):int(0.75*t_ret.size)], dt, axis=2), axis=2)

    image = 20 * np.log10(image / np.max(image))

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(image , cmap="jet_r", aspect="equal", interpolation="catrom")
    # ctr = ax.contour(image, levels=[-35, -25, -18, -10, -5], colors="white", linewidths=0.6, linestyles="solid")
    ax.set_title("Transmission Loss")
    ax.set_xlabel(r"x ($m$)", fontsize="10")
    ax.set_ylabel(r"y ($m$)", fontsize="10")
    ax.tick_params(axis="both", labelsize="7")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * dx))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * dy))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="8%", pad=0.8)
    clb = fig.colorbar(im, orientation="horizontal", cax=cax)
    cax.set_title(r"Transmission Loss ($dB$)", fontsize="10")
    cax.tick_params(axis="both", labelsize="7")
    plt.savefig(f"./output/ampltude.png", dpi=180, bbox_inches='tight')
