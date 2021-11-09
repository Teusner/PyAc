from Acoustics import *
from Material import *
from FDTD_solver import SolverFDTD2D

if __name__ == "__main__":
    dt = 5e-5
    dx = 1
    dy = 1

    xmin, xmax = 0, 1500
    ymin, ymax = 0, 140
    tmin, tmax = 0, 3

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[50, 5], [50, 50]]))

    # Emitters
    e1 = Emitter(0, 95, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Recievers
    r1 = Reciever(int(1400 / dx), int(95 / dy))
    s.recievers.append(r1)
    r2 = Reciever(int(1400 / dx), int(75 / dy))
    s.recievers.append(r2)
    r3 = Reciever(int(1400 / dx), int(55 / dy))
    s.recievers.append(r3)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    # M[:, :3] = air
    M[:, 100:120] = sediment
    M[:, 120:140] = basalt
    s.add_scene(M)

    # Simulation
    fps = 60
    dT = 1 / fps

    fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(np.zeros((s.m, s.n)), cmap="RdBu", vmin=-1e-3, vmax=1e-3, aspect="auto", interpolation="catrom")
    ax.set_xlabel(r"x ($m$)")
    ax.set_ylabel(r"y ($m$)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dx))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dy))
    fig.colorbar(im, orientation="horizontal")
    plt.title("Time: {:4.0f} ms".format(0))
    plt.savefig(f"./output/2dfdtd_{0:05}.png", dpi=180)

    for i, P in s.solve(dT):
        plt.title("Time: {:4.0f} ms".format(i * dt * 1000))
        im.set_array(np.sum(np.rot90(P[s.xindex, s.yindex]), axis=2))
        plt.savefig(f"./output/2dfdtd_{int(i * dt / dT) + 1:05}.png", dpi=180)

    for i, r in enumerate(s.recievers):
        fig, ax = r.temporal()
        plt.savefig(f"./output/temporal_reciever_{i}.png")
        fig, ax = r.fft()
        plt.savefig(f"./output/fft_reciever_{i}.png")
        fig, ax = r.spectrogram()
        plt.savefig(f"./output/spectorgram_reciever_{i}.png")