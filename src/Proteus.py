from Acoustics import *
from Material import *
from FDTD_solver import SolverFDTD2D

if __name__ == "__main__":
    dt = 1e-6
    dx = 0.1
    dy = 0.1

    xmin, xmax = 0, 2000
    ymin, ymax = 0, 120
    tmin, tmax = 0, 1

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 20], [20, 20]]))

    # Emitters
    e1 = Emitter(int((xmax - xmin) / (2 * dx)), int((ymax - ymin) / (2 * dy)), lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Recievers
    # r1 = Reciever(int(1950 / dx), int(20 / dy))
    # s.recievers.append(r1)
    # r2 = Reciever(int(1950 / dx), int(40 / dy))
    # s.recievers.append(r2)
    # r3 = Reciever(int(1950 / dx), int(60 / dy))
    # s.recievers.append(r2)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    s.add_scene(M)

    # Simulation
    fps = 10000
    dT = 1 / fps

    fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(np.zeros((s.m, s.n)), cmap="RdBu", vmin=-5e-4, vmax=5e-4, aspect="auto")
    ax.set_xlabel(r"x ($m$)")
    ax.set_ylabel(r"y ($m$)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dx))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dy))
    fig.colorbar(im, orientation="horizontal")
    plt.title("Time: {:.4f}s".format(0))
    plt.savefig(f"./output/2dfdtd_{0:04}.png", dpi=180)

    for i, P in s.solve(dT):
        plt.title("Time: {:.4f}s".format(i * dt))
        im.set_array(np.sum(P[s.yindex, s.xindex], axis=2).T)
        plt.savefig(f"./output/2dfdtd_{int(i * dt / dT) + 1:04}.png", dpi=180)