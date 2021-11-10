from Acoustics import *
from Material import *
from FDTD_solver_cpu import SolverFDTD2D
import pickle
import time
import acoustics as ac

if __name__ == "__main__":
    dt = 2e-4
    dx = 2
    dy = 2

    xmin, xmax = 0, 2000
    ymin, ymax = 0, 140
    tmin, tmax = 0, 2

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 20], [20, 20]]))

    # Emitters
    e1 = Emitter(0, 20, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Recievers
    r1 = Reciever(2000, 20)
    s.recievers.append(r1)
    # r2 = Reciever(2000, 40)
    # s.recievers.append(r2)
    # r3 = Reciever(2000, 60 / dy)
    # s.recievers.append(r3)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    # M[:3, :] = air
    # M[int(100/dy):int(120/dy), :] = sediment
    # M[int(120/dy):int(140/dy), :] = basalt
    s.add_scene(M)

    # Simulation
    fps = 30
    dT = 1 / fps

    fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(np.zeros((s.n, s.m)), cmap="RdBu", vmin=-1e-3, vmax=1e-3, aspect="auto", interpolation="catrom")
    ax.set_xlabel(r"x ($m$)")
    ax.set_ylabel(r"y ($m$)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dx))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dy))
    fig.colorbar(im, orientation="horizontal")
    plt.title("Time: {:4.0f} ms".format(0))
    plt.savefig(f"./output/2dfdtd_{0:05}.png", dpi=180)

    t0 = time.time()
    for i, P in s.solve():
        plt.title("Time: {:4.0f} ms".format(i * dt * 1000))
        im.set_data(np.sum(P[s.i_index, s.j_index], axis=2))
        plt.savefig(f"./output/2dfdtd_{int(i * dt / dT) + 1:05}.png", dpi=180)
    print(f"Took : {time.time() - t0}")

    # for i, r in enumerate(s.recievers):
    #     fig, ax = r.temporal()
    #     plt.savefig(f"./output/temporal_reciever_{i}.png")
    #     fig, ax = r.fft()
    #     plt.savefig(f"./output/fft_reciever_{i}.png")
    #     fig, ax = r.spectrogram()
    #     plt.savefig(f"./output/spectorgram_reciever_{i}.png")

    #     filehandler = open(f"./output/Reciever_{i}.obj","wb")
    #     pickle.dump(r, filehandler)
    #     filehandler.close()
