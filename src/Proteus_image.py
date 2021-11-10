from Acoustics import *
from Material import *
from FDTD_solver_cpu import SolverFDTD2D
import time
import acoustics as ac

import sys

if __name__ == "__main__":
    dt = 1e-4
    dx = 1
    dy = 1

    xmin, xmax = 0, 1000
    ymin, ymax = 0, 100
    tmin, tmax = 0, 1

    j = int(sys.argv[1])

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 20], [20, 20]]))

    # Recievers
    r1 = Reciever(1000, 20)
    s.recievers.append(r1)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    s.add_scene(M)

    # Emitters
    e1 = Emitter(0, 20, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Generating image
    f = open(f"output/line_{j}.txt", "a")
    f.write("# X\tY\tValue\tTime\n")
    f.close()
    for i in s.x:
        t0 = time.time()
        s.emitters[0].x, s.emitters[0].y = i, j
        for _ in s.solve():
            pass
        t = time.time() - t0
        val = np.mean(ac.signal.amplitude_envelope(r1.s, dt)[-100:-50])
        f = open(f"output/line_{j}.txt", "a")
        f.write(f"{i}\t{j}\t{val}\t{t}\n")
        f.close()

