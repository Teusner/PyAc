from Acoustics import *
from Material import *
from FDTD_solver_cpu import SolverFDTD2D
import time
import acoustics as ac

import sys

if __name__ == "__main__":
    # Usage python3 Proteus_image.py t x y dt dx dy xr yr i n, where:
    # t = time-range in seconds
    # x = x-range in meters
    # y = y-range in meters
    # dt = t-step in seconds
    # dx = x-step in meters
    # dy = y-step in meters
    # xr = reciever x position
    # yr = reciever y position
    # i = current job number
    # n = total number of jobs

    tmin, tmax = 0, float(sys.argv[1])
    xmin, xmax = 0, float(sys.argv[2])
    ymin, ymax = 0, float(sys.argv[3])
    dt = float(sys.argv[4])
    dx = float(sys.argv[5])
    dy = float(sys.argv[6])
    xr = float(sys.argv[7])
    yr = float(sys.argv[8])
    j_job = int(sys.argv[9])
    n = int(sys.argv[10])

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 20], [20, 20]]))

    # Recievers
    r1 = Reciever(xr, yr)
    s.recievers.append(r1)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    s.add_scene(M)

    # Emitters
    e1 = Emitter(0, 0, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Generating data
    f = open(f"output/job_{j_job}.out", "a")
    f.write(f"# Time: [{tmin}, {tmax}], step: {dt}\n")
    f.write(f"# X: [{xmin}, {xmax}], step: {dx}\n")
    f.write(f"# Y: [{ymin}, {ymax}], step: {dy}\n")
    f.write(f"# Reciever: [{xr}, {yr}]\n")
    f.write(f"# Job: [{j_job}/{n}]\n")
    f.write("# i\tj\tX\tY\tValue\tTime\n")
    f.close()

    X = int((xmax - xmin) / dx)
    Y = int((ymax - ymin) / dy)
    for index in range(j_job, X * Y, n + 1):
        t0 = time.time()
        s.emitters[0].x, s.emitters[0].y = (index%s.m) * dx, (index//s.m) * dy
        for _ in s.solve():
            pass
        t = time.time() - t0
        val = np.mean(ac.signal.amplitude_envelope(r1.s, dt)[-100:-50])
        f = open(f"output/job_{j_job}.out", "a")
        f.write(f"{index//s.m}\t{index%s.m}\t{s.emitters[0].x}\t{s.emitters[0].y}\t{val}\t{t}\n")
        f.close()