from Acoustics import *
from Material import *
from FDTD_solver import SolverFDTD2D
import time
from numba import cuda

if __name__ == "__main__":
    dt = 5e-5
    dx = 1
    dy = 1

    xmin, xmax = 0, 1000
    ymin, ymax = 0, 100
    tmin, tmax = 0, 3

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 20], [20, 20]]))

    # Emitters
    e1 = Emitter(0, 20, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Recievers
    r1 = Reciever(1000, 20)
    s.recievers.append(r1)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    s.add_scene(M)

    # Simulation
    print(f"Courant Number: {s.CourantNumber()}")
    fps = 30
    dT = 1 / fps

    # A = np.memmap("output/Pressure_Field.npy", dtype='float32', mode='w+', shape=(s.n, s.m, 3, s.t.size))
    # A[:, :, :, 0] = np.zeros((s.n, s.m, 3))

    t0 = time.time()
    k = 1
    for i, P in s.solve(dT):
        # A[:, :, :, k] = P[s.i_index, s.j_index]
        k += 1
        # np.save(f"output/Pressure_Field_{int(i * dt / dT) + 1:05}.npy", P)
    print(f"Took : {time.time() - t0}")
    cuda.profile_stop()
