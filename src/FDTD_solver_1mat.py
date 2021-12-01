import scipy
from Acoustics import *
from Material import *

import cupy as cp
import math
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
from scipy import signal, integrate

from mpl_toolkits.axes_grid1 import make_axes_locatable

import time

# Global variables to be stored in shared memory [n, m, dx, dy, dt]
global_data = np.zeros(7)

class SolverFDTD2D:
    def __init__(self, xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt):
        # Time and spaces variables
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.t = cp.arange(tmin, tmax, dt)
        self.x = cp.arange(xmin, xmax, dx)
        self.y = cp.arange(ymin, ymax, dy)

        # Acoustics modules
        self.emitters = []
        self.recievers = []

        # Matrix size
        self.n = self.y.size
        self.m = self.x.size
        self.size = np.array([self.n, self.m])

        # Border variable: [[top, left], [bottom, right]]
        self.border = np.zeros((2, 2))

        # Frequency range
        self.omega_range = 2 * np.pi * np.arange(20, 100, 0.1)

        # Relaxation time for Q approximation
        self.tau_sigma = np.array([1 / (2 * np.pi * 50)])

    def add_scene(self, M):
        if M.shape != (self.n, self.m):
            raise ValueError("Scene has not the solver dimension!")
        self.M = M

    def allocate_arrays(self):
        global global_data
        global_data = np.array([self.size[0], self.size[1], 1/(24*self.dx), 1/(24*self.dy), self.dt, (2 * self.tau_sigma[0] - dt) / (2 * self.tau_sigma[0] + dt), 2 * dt / (2 * self.tau_sigma[0] + dt)])
        self.Field = cuda.to_device(np.zeros((self.size[0], self.size[1], 8), dtype=np.float32))

        b = (self.border / np.array([[self.dy, self.dx], [self.dy, self.dx]])).astype(np.int32)
        self.dtrho_i = cuda.to_device(np.pad(np.array([[dt / m.rho for m in line] for line in self.M]), pad_width=b, mode="edge"))
        self.eta_i = cuda.to_device(np.pad(np.array([[m.eta for m in line] for line in self.M]), pad_width=b, mode="edge"))
        self.mu_i = cuda.to_device(np.pad(np.array([[m.mu for m in line] for line in self.M]), pad_width=b, mode="edge"))
        Q_tau_g = self.Q_tau_gamma()
        self.tau_gamma_p_i = cuda.to_device(np.pad(np.array([[Q_tau_g / m.Qp for m in line] for line in self.M]), pad_width=b, mode="edge"))
        self.tau_gamma_s_i = cuda.to_device(np.pad(np.array([[Q_tau_g / m.Qs for m in line] for line in self.M]), pad_width=b, mode="edge"))
        self.B_i = cuda.to_device(self.Border().T)

    def set_border_attenuation(self, x):
        self.border = x
        self.size += (np.sum(x, axis=0).flatten() / np.array([self.dy, self.dx])).astype(np.int32)
        self.i_index = np.s_[int(x[0, 0] / self.dy):int(x[0, 0] / self.dy)+self.n]
        self.j_index = np.s_[int(x[0, 1] / self.dx):int(x[0, 1] / self.dx)+self.m]

    def Q_tau_gamma(self):
        def F(omega, tau_sigma):
            return np.sum([omega * t_s / (1 + (omega * t_s)**2) for t_s in tau_sigma])
        i1, _ = integrate.quad(F, self.omega_range[0], self.omega_range[-1], args=(self.tau_sigma))
        i2, _ = integrate.quad(lambda omega, tau : F(omega, tau)**2, self.omega_range[0], self.omega_range[-1], args=(self.tau_sigma))
        return i1 / i2

    def Border(self):
        k_min = 0.02
        ksi = lambda i, N, ksi_min: (1 - ksi_min) * ((1 + np.cos(i * np.pi / N)) / 2) ** 2 + ksi_min
        first_i = ksi(np.arange(-int(self.border[0, 0] / self.dy), 0, 1), int(self.border[0, 0] / self.dy), k_min)
        last_i = ksi(np.arange(0, int(self.border[1, 0] / self.dy), 1), int(self.border[1, 0] / self.dy), k_min)
        first_j = ksi(np.arange(-int(self.border[0, 1] / self.dx), 0, 1), int(self.border[0, 1] / self.dx), k_min)
        last_j = ksi(np.arange(0, int(self.border[1, 1] / self.dx), 1), int(self.border[1, 1] / self.dx), k_min)
        return np.sqrt(np.outer(np.hstack((first_j, np.ones(self.m), last_j)), np.hstack((first_i, np.ones(self.n), last_i))))

    def r(self, t, t0, omega_p):
        return (1 - 0.5 * (omega_p * (t - t0))**2) * cp.exp(- (omega_p * (t - t0))**2 / 4)

    def g(self, t, mu, sigma):
        return 1 / (sigma * cp.sqrt(2 * cp.pi)) * cp.exp(- (t - mu) ** 2 * (2 * sigma ** 2))

    def f(self, ti):
        F = cp.zeros(self.size, dtype=cp.float32)
        for e in self.emitters:
            F[int((e.y + self.border[0, 0]) / self.dy), int((e.x + self.border[0, 1]) / self.dx)] = e[ti]
        return F

    def CourantNumber(self):
        Cp = np.array([[m.cp for m in line] for line in self.M])
        return np.max(Cp) * self.dt / np.sqrt(self.dx**2 + self.dy**2)

    def solve(self, dT):
        # Allocating extended arrays
        self.allocate_arrays()

        # Cuda configuration
        self.threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(self.size[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.size[1] / self.threadsperblock[1])
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

        # ret = False
        # if dT is not None:
        #     N = int(dT / self.dt)
        #     ret = True

        # # Getting the number of frames to render
        # for i in range (2, len(self.t) - 1):
        #     self.FDTD2D[self.blockspergrid, self.threadsperblock](self.Field, self.B_i, self.f(i * self.dt), self.dtrho_i, self.eta_i, self.mu_i, self.tau_gamma_p_i, self.tau_gamma_s_i)
        #     for r in self.recievers:
        #         r[i * self.dt] = np.sum(self.P[int((r.y + self.border[0, 0])/ self.dy), int((r.x + self.border[0, 1]) / self.dx)])
        #     if ret and i % N == 0:
        #         yield i, np.sum(self.Field[self.i_index, self.j_index, 2:5], axis=2)

        def step(current_t):
            self.FDTD2D[self.blockspergrid, self.threadsperblock](self.Field, self.B_i, self.f(current_t), self.dtrho_i, self.eta_i, self.mu_i, self.tau_gamma_p_i, self.tau_gamma_s_i)
            for r in self.recievers:
                r[current_t] = np.sum(self.P[int((r.y + self.border[0, 0])/ self.dy), int((r.x + self.border[0, 1]) / self.dx)])

        current_t = 0
        for t_ret in dT:
            if t_ret > self.t[-1]:
                break
            while current_t < t_ret:
                current_t += self.dt
                step(current_t)
            yield current_t, np.sum(self.Field[self.i_index, self.j_index, 2:5], axis=2)
        

    @staticmethod
    @cuda.jit
    def get_P(Field, P):
        i, j = cuda.grid(2)
        P[i, j] = 0
        for k in range(3):
            P[i, j] += Field[i, j, k]

    @staticmethod
    @cuda.jit
    def FDTD2D(Field, B, F, dtrho, eta, mu, tau_gamma_p, tau_gamma_s):
        i, j = cuda.grid(2)
        g = cuda.const.array_like(global_data)
        n, m, dx24, dy24, dt, a, b = g[0], g[1], g[2], g[3], g[4], g[5], g[6]
        if 2 <= i <= n - 3  and 2 <= j <= m - 3:
            # Velocity computing
            dPxx = (- Field[i, j+2, 2] + 27.0 * (Field[i, j+1, 2] - Field[i, j, 2]) + Field[i, j-1, 2]) * dx24
            dPxy = (- Field[i+1, j, 4] + 27.0 * (Field[i, j, 4] - Field[i-1, j, 4]) + Field[i-2, j, 4]) * dy24
            Field[i, j, 0] += dtrho[i, j] * (dPxx + dPxy)
            dPxy = (- Field[i, j+1, 4] + 27.0 * (Field[i, j, 4] - Field[i, j-1, 4]) + Field[i, j-2, 4]) * dx24
            dPyy = (- Field[i+2, j, 3] + 27.0 * (Field[i+1, j, 3] - Field[i, j, 3]) + Field[i-1, j, 3]) * dy24
            Field[i, j, 1] += dtrho[i, j] * (dPyy + dPxy)

            # Pressure computing
            Uxx = (- Field[i, j+1, 0] + 27 * (Field[i, j, 0] - Field[i, j-1, 0]) + Field[i, j-2, 0]) * dx24
            Uyy = (- Field[i+1, j, 1] + 27 * (Field[i, j, 1] - Field[i-1, j, 1]) + Field[i-2, j, 1]) * dy24
            Uyx = (- Field[i, j+2, 1] + 27 * (Field[i, j+1, 1] - Field[i, j, 1]) + Field[i, j-1, 1]) * dx24
            Uxy = (- Field[i+2, j, 0] + 27 * (Field[i+1, j, 0] - Field[i, j, 0]) + Field[i-1, j, 0]) * dy24

            Field[i, j, 5] = a * Field[i, j, 5] - b * (eta[i, j] * tau_gamma_p[i, j] * (Uxx + Uyy) - 2 * mu[i, j] * tau_gamma_s[i, j] * Uyy)
            Field[i, j, 6] = a * Field[i, j, 6] - b * (eta[i, j] * tau_gamma_p[i, j] * (Uxx + Uyy) - 2 * mu[i, j] * tau_gamma_s[i, j] * Uxx)
            Field[i, j, 7] = a * Field[i, j, 7] - b * (mu[i, j] * tau_gamma_s[i, j] * (Uxy + Uyx))

            Field[i, j, 2] += dt * (eta[i, j] * (tau_gamma_p[i, j] + 1) * (Uxx + Uyy) - 2 * mu[i, j] * (tau_gamma_s[i, j] + 1) * Uyy + Field[i, j, 5] + F[i, j])
            Field[i, j, 3] += dt * (eta[i, j] * (tau_gamma_p[i, j] + 1) * (Uxx + Uyy) - 2 * mu[i, j] * (tau_gamma_s[i, j] + 1) * Uxx + Field[i, j, 6] + F[i, j])
            Field[i, j, 4] += dt * (mu[i, j] * (tau_gamma_s[i, j] + 1) * (Uxy + Uyx) + Field[i, j, 7])

        # Boundary attenuation
        for q in range(8):
            Field[i, j, q] *= B[i, j]

    
if __name__ == "__main__":
    dt = 1e-5
    dx = 0.5
    dy = 0.5

    xmin, xmax = 0, 100
    ymin, ymax = 0, 100
    tmin, tmax = 0, 1

    # Solver
    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    # Boundary conditions
    s.set_border_attenuation(np.array([[0, 40], [40, 40]]))

    # Emitters
    e1 = Emitter(50, 50, lambda x : 100 * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)

    # Recievers
    # r1 = Reciever(1000, 20)
    # s.recievers.append(r1)

    # Materials
    M = np.empty((s.n, s.m), dtype=object)
    M[:, :] = water
    s.add_scene(M)

    # Simulation
    print(f"Courant Number: {s.CourantNumber()}")
    fps = 5000
    dT = 1 / fps

    t_ret = np.arange(0.9, 1, 0.0005)

    # fig, ax = plt.subplots(1, 1)
    # im = ax.imshow(np.zeros((s.n, s.m)), cmap="RdBu", vmin=-6e-4, vmax=6e-4, aspect="equal", interpolation="catrom")
    # ax.set_title("Time: {:4.0f} ms".format(0))
    # ax.set_xlabel(r"x ($m$)", fontsize="10")
    # ax.set_ylabel(r"y ($m$)", fontsize="10")
    # ax.tick_params(axis="both", labelsize="7")
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dx))
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dy))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("bottom", size="10%", pad=0.8)
    # clb = fig.colorbar(im, orientation="horizontal", cax=cax)
    # cax.set_title(r"Pressure Field ($Pa$)", fontsize="10")
    # cax.tick_params(axis="both", labelsize="7")
    # plt.savefig(f"./output/2dfdtd_{0:05}.png", dpi=180, bbox_inches='tight')

    print(s.n, s.m)

    # A = np.memmap("output/Pressure_Field.npy", dtype='float32', mode='w+', shape=(s.n, s.m, t_ret.size))
    # A[:, :, :] = np.zeros((s.n, s.m, t_ret.size))

    # k = 0
    # t0 = time.time()
    # for current_time, P in s.solve(t_ret):
    #     A[:, :, k] = P
    #     k += 1
    #     # ax.set_title("Time: {:4.0f} ms".format(i * dt * 1000))
    #     # im.set_data(P)
    #     # plt.savefig(f"./output/2dfdtd_{int(i * dt / dT) + 1:05}.png", dpi=180, bbox_inches='tight')
    # print(f"Took : {time.time() - t0}")

    # np.save("output/time.npy", t_ret)