import scipy
from Acoustics import *
from Material import *

import cupy as cp
import math
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
from scipy import signal, integrate


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
        self.n = self.x.size
        self.m = self.y.size
        self.size = [self.x.size, self.y.size]

        # Border variable: [[top, left], [bottom, right]]
        self.border = np.zeros((2, 2))

        # Frequency range
        self.omega_range = 2 * np.pi * np.arange(20, 100, 0.1)

        # Relaxation time for Q approximation
        self.tau_sigma = np.array([1 / (2 * np.pi * 50)])

        # Cuda configuration
        self.threadsperblock = (8, 8)
        blockspergrid_x = math.ceil(self.size[0] / self.threadsperblock[0])
        blockspergrid_y = math.ceil(self.size[1] / self.threadsperblock[1])
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

    def add_scene(self, M):
        if M.shape != (self.n, self.m):
            raise ValueError("Scene has not the solver dimension!")
        self.M = M

    def allocate_arrays(self):
        self.U = cuda.to_device(np.zeros((self.size[0], self.size[1], 2), dtype=cp.float64))
        self.P = cuda.to_device(np.zeros((self.size[0], self.size[1], 3), dtype=cp.float64))
        self.R = cuda.to_device(np.zeros((self.size[0], self.size[1], 3), dtype=cp.float64))

        self.rho_i = cuda.to_device(np.array([[m.rho for m in line] for line in self.M]))
        self.eta_i = cuda.to_device(np.array([[m.eta for m in line] for line in self.M]))
        self.mu_i = cuda.to_device(np.array([[m.mu for m in line] for line in self.M]))
        Q_tau_g = self.Q_tau_gamma()
        self.tau_gamma_p_i = cuda.to_device(np.array([[Q_tau_g / m.Qp for m in line] for line in self.M]))
        self.tau_gamma_s_i = cuda.to_device(np.array([[Q_tau_g / m.Qs for m in line] for line in self.M]))
        self.B_i = cuda.to_device(self.Border().T)

    def set_border_attenuation(self, x):
        self.border = x
        self.size += np.array([x[0, 0] + x[1, 0], x[0, 1] + x[1, 1]])
        self.xindex = np.s_[x[0, 0] :self.n]
        self.yindex = np.s_[x[0, 1] :self.m]

    def Q_tau_gamma(self):
        def F(omega, tau_sigma):
            return np.sum([omega * t_s / (1 + (omega * t_s)**2) for t_s in tau_sigma])
        i1, _ = integrate.quad(F, self.omega_range[0], self.omega_range[-1], args=(self.tau_sigma))
        i2, _ = integrate.quad(lambda omega, tau : F(omega, tau)**2, self.omega_range[0], self.omega_range[-1], args=(self.tau_sigma))
        return i1 / i2

    def Border(self):
        k_min = 0.02
        ksi = lambda i, N, ksi_min: (1 - ksi_min) * ((1 + np.cos(i * np.pi / N)) / 2) ** 3 + ksi_min
        first_x = ksi(np.arange(-self.border[0, 0], 0, 1), self.border[0, 0], k_min)
        last_x = ksi(np.arange(0, self.border[1, 0], 1), self.border[1, 0], k_min)
        first_y = ksi(np.arange(-self.border[0, 1], 0, 1), self.border[0, 1], k_min)
        last_y = ksi(np.arange(0, self.border[1, 1], 1), self.border[1, 1], k_min)
        return np.sqrt(np.outer(np.hstack((first_y, np.ones(self.m), last_y)), np.hstack((first_x, np.ones(self.n), last_x))))

    def r(self, t, t0, omega_p):
        return (1 - 0.5 * (omega_p * (t - t0))**2) * cp.exp(- (omega_p * (t - t0))**2 / 4)

    def g(self, t, mu, sigma):
        return 1 / (sigma * cp.sqrt(2 * cp.pi)) * cp.exp(- (t - mu) ** 2 * (2 * sigma ** 2))

    def f(self, ti):
        F = cp.zeros(self.size, dtype=cp.float64)
        for e in self.emitters:
            c = e.coords + self.border[0]
            F[c[0], c[1]] = e[ti]
        return F

    def CourantNumber(self):
        Cp = np.array([[m.cp for m in line] for line in self.M])
        return np.max(Cp) * self.dt / np.sqrt(self.dx**2 + self.dy**2)

    def solve(self, dT):
        # Allocating extended arrays
        self.allocate_arrays()

        print(f"Courant Number: {self.CourantNumber()}")

        # Getting the number of frames to render
        N = int(dT / self.dt)
        for i in range (2, len(self.t) - 1):
            self.FDTD2D[self.blockspergrid, self.threadsperblock](self.P, self.R, self.U, self.B_i, self.f(i * self.dt), self.dx, self.dy, self.dt, self.rho_i, self.eta_i, self.mu_i, self.tau_gamma_p_i, self.tau_gamma_s_i, self.tau_sigma[0])
            for r in self.recievers:
                c = r.coords + self.border[0]
                r[i * self.dt] = np.sum(self.P[c[0], c[1]])
            if i % N == 0:
                yield i, self.P

    @staticmethod
    @cuda.jit
    def FDTD2D(P, R, U, B, F, dx, dy, dt, rho, eta, mu, tau_gamma_p, tau_gamma_s, tau_s):
        i, j = cuda.grid(2)
        n, m, _ = P.shape
        if 2 <= i <= n - 3  and 2 <= j <= m - 3:
            # Velocity computing
            dPxx = (- P[i, j+2, 0] + 27.0 * (P[i, j+1, 0] - P[i, j, 0]) + P[i, j-1, 0]) / (24 * dx)
            dPxy = (- P[i+1, j, 2] + 27.0 * (P[i, j, 2] - P[i-1, j, 2]) + P[i-2, j, 2]) / (24 * dy)
            U[i, j, 0] += dt / rho[i, j] * (dPxx + dPxy)
            dPxy = (- P[i, j+1, 2] + 27.0 * (P[i, j][2] - P[i, j-1, 2]) + P[i, j-2, 2]) / (24 * dx)
            dPyy = (- P[i+2, j, 1] + 27.0 * (P[i+1, j, 1] - P[i, j, 1]) + P[i-1, j, 1]) / (24 * dy)
            U[i, j, 1] += dt / rho[i, j] * (dPyy + dPxy)

            # Pressure computing
            Uxx = (- U[i, j+1, 0] + 27 * (U[i, j, 0] - U[i, j-1, 0]) + U[i, j-2, 0]) / (24 * dx)
            Uyy = (- U[i+1, j, 1] + 27 * (U[i, j, 1] - U[i-1, j, 1]) + U[i-2, j, 1]) / (24 * dy)
            Uyx = (- U[i, j+2, 1] + 27 * (U[i, j+1, 1] - U[i, j, 1]) + U[i, j-1, 1]) / (24 * dx)
            Uxy = (- U[i+2, j, 0] + 27 * (U[i+1, j, 0] - U[i, j, 0]) + U[i-1, j, 0]) / (24 * dy)

            R[i, j, 0] = (2 * tau_s - dt) / (2 * tau_s + dt) * R[i, j, 0] - 2 * dt / (2 * tau_s + dt) * (eta[i, j] * tau_gamma_p[i, j] * (Uxx + Uyy) - 2 * mu[i, j] * tau_gamma_s[i, j] * Uyy)
            R[i, j, 1] = (2 * tau_s - dt) / (2 * tau_s + dt) * R[i, j, 1] - 2 * dt / (2 * tau_s + dt) * (eta[i, j] * tau_gamma_p[i, j] * (Uxx + Uyy) - 2 * mu[i, j] * tau_gamma_s[i, j] * Uxx)
            R[i, j, 2] = (2 * tau_s - dt) / (2 * tau_s + dt) * R[i, j, 2] - 2 * dt / (2 * tau_s + dt) * (mu[i, j] * tau_gamma_s[i, j] * (Uxy + Uyx))

            P[i, j, 0] += dt * (eta[i, j] * (tau_gamma_p[i, j] + 1) * (Uxx + Uyy) - 2 * mu[i, j] * (tau_gamma_s[i, j] + 1) * Uyy + R[i, j, 0] + F[i, j])
            P[i, j, 1] += dt * (eta[i, j] * (tau_gamma_p[i, j] + 1) * (Uxx + Uyy) - 2 * mu[i, j] * (tau_gamma_s[i, j] + 1) * Uxx + R[i, j, 1] + F[i, j])
            P[i, j, 2] += dt * (mu[i, j] * (tau_gamma_s[i, j] + 1) * (Uxy + Uyx) + R[i, j, 2])

        # Boundary attenuation
        P[i, j, 0] *= B[i, j]
        P[i, j, 1] *= B[i, j]
        P[i, j, 2] *= B[i, j]
        U[i, j, 0] *= B[i, j]
        U[i, j, 1] *= B[i, j]
        R[i, j, 0] *= B[i, j]
        R[i, j, 1] *= B[i, j]
        R[i, j, 2] *= B[i, j]
        

    
if __name__ == "__main__":
    dt = 0.001
    dx = 0.5
    dy = 0.5

    xmin, xmax = 0, 200
    ymin, ymax = 0, 120
    tmin, tmax = 0, 30

    s = SolverFDTD2D(xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt)

    window = np.zeros(s.t.shape)
    window[:int(10 / dt)] = signal.tukey(int(10 / dt))
    e1 = Emitter(((xmax - xmin) / dx) // 2, ((ymax - ymin) / dy) // 2, lambda x : window[int(x / dt)] * np.sin(2 * np.pi * 50 * x))
    s.emitters.append(e1)
    # e2 = Emitter((3 *(M - m) / dx) // 4, (3 * (M - m) / dy) // 4, lambda x : 100 * np.sin(5 * x))
    # s.emitters.append(e2)

    r1 = Reciever((5 * (xmax - xmin) / dx) // 6, ((ymax - ymin) / dy) // 6)
    s.recievers.append(r1)
    r2 = Reciever((5 * (xmax - xmin) / dx) // 6, (4 * (ymax - ymin) / dy) // 6)
    s.recievers.append(r2)

    s.allocate_arrays()

    fps = 30
    dT = 1 / fps

    fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(np.zeros((s.m, s.n)), cmap="RdBu", vmin=-5e-4, vmax=5e-4, aspect="auto")
    ax.set_xlabel(r"x ($m$)")
    ax.set_ylabel(r"y ($m$)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dx))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: value * s.dy))
    fig.colorbar(im, orientation="horizontal")

    for i, P in s.solve(dT):
        plt.title("Time: {:.2f}s".format(i * dt))
        im.set_array(np.sum(P, axis=2).T) #[s.xindex, s.yindex]
        plt.savefig(f"./output/2dfdtd_{i:05}.png", dpi=180)

    # fig, ax = r1.temporal()
    # plt.savefig("./output/temporal_reciever.png")
    # fig, ax = r1.fft()
    # plt.savefig("./output/fft_reciever.png")
    # fig, ax = r1.spectrogram()
    # plt.savefig("./output/spectorgram_reciever.png")
    # fig, ax = e1.temporal()
    # plt.savefig("./output/temporal_emitter.png")
    # fig, ax = e1.fft()
    # plt.savefig("./output/fft_emitter.png")
    # fig, ax = e1.spectrogram()
    # plt.savefig("./output/spectrogram_emitter.png")