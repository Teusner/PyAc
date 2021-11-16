from Acoustics import *
from Material import *

import math
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from scipy import signal, integrate


class SolverFDTD2D:
    def __init__(self, xmin, xmax, dx, ymin, ymax, dy, tmin, tmax, dt):
        # Time and spaces variables
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.t = np.arange(tmin, tmax, dt)
        self.x = np.arange(xmin, xmax, dx)
        self.y = np.arange(ymin, ymax, dy)

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
        self.P = np.zeros((self.size[0], self.size[1], 3), dtype=np.float64)
        self.U = np.zeros((self.size[0], self.size[1], 2), dtype=np.float64)
        self.R = np.zeros((self.size[0], self.size[1], 3), dtype=np.float64)

        b = (self.border / np.array([[self.dy, self.dx], [self.dy, self.dx]])).astype(np.int64)
        self.rho_i = np.pad(np.array([[m.rho for m in line] for line in self.M]), pad_width=b, mode="edge")
        self.eta_i = np.pad(np.array([[m.eta for m in line] for line in self.M]), pad_width=b, mode="edge")
        self.mu_i = np.pad(np.array([[m.mu for m in line] for line in self.M]), pad_width=b, mode="edge")
        Q_tau_g = self.Q_tau_gamma()
        self.tau_gamma_p_i = np.pad(np.array([[Q_tau_g / m.Qp for m in line] for line in self.M]), pad_width=b, mode="edge")
        self.tau_gamma_s_i = np.pad(np.array([[Q_tau_g / m.Qs for m in line] for line in self.M]), pad_width=b, mode="edge")
        self.B_i = self.Border().T

    def set_border_attenuation(self, x):
        self.border = x
        self.size += (np.sum(x, axis=0).flatten() / np.array([self.dy, self.dx])).astype(np.int64)
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
        return (1 - 0.5 * (omega_p * (t - t0))**2) * np.exp(- (omega_p * (t - t0))**2 / 4)

    def g(self, t, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (t - mu) ** 2 * (2 * sigma ** 2))

    def f(self, ti):
        F = np.zeros(self.size, dtype=np.float64)
        for e in self.emitters:
            F[int((e.y + self.border[0, 0]) / self.dy), int((e.x + self.border[0, 1]) / self.dx)] = e[ti]
        return F

    def CourantNumber(self):
        Cp = np.array([[m.cp for m in line] for line in self.M])
        return np.max(Cp) * self.dt / np.sqrt(self.dx**2 + self.dy**2)

    def solve(self, *args, **kwargs):
        # Allocating extended arrays
        self.allocate_arrays()

        print(f"Courant Number: {self.CourantNumber()}")

        ret = False
        dT = None
        for arg in args:
            dT = arg
        if dT is not None:
            N = int(dT / self.dt)
            ret = True

        # Getting the number of frames to render
        for i in range (2, len(self.t) - 1):
            self.FDTD2D(self.P, self.R, self.U, self.B_i, self.f(i * self.dt), self.dx, self.dy, self.dt, self.rho_i, self.eta_i, self.mu_i, self.tau_gamma_p_i, self.tau_gamma_s_i, self.tau_sigma[0])
            for r in self.recievers:
                r[i * self.dt] = np.sum(self.P[int((r.y + self.border[0, 0])/ self.dy), int((r.x + self.border[0, 1]) / self.dx)])
            if ret and i % N == 0:
                    yield i, self.P

    @staticmethod
    @jit
    def FDTD2D(P, R, U, B, F, dx, dy, dt, rho, eta, mu, tau_gamma_p, tau_gamma_s, tau_s):
        n, m, _ = P.shape
        for i in range(2, n-2):
            for j in range(2, m-2):
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