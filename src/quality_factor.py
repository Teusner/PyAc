import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as integrate


def tau(omega, Q0, tau_sigma):
    def F(omega, tau_sigma):
        return np.sum([omega * t_s / (1 + (omega * t_s)**2) for t_s in tau_sigma])
    i1, _ = integrate.quad(F, omega[0], omega[-1], args=(tau_sigma))
    i2, _ = integrate.quad(lambda omega, tau : F(omega, tau)**2, omega[0], omega[-1], args=(tau_sigma))
    return i1 / (i2 * Q0)

def Q(omega, tau, tau_sigma):
    def q(omega, tau, tau_sigma_l):
        return omega * tau_sigma_l * tau / (1 + (omega * tau_sigma_l)**2)
    return 1 / np.sum([q(omega, tau, t_s) for t_s in tau_sigma], axis=0)

if __name__ == "__main__":
    # Quality Factor
    Q_0 = 1000

    # Frequency / Pulsation range
    f = np.arange(1e1, 1e2, 1)
    omega = 2 * np.pi * f

    # Tau_sigma distribution over the frequency range defining the relaxation mechanism
    tau_s = np.array([1 / (2 * np.pi * 50)])

    # Optimal tau computation
    tau_g = tau(omega, Q_0, tau_s)
    print(tau_g)
    print((tau_g + 1) * tau_s)

    # Q computation using optimal tau and tau_sigma
    Q_tau = Q(omega, tau_g, tau_s)

    plt.figure()
    plt.plot(f, Q_tau, label=r"Optimal $Q(\omega)$")
    plt.plot(f, Q_0 * np.ones(f.shape), label=r"$Q_0$")
    plt.grid(True)
    plt.axis([f[0], f[-1], 0, 1.5*np.max(Q_tau)])
    plt.title(r"Optimal quality factor on a frequency range")
    plt.xlabel(r"Frequency ($Hz$)")
    plt.ylabel(r"Quality factor")
    plt.legend(loc="best")
    plt.show()

    print(tau)