"""
==============================================================
Acoustics, (:mod:`pyac.Acoustics`)
==============================================================
This module provides a number of helpful objects to deal with
Finite Difference Time Domain simulation, such as `Emitter`,
`Reciever` modules.

"""
from dataclasses import dataclass, field, InitVar
from typing import Callable
import abc

import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))

def d_gaussian(x, mu, sig):
    return - (x - mu) / (np.power(sig, 3.) * np.sqrt(2 * np.pi)) * np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

@dataclass
class Module(abc.ABC):
    """An abstract base class for acoustic module.
    
    Module is an abstract base class for representing acoustic
    module such as Emitter or Reciever.

    Parameters
    ----------
    x : float
        `x`-coordinate of the Module in meter.
    y : float
        `y`-coordinate of the Module in meter.
    """
    x: float
    y: float
    t: list = field(init=False, default_factory=list)
    s: list = field(init=False, default_factory=list)

    def temporal(self):
        """Plot the signal of the Module
        Returns
        -------
        [fig, ax] : matplotlib.pyplot.figure, matplotlib.pyplot.axes
            Figure and axes representing the signal of the module.
        See Also
        --------
        fft, spectrogram
        """
        fig, ax = plt.subplots()
        ax.set_title("Signal of " + str(self))
        ax.set_xlabel(r"Time ($s$)")
        ax.set_ylabel(r"Pressure ($Pa$)")
        ax.grid(True)
        ax.plot(self.t, self.s)
        ax.set_xlim(np.min(self.t), np.max(self.t))
        return fig, ax

    def fft(self):
        """Plot the Fast Fourier Transform of the Module
        Returns
        -------
        [fig, ax] : matplotlib.pyplot.figure, matplotlib.pyplot.axes
            Figure and axes representing the Fast Fourier Transform of the module.
        See Also
        --------
        temporal, spectrogram
        """
        fig, ax = plt.subplots()
        ax.set_title(f"FFT of " + str(self))
        ax.set_xlabel(r"Frequency ($Hz$)")
        ax.set_ylabel(r"Magnitude ($Power$)")
        ax.grid(True)
        freq = np.fft.fftfreq(len(self.t), self.t[1] - self.t[0])
        sp = np.fft.fft(self.s)
        ax.plot(freq, np.abs(sp.real))
        ax.set_xlim(np.min(freq), np.max(freq))
        return fig, ax

    def spectrogram(self):
        """Plot the spectrogram of the Module
        Returns
        -------
        [fig, ax] : matplotlib.pyplot.figure, matplotlib.pyplot.axes
            Figure and axes representing the spectrogram of the module.
        See Also
        --------
        temporal, fft
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Spectrogram of " + str(self))
        ax.set_xlabel(r"Time ($s$)")
        ax.set_ylabel(r"Frequency ($Hz$)")
        ax.grid(True)
        f, t, Sxx = signal.spectrogram(np.array(self.s), fs=30)
        ax.pcolormesh(t, f, Sxx, shading="auto")
        return fig, ax


@dataclass
class Emitter(Module):
    dt: float
    t: np.ndarray
    dx: float
    dy: float
    dz: float

    omega_0: InitVar[float]
    signal: Callable[[float], float] = lambda x: 0*x

    M: int = 0.1
    A: float = 6
    Q: InitVar[float] = 0.0001
    rho: float = 1000

    def __post_init__(self, omega_0: float, Q: float):
        self.sphere(omega_0, Q)

        beta = omega_0 / np.tan(omega_0 * self.dt / 2)

        a = np.array([1, 2 * (self.K - self.M * beta**2) / (self.M * beta**2 + self.R * beta + self.K), 1 - 2 * self.R * beta / (self.M * beta**2 + self.R * beta + self.K)])
        b = np.array([beta / (self.M * beta**2 + self.R * beta + self.K), 0, - beta / (self.M * beta**2 + self.R * beta + self.K)])
        self.u = signal.lfilter(b, a, self.signal(self.t))
        self.q = self.rho * self.A / (self.dx * self.dy * self.dz) * self.u
        self.psi = 1 / (2 * self.dt) * signal.lfilter([1, 0, -1], [1], self.q)

        self.f = interpolate.interp1d(self.t, self.psi)
        self.s = self.f(self.t)

    def sphere(self, omega_0, Q):
        self.R = omega_0 * self.M / Q
        self.K =  self.M * omega_0**2

    def __getitem__(self, key):
        return self.f(key)
    
    def __repr__(self):
        return f"Emitter at ({self.x}, {self.y})"


class Reciever(Module):
    def __repr__(self):
        return f"Reciever at ({self.x}, {self.y})"

    def __getitem__(self, key):
        if key in self.t:
            i = self.t.index(key)
            return self.s[i]

    def __setitem__(self, key, value):
        self.t.append(key)
        self.s.append(value)

def function(t, mu=5, sig = 1.5):
    omega_0 = 2 * np.pi * 50
    return np.sin(omega_0 * t) * np.exp(-np.power(t - mu, 2.) / (2 * np.power(sig, 2.)))

def function2(t, n):
    sig = np.ones((t.shape))
    sig[:n] = 0
    sig[t.size - n:] = 0
    filt = signal.windows.hann(n)
    return signal.convolve(sig, filt, mode='same') / np.sum(filt) * np.sin(2 * np.pi * 50 * t)


if __name__ == "__main__":
    # Pulsing sphere source
    dt = 0.001
    dx, dy, dz = 0.5, 0.5, 0.5
    T = np.arange(0, 10, dt)

    omega_0 = 2 * np.pi * 50

    e = Emitter(0, 0, T, dt, dx, dy, dz, omega_0, signal=function)

    fig, ax = plt.subplots()
    ax.plot(e.t, e.signal(e.t), label=r"$F(t)$", color="crimson")
    ax.legend(loc="best")
    ax.set_title("Force applied to the sphere", fontsize=14)
    ax.set_xlabel(r"Time ($s$)", fontsize=11)
    ax.set_ylabel(r"Force ($N$)", fontsize=11)
    ax.set_xlim(T[0], T[-1])
    ax.grid(True)
    fig.set_tight_layout(True)

    fig, ax = plt.subplots()
    ax.plot(e.t, e.q, label=r"$q_n$", color="purple")
    ax.plot(e.t, e.psi, label=r"$\psi_n$", color="teal")
    ax.legend(loc="best")
    ax.set_title("Pressure generated by a pulsing sphere", fontsize=14)
    ax.set_xlabel(r"Time ($s$)", fontsize=11)
    ax.set_ylabel(r"Pressure ($Pa$)", fontsize=11)
    ax.set_xlim(T[0], T[-1])
    ax.grid(True)
    fig.set_tight_layout(True)

    # Source
    fig, ax = e.temporal()
    fig, ax = e.fft()
    fig, ax = e.spectrogram()

    # Filtering
    sos = signal.butter(10, 20, 'lp', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, e.s)

    # Hydrophone
    r = Reciever(100, 200)
    for i, time in enumerate(T):
        r[time] = filtered[i]

    fig, ax = r.temporal()
    fig, ax = r.fft()
    fig, ax = r.spectrogram()
    plt.show()