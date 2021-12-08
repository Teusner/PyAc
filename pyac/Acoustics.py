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
from scipy import signal, interpolate, stats
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

    Attributes
    ----------
    t : (N,) array_like
        Time domain in seconds
    s : (N,) array_like
        Signal `s(t)`
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
        ax.plot(self.t, self(self.t))
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
        sp = np.fft.fft(self(self.t))
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
        f, t, Sxx = signal.spectrogram(np.array(self(self.t)), fs=30)
        ax.pcolormesh(t, f, Sxx, shading="auto")
        return fig, ax


@dataclass
class Emitter(Module):
    omega_0: float
    Q: float = 1
    M: int = 0.1
    A: float = 6
    f: Callable[[float], float] = lambda x: 0 * x

    def __post_init__(self):
        self.R = self.omega_0 * self.M / self.Q
        self.K =  self.M * self.omega_0**2

    def update(self, t, dt, rho, v):
        """Updating the Reciever with scene parameters
        
        Parameters
        ----------
        t : list or 1-D array
            Time domain
        dt : float
            Time step
        rho : float
            Medium density at Emitter position
        v : float
            Finite differences cell volume
        """
        beta = self.omega_0 / np.tan(self.omega_0 * dt / 2)
        a = np.array([1, 2 * (self.K - self.M * beta**2) / (self.M * beta**2 + self.R * beta + self.K), 1 - 2 * self.R * beta / (self.M * beta**2 + self.R * beta + self.K)])
        b = np.array([beta / (self.M * beta**2 + self.R * beta + self.K), 0, - beta / (self.M * beta**2 + self.R * beta + self.K)])
        self.u = signal.lfilter(b, a, self.f(t))
        self.q = rho * self.A / v * self.u
        self.t = t.tolist()
        self.s = (1 / (2 * dt) * signal.lfilter([1, 0, -1], [1], self.q)).tolist()

    def __getitem__(self, time):
        return self.f(time)

    def __call__(self, t):
        f = interpolate.interp1d(self.t, self.s, kind="linear")
        return f(t)
    
    def __repr__(self):
        return f"Emitter at ({self.x}, {self.y})"

@dataclass
class Reciever(Module):
    """An acoustic Reciever.

    Record an acoustic signal at his position.
    This module is filled up by a solver.

    Parameters
    ----------
    n : None or scipy.stats.rv*. Default: None.
        Specifies an aditional noise model to be added on
        the recorded signal. If not None, the `rvs()` mehtod
        will be used to generate the noise on the
        `scipy.stats.rv*` object.
    """

    n: None = field(default_factory=None)
    noise: list = field(init=False, default_factory=list)

    def __repr__(self):
        return f"Reciever at ({self.x}, {self.y})"

    def __getitem__(self, key):
        if key in self.t:
            i = self.t.index(key)
            return self.s[i]

    def __setitem__(self, key, value):
        self.t.append(key)
        self.s.append(value)
        if self.n is not None:
            self.noise.append(self.n.rvs(1)[0])
        else:
            self.noise.append(0)

    def __call__(self, t):
        """Get the recorded signal at t

        Parameters
        ----------
        t : float or 1-D array of floats
            Times of the wanted signal

        Returns
        -------
        s : float
            The recorded signal interpolated at t
            using a previous interpolation method.
        """
        f = interpolate.interp1d(self.t, np.asarray(self.s) + np.asarray(self.noise), kind="previous")
        return f(t)


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

    e = Emitter(0, 0, omega_0, signal=function)
    e.update(T, dt, 1000, 1)

    fig, ax = plt.subplots()
    ax.plot(T, e(T), label=r"$F(t)$", color="crimson")
    ax.legend(loc="best")
    ax.set_title("Force applied to the sphere", fontsize=14)
    ax.set_xlabel(r"Time ($s$)", fontsize=11)
    ax.set_ylabel(r"Force ($N$)", fontsize=11)
    ax.set_xlim(T[0], T[-1])
    ax.grid(True)
    fig.set_tight_layout(True)

    fig, ax = plt.subplots()
    ax.plot(T, e.q, label=r"$q_n$", color="purple")
    ax.plot(T, e(T), label=r"$\psi_n$", color="teal")
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
    n = stats.norm(0, 0.5)
    r = Reciever(100, 200, n)
    for i, time in enumerate(T):
        r[time] = filtered[i]

    fig, ax = r.temporal()
    fig, ax = r.fft()
    fig, ax = r.spectrogram()
    plt.show()