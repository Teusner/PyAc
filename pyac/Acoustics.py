from dataclasses import dataclass, field
from typing import Callable
from abc import ABC

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2 * np.pi))

def d_gaussian(x, mu, sig):
    return - (x - mu) / (np.power(sig, 3.) * np.sqrt(2 * np.pi)) * np.exp(- np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

@dataclass
class Module(ABC):
    x: float
    y: float
    t: list = field(init=False, default_factory=list)
    s: list = field(init=False, default_factory=list)

    def temporal(self):
        fig, ax = plt.subplots()
        ax.set_title("Signal of " + str(self))
        ax.set_xlabel(r"Time ($s$)")
        ax.set_ylabel(r"Pressure ($Pa$)")
        ax.grid(True)
        if self.t and self.s:
            ax.plot(self.t, self.s)
            ax.set_xlim(np.min(self.t), np.max(self.t))
        return fig, ax

    def fft(self):
        fig, ax = plt.subplots()
        ax.set_title(f"FFT of " + str(self))
        ax.set_xlabel(r"Frequency ($Hz$)")
        ax.set_ylabel(r"Magnitude ($Power$)")
        ax.grid(True)
        if self.t and self.s:
            freq = np.fft.fftfreq(len(self.t), self.t[1] - self.t[0])
            sp = np.fft.fft(self.s)
            ax.plot(freq, np.abs(sp.real))
            ax.set_xlim(np.min(freq), np.max(freq))
        return fig, ax

    def spectrogram(self):
        fig, ax = plt.subplots()
        ax.set_title(f"Spectrogram of " + str(self))
        ax.set_xlabel(r"Time ($s$)")
        ax.set_ylabel(r"Frequency ($Hz$)")
        ax.grid(True)
        if self.t and self.s:
            f, t, Sxx = signal.spectrogram(np.array(self.s), fs=30)
            ax.pcolormesh(t, f, Sxx, shading="auto")
            # ax.specgram(self.y, Fs=30)
        return fig, ax


@dataclass
class Emitter(Module):
    signal: Callable[[float], float]

    def __repr__(self):
        return f"Emitter at ({self.x}, {self.y})"

    def __getitem__(self, key):
        if key in self.t:
            i = self.t.index(key)
            return self.s[i]
        
        self.t.append(key)
        self.s.append((self.signal(key)).tolist())
        return self.s[-1]

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


if __name__ == "__main__":
    # Time sequence
    t = np.arange(0, 10, 0.001)

    # Source
    e = Emitter(100, 200, lambda x: signal.chirp(x, 1, 10, 100))
    for time in t:
        value = e[time]

    fig, ax = e.temporal()
    fig, ax = e.fft()
    fig, ax = e.spectrogram()

    # Filtering
    sos = signal.butter(10, 20, 'lp', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, e.s)

    # Hydrophone
    r = Reciever(100, 200)
    for i, time in enumerate(t):
        r[time] = filtered[i]

    fig, ax = r.temporal()
    fig, ax = r.fft()
    fig, ax = r.spectrogram()
    plt.show()