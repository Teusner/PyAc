import numpy as np
from dataclasses import dataclass, field


@dataclass
class Material:
    rho: float
    cp: float
    Qp: float
    cs: float = field(default=0)
    Qs: float = field(default=1000)

    def __post_init__(self):
        self.eta = self.rho * self.cp**2
        self.mu = self.rho * self.cs**2

water = Material(1000, 1500, 1000)
basalt = Material(2700, 5250, 1000, 2500, 500)
sand = Material(1900, 1650, 200, 500, 40)
sediment = Material(1700, 1800, 50)
air = Material(1, 340, 1)

if __name__ == "__main__":
    print(water, basalt, sand)
    print(f"{water.eta}, {water.mu}")