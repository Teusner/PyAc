import acoustics as ac
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    R = []
    for i in range(3):
        file = open(f"output/Reciever_{i}.obj", "rb")
        R.append(pickle.load(file))
        file.close()

    for r in R:
        plt.figure()
        env = ac.signal.amplitude_envelope(r.s, r.t[1] - r.t[0])
        print(np.mean(env[-100:-50]))
        plt.plot(env)
    plt.show()