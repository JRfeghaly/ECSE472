import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

R = 1e3  # 1 kOhm
C = 1e-6  # 1 uF
f = 1e3  # 1 kHz
w = 2.0 * math.pi * f
I0 = 10e-3
Is = 1e-14
Vt = 0.025


def diode_I(v):
    return Is * (math.exp(v / Vt) - 1.0)


def diode_dIdv(v):
    return (Is / Vt) * math.exp(v / Vt)


def v_and_dvdt_from_coeffs(c, t, w):
    K = (len(c) - 1) // 2
    v = c[0]
    dvdt = 0.0

    for k in range(1, K + 1):
        ak = c[2 * k - 1]
        bk = c[2 * k]
        kwt = k * w * t
        v += ak * math.cos(kwt) + bk * math.sin(kwt)
        dvdt += -ak * k * w * math.sin(kwt) + bk * k * w * math.cos(kwt)

    return v, dvdt

def harmonic_balance(num_harmonics=5, max_iter=30, tol=1e-10):
    K = num_harmonics
    N_unknowns = 2 * K + 1
    N_col = N_unknowns

    T = 1.0 / f
    t_col = np.linspace(0.0, T, N_col, endpoint=False)

    Y = 1.0 / R + 1j * w * C
    V_ph = I0 / Y
    Vmag = abs(V_ph)
    phi = math.atan2(V_ph.imag, V_ph.real)

    a1 = Vmag * math.cos(phi)
    b1 = -Vmag * math.sin(phi)

    c = np.zeros(N_unknowns)
    c[0] = 0.0
    c[1] = a1
    c[2] = b1

    for it in range(max_iter):
        F = np.zeros(N_col)

        for i, t in enumerate(t_col):
            v, dvdt = v_and_dvdt_from_coeffs(c, t, w)
            i_src = I0 * math.cos(w * t)
            F[i] = i_src - (v / R + C * dvdt + diode_I(v))

        if np.linalg.norm(F, ord=2) < tol:
            break

        J = np.zeros((N_col, N_unknowns))
        eps = 1e-6

        for m in range(N_unknowns):
            c2 = c.copy()
            c2[m] += eps
            Fp = np.zeros(N_col)
            for i, t in enumerate(t_col):
                v, dvdt = v_and_dvdt_from_coeffs(c2, t, w)
                i_src = I0 * math.cos(w * t)
                Fp[i] = i_src - (v / R + C * dvdt + diode_I(v))
            J[:, m] = (Fp - F) / eps

        delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        c += delta

        if np.linalg.norm(delta, ord=2) < tol:
            break

    t_dense = np.linspace(0.0, T, 1000)
    v_dense = np.array([v_and_dvdt_from_coeffs(c, t, w)[0] for t in t_dense])

    return t_dense, v_dense, c

def backward_euler(num_periods=3, steps_per_period=1000):
    T = 1.0 / f
    h = T / steps_per_period
    total_time = num_periods * T
    N_steps = int(total_time / h) + 1

    times = np.linspace(0.0, total_time, N_steps)
    v = np.zeros(N_steps)

    for k in range(1, N_steps):
        t_k = times[k]
        v_prev = v[k - 1]
        v_k = v_prev

        for _ in range(40):
            Id = diode_I(v_k)
            dId = diode_dIdv(v_k)
            F = C * (v_k - v_prev) / h + v_k / R + Id - I0 * math.cos(w * t_k)
            dF = C / h + 1.0 / R + dId
            dv = -F / dF
            v_k += dv
            if abs(dv) < 1e-9:
                break

        v[k] = v_k

    return times, v

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def neural_pytorch(num_periods=3, steps_per_period=1000, epochs=3000, lr=1e-3):
    T = 1.0 / f

    # get BE waveform
    t_be, v_be = backward_euler(num_periods, steps_per_period)

    mask_last = t_be >= (num_periods - 1) * T
    t_train = t_be[mask_last]
    v_train = v_be[mask_last]

    # normalize time to [0, 1]
    x = torch.tensor(((t_train - t_train[0]) / (t_train[-1] - t_train[0])).reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(v_train.reshape(-1, 1), dtype=torch.float32)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for e in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # uncomment to watch training
        # if e % 500 == 0:
        #     print("epoch", e, "loss", loss.item())

    # evaluate over one period
    t_nn = np.linspace(0.0, T, 1000)
    x_nn = torch.tensor((t_nn / T).reshape(-1, 1), dtype=torch.float32)
    v_nn = model(x_nn).detach().numpy()

    return t_nn, v_nn


def main():
    t_hb, v_hb, coeffs = harmonic_balance(num_harmonics=5)
    print("HB Fourier coeffs:")
    print(coeffs)

    t_be, v_be = backward_euler(num_periods=3, steps_per_period=1000)

    T = 1.0 / f
    mask = t_be >= 2 * T
    t_be_last = t_be[mask] - 2 * T
    v_be_last = v_be[mask]

    # PyTorch NN
    t_nn, v_nn = neural_pytorch()

    # Plot 1
    plt.figure()
    plt.plot(t_hb * 1e3, v_hb, label="HB")
    plt.plot(t_be_last * 1e3, v_be_last, '--', label="BE last period")
    plt.plot(t_nn * 1e3, v_nn, ':', label="PyTorch NN")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [V]")
    plt.title("HB vs Backward Euler vs Neural (PyTorch)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()