import math
import numpy as np
import matplotlib.pyplot as plt

IS = 1e-14     # saturation current [A]
VT = 0.025     # thermal voltage [V]


def diode_I(v):
    """Diode current I(v) = Is (exp(v/Vt) - 1)."""
    return IS * (math.exp(v / VT) - 1.0)

def parse_time(token: str) -> float:
    """
    Parse a time string like '1ms', '0.1us', '10ns', '1e-3'
    into seconds (float).
    """
    token = token.strip().lower()
    num_str = ''.join(ch for ch in token if ch.isdigit() or ch in ('.', '+', '-'))
    suf = ''.join(ch for ch in token if not (ch.isdigit() or ch in ('.', '+', '-')))

    if not num_str:
        raise ValueError(f"Invalid time value: {token}")

    value = float(num_str)
    multipliers = {"": 1.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if suf not in multipliers:
        raise ValueError(f"Unknown time suffix '{suf}' in {token}")

    return value * multipliers[suf]


def load_netlist(path):
    """
    Read the netlist file.

    Returns:
        lines  : list of element lines (without .tran/.end/comments)
        t_stop : transient stop time (seconds) from .tran (may be None)
        dt     : time step (seconds) from .tran (may be None)
    """
    lines = []
    t_stop = None
    dt = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue

            low = line.lower()

            if low.startswith(".tran"):
                tokens = line.split()
                if len(tokens) >= 3:
                    t_stop = parse_time(tokens[1])
                    dt = parse_time(tokens[2])
                continue

            if low.startswith(".end"):
                break

            lines.append(line)

    return lines, t_stop, dt


def parse_components(lines):
    """
    Parse element lines into a list of component dictionaries.
    Supported:
      R, C, I (DC or COS), V (DC or COS), D
    """
    comps = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 3:
            raise ValueError("Bad element line: " + line)

        name = tokens[0]
        ctype = name[0].upper()
        n1, n2 = tokens[1], tokens[2]

        # Current source
        if ctype == "I":
            if len(tokens) >= 6 and tokens[3].upper() == "COS":
                ampl = float(tokens[4])
                freq = float(tokens[5])
                comps.append({
                    "type": "I",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "waveform": "COS",
                    "ampl": ampl,
                    "freq": freq,
                })
            elif len(tokens) >= 5 and tokens[3].upper() in ("DC", "AC"):
                val = float(tokens[4])
                comps.append({
                    "type": "I",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "value": val,
                })
            else:
                val = float(tokens[3])
                comps.append({
                    "type": "I",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "value": val,
                })

        # Voltage source
        elif ctype == "V":
            if len(tokens) >= 6 and tokens[3].upper() == "COS":
                ampl = float(tokens[4])
                freq = float(tokens[5])
                comps.append({
                    "type": "V",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "waveform": "COS",
                    "ampl": ampl,
                    "freq": freq,
                })
            elif len(tokens) >= 5 and tokens[3].upper() in ("DC", "AC"):
                val = float(tokens[4])
                comps.append({
                    "type": "V",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "value": val,
                })
            else:
                val = float(tokens[3])
                comps.append({
                    "type": "V",
                    "name": name,
                    "n1": n1, "n2": n2,
                    "value": val,
                })

        # Resistor / capacitor
        elif ctype in ("R", "C"):
            if len(tokens) < 4:
                raise ValueError("R/C line too short: " + line)
            val = float(tokens[3])
            comps.append({
                "type": ctype,
                "name": name,
                "n1": n1, "n2": n2,
                "value": val,
            })

        # Diode
        elif ctype == "D":
            comps.append({
                "type": "D",
                "name": name,
                "n1": n1, "n2": n2,
            })

        else:
            raise ValueError("Unsupported element type in line: " + line)

    return comps

def extract_current_test_params(components):
    """
    Extract parameters for the HB current-source test circuit:

        I1 n1 0 COS I0 f
        D1 n1 0
        R1 n1 0 R
        C1 n1 0 C
    """
    node = None
    I0 = None
    freq = None
    R = None
    C = None
    has_diode = False

    for c in components:
        if c["type"] == "I" and c.get("waveform", "").upper() == "COS":
            node = c["n1"]
            if c["n2"] != "0":
                raise ValueError("Expected current source to ground.")
            I0 = c["ampl"]
            freq = c["freq"]

    if node is None:
        raise ValueError("No cosine current source found.")

    for c in components:
        if c["type"] == "R" and c["n1"] == node and c["n2"] == "0":
            R = c["value"]
        if c["type"] == "C" and c["n1"] == node and c["n2"] == "0":
            C = c["value"]
        if c["type"] == "D" and ((c["n1"] == node and c["n2"] == "0") or
                                 (c["n2"] == node and c["n1"] == "0")):
            has_diode = True

    if R is None or C is None or not has_diode:
        raise ValueError("Missing R, C, or D to ground for the test circuit.")

    return node, I0, freq, R, C


def extract_half_wave_params(components):
    """
    Extract parameters for a half-wave rectifier:

        Vin n1 0 COS A f
        D1  n1 n2   (or n2 n1)
        R1  n2 0 R
        C1  n2 0 C
    """
    src_node = None
    ampl = None
    freq = None

    # Sinusoidal voltage source to ground
    for c in components:
        if c["type"] == "V" and c.get("waveform", "").upper() == "COS":
            src_node = c["n1"]
            if c["n2"] != "0":
                raise ValueError("Expected sinusoidal source to ground.")
            ampl = c["ampl"]
            freq = c["freq"]
            break

    if src_node is None:
        raise ValueError("No cosine voltage source found.")

    # Resistor from some node to ground (output node)
    out_node = None
    R = None
    for c in components:
        if c["type"] == "R" and c["n2"] == "0":
            out_node = c["n1"]
            R = c["value"]
            break
    if out_node is None:
        raise ValueError("No load resistor to ground found.")

    # Capacitor from same node to ground
    C = None
    for c in components:
        if c["type"] == "C" and c["n1"] == out_node and c["n2"] == "0":
            C = c["value"]
            break
    if C is None:
        raise ValueError("No capacitor to ground found at output node.")

    # Diode between source node and output node
    diode_from_src_to_out = None
    for c in components:
        if c["type"] == "D":
            nodes = {c["n1"], c["n2"]}
            if nodes == {src_node, out_node}:
                diode_from_src_to_out = (c["n1"] == src_node)
                break
    if diode_from_src_to_out is None:
        raise ValueError("No diode between source and output node found.")

    return src_node, out_node, ampl, freq, R, C, diode_from_src_to_out

def v_and_dvdt_from_coeffs(c, t, w):
    """
    Given Fourier coefficients c of size (2K+1), compute
    v(t) and dv/dt at time t.

    Indexing:
      c[0] = a0
      c[2k-1] = ak
      c[2k] = bk
    """
    K = (len(c) - 1) // 2
    v = c[0]
    dv_dt = 0.0

    for k in range(1, K + 1):
        ak = c[2 * k - 1]
        bk = c[2 * k]
        kwt = k * w * t
        v += ak * math.cos(kwt) + bk * math.sin(kwt)
        dv_dt += -ak * k * w * math.sin(kwt) + bk * k * w * math.cos(kwt)

    return v, dv_dt


def eval_periodic(c, times, w):
    """Evaluate v(t) from Fourier coefficients on an array of times."""
    v = np.zeros_like(times)
    for i, t in enumerate(times):
        v[i] = v_and_dvdt_from_coeffs(c, t, w)[0]
    return v

def harmonic_balance_current_test(I0, freq, R, C, num_harmonics=9, max_iter=40, tol=1e-10):
    """
    Solve periodic steady state of:
        I0 cos(ω t) = v/R + C dv/dt + I_D(v)
    """
    w = 2.0 * math.pi * freq
    K = num_harmonics
    N_unknowns = 2 * K + 1
    N_col = N_unknowns
    T = 1.0 / freq

    t_col = np.linspace(0.0, T, N_col, endpoint=False)

    c = np.zeros(N_unknowns)
    c[0] = 0.2  # small DC guess

    for _ in range(max_iter):
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
                v2, dvdt2 = v_and_dvdt_from_coeffs(c2, t, w)
                i_src = I0 * math.cos(w * t)
                Fp[i] = i_src - (v2 / R + C * dvdt2 + diode_I(v2))
            J[:, m] = (Fp - F) / eps

        delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        c += delta
        if np.linalg.norm(delta, ord=2) < tol:
            break

    return c

def harmonic_balance_half_wave(ampl, freq, R, C, diode_from_src_to_out=True, num_harmonics=9, max_iter=40, tol=1e-10):
    """
    Solve periodic steady state of node n2 in a half-wave rectifier:

        Vin -- D --> n2 -- R // C --> ground

    KCL at n2:
      v2/R + C dv2/dt - Id(Vin - v2) = 0   (for diode from src->out)
    """
    w = 2.0 * math.pi * freq
    K = num_harmonics
    N_unknowns = 2 * K + 1
    N_col = N_unknowns
    T = 1.0 / freq

    t_col = np.linspace(0.0, T, N_col, endpoint=False)

    c = np.zeros(N_unknowns)
    c[0] = 0.2

    for _ in range(max_iter):
        F = np.zeros(N_col)
        for i, t in enumerate(t_col):
            v2, dv2dt = v_and_dvdt_from_coeffs(c, t, w)
            v1 = ampl * math.cos(w * t)

            if diode_from_src_to_out:
                vd = v1 - v2
                Id = diode_I(vd)
                F[i] = v2 / R + C * dv2dt - Id
            else:
                vd = v2 - v1
                Id = diode_I(vd)
                F[i] = v2 / R + C * dv2dt + Id

        if np.linalg.norm(F, ord=2) < tol:
            break

        J = np.zeros((N_col, N_unknowns))
        eps = 1e-6
        for m in range(N_unknowns):
            c2 = c.copy()
            c2[m] += eps
            Fp = np.zeros(N_col)
            for i, t in enumerate(t_col):
                v2p, dv2dtp = v_and_dvdt_from_coeffs(c2, t, w)
                v1 = ampl * math.cos(w * t)

                if diode_from_src_to_out:
                    vdp = v1 - v2p
                    Idp = diode_I(vdp)
                    Fp[i] = v2p / R + C * dv2dtp - Idp
                else:
                    vdp = v2p - v1
                    Idp = diode_I(vdp)
                    Fp[i] = v2p / R + C * dv2dtp + Idp

            J[:, m] = (Fp - F) / eps

        delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        c += delta
        if np.linalg.norm(delta, ord=2) < tol:
            break

    return c


def backward_euler_current_test(I0, freq, R, C,
                                t_stop, dt,
                                max_iter=40,
                                tol=1e-9):
    """
    Transient solve using backward Euler for:

        C (v_k - v_{k-1})/dt + v_k/R + I_D(v_k) = I0 cos(ω t_k)
    """
    w = 2.0 * math.pi * freq
    n_steps = int(round(t_stop / dt)) + 1
    times = np.linspace(0.0, t_stop, n_steps)
    v = np.zeros(n_steps)

    for k in range(1, n_steps):
        t_k = times[k]
        v_prev = v[k - 1]
        v_k = v_prev

        for _ in range(max_iter):
            Id = diode_I(v_k)
            dId = (IS / VT) * math.exp(v_k / VT)
            F = C * (v_k - v_prev) / dt + v_k / R + Id - I0 * math.cos(w * t_k)
            dF = C / dt + 1.0 / R + dId
            dv = -F / dF
            v_k += dv
            if abs(dv) < tol:
                break

        v[k] = v_k

    return times, v

def backward_euler_half_wave(ampl, freq, R, C,
                             t_stop, dt,
                             diode_from_src_to_out=True,
                             max_iter=40,
                             tol=1e-9):
    """
    Transient solve for node n2 in the half-wave rectifier using BE.
    """
    w = 2.0 * math.pi * freq
    n_steps = int(round(t_stop / dt)) + 1
    times = np.linspace(0.0, t_stop, n_steps)
    v2 = np.zeros(n_steps)

    for k in range(1, n_steps):
        t_k = times[k]
        v_prev = v2[k - 1]
        v_k = v_prev

        v1 = ampl * math.cos(w * t_k)

        for _ in range(max_iter):
            if diode_from_src_to_out:
                vd = v1 - v_k
                Id = diode_I(vd)
                Gd = (IS / VT) * math.exp(vd / VT)
                # F = C (v_k - v_prev)/dt + v_k/R - Id = 0
                F = C * (v_k - v_prev) / dt + v_k / R - Id
                dF = C / dt + 1.0 / R + Gd
            else:
                vd = v_k - v1
                Id = diode_I(vd)
                Gd = (IS / VT) * math.exp(vd / VT)
                # F = C (v_k - v_prev)/dt + v_k/R + Id = 0
                F = C * (v_k - v_prev) / dt + v_k / R + Id
                dF = C / dt + 1.0 / R + Gd

            dv = -F / dF
            v_k += dv
            if abs(dv) < tol:
                break

        v2[k] = v_k

    return times, v2

def main():
    netlist_path = "circuit.txt"

    lines, t_stop, dt = load_netlist(netlist_path)
    components = parse_components(lines)

    # Try current-source topology first
    mode = None
    try:
        node_cs, I0, freq_cs, R_cs, C_cs = extract_current_test_params(components)
        mode = "current"
    except ValueError:
        mode = None

    if mode == "current":
        # ... your current-source code unchanged ...
        # (keep what you already have there)
        # ----------------------------------------
        print("Detected CURRENT-SOURCE test circuit.")
        node = node_cs
        freq = freq_cs
        R = R_cs
        C = C_cs

        if freq <= 0.0:
            raise ValueError("Frequency must be positive.")
        T = 1.0 / freq
        w = 2.0 * math.pi * freq

        if t_stop is None:
            t_tran_stop = 6.0 * T
        else:
            t_tran_stop = t_stop

        dt_tran = dt if dt is not None else T / 1000.0

        t_tran, v_tran = backward_euler_current_test(
            I0, freq, R, C, t_stop=t_tran_stop, dt=dt_tran
        )

        coeffs = harmonic_balance_current_test(I0, freq, R, C, num_harmonics=30)
        v_hb_full = eval_periodic(coeffs, t_tran, w)

        plt.figure(figsize=(8, 4))
        plt.plot(t_tran, v_hb_full, label=f"{node} (HB)")
        plt.plot(t_tran, v_tran, "--", label=f"{node} (BE)")
        plt.xlabel("Time [s]")
        plt.ylabel(f"Voltage at {node} [V]")
        plt.title("Current-source test: Harmonic Balance vs Backward Euler")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        src_node, out_node, ampl, freq, R, C, diode_dir = extract_half_wave_params(components)
        print("Detected HALF-WAVE RECTIFIER circuit.")
        print(f"  Vin node : {src_node}")
        print(f"  Vout node: {out_node}")

        if freq <= 0.0:
            raise ValueError("Frequency must be positive.")
        T = 1.0 / freq
        w = 2.0 * math.pi * freq

        if t_stop is None:
            t_tran_stop = 3.0 * T
        else:
            t_tran_stop = t_stop

        dt_tran = dt if dt is not None else T / 1000.0

        t_tran, v_out_tran = backward_euler_half_wave(
            ampl, freq, R, C,
            t_stop=t_tran_stop,
            dt=dt_tran,
            diode_from_src_to_out=diode_dir
        )
        v_src_tran = ampl * np.cos(w * t_tran)  # Vin (BE/analytic)

        coeffs = harmonic_balance_half_wave(
            ampl, freq, R, C,
            diode_from_src_to_out=diode_dir,
            num_harmonics=9
        )
        v_out_hb_full = eval_periodic(coeffs, t_tran, w)   # Vout (HB)
        v_src_hb_full = ampl * np.cos(w * t_tran)          # Vin (HB – same cosine)

        #FIGURE 1: Deliverable-style graph + HB FOR BOTH NODES
        plt.figure(figsize=(8, 4))
        # n1
        plt.plot(t_tran, v_src_tran, label="Voltage at n1 (BE)", color="b")
        plt.plot(t_tran, v_src_hb_full, "--", label="Voltage at n1 (HB)", color="c")
        # n2
        plt.plot(t_tran, v_out_tran, label="Voltage at n2 (BE)", color="k")
        plt.plot(t_tran, v_out_hb_full, "--", label="Voltage at n2 (HB)", color="r")

        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title("Node voltages of Half-wave rectifier using Transient Analysis")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()