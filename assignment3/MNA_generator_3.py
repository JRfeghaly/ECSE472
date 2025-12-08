import math
import numpy as np
import matplotlib.pyplot as plt

def parse_time(token: str) -> float:
    """
    Parse a time string like '0.05s', '0.1ms', '10us', '1e-3'
    into seconds (float).
    """
    token = token.strip().lower()
    num_str = ''.join(ch for ch in token if ch.isdigit() or ch in (".", "+", "-"))
    suf = ''.join(ch for ch in token if not (ch.isdigit() or ch in (".", "+", "-")))

    if not num_str:
        raise ValueError(f"Invalid time value: {token}")

    value = float(num_str)
    multipliers = {"": 1.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if suf not in multipliers:
        raise ValueError(f"Unknown time suffix '{suf}' in {token}")
    
    return value * multipliers[suf]


def load_netlist(path):
    """
    Read the netlist file
    Returns:
        lines: list of element lines (without .tran / .end / comments)
        t_stop: transient stop time (seconds)
        dt:     time step (seconds)
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
      R, C, I, V (DC or COS), D (diode), E (VCVS)
    """
    comps = []
    for line in lines:
        tokens = line.split()
        name = tokens[0]
        ctype = name[0].upper()

        #VCVS
        if ctype == "E":
            if len(tokens) < 6:
                raise ValueError("VCVS line too short: " + line)
            n1, n2, nc1, nc2 = tokens[1], tokens[2], tokens[3], tokens[4]
            gain = float(tokens[5])
            comps.append({
                "type": "E",
                "name": name,
                "n1": n1, "n2": n2,
                "ctrl_p": nc1, "ctrl_n": nc2,
                "gain": gain,
            })
            continue

        if len(tokens) < 3:
            raise ValueError("Bad element line: " + line)

        n1, n2 = tokens[1], tokens[2]

        #voltage source
        if ctype == "V":
            # Vin n1 0 COS 1 60
            # V1  n1 0 DC 5
            # V1  n1 0 5
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

        # Resistor, capacitor, current source
        elif ctype in ("R", "C", "I"):
            if len(tokens) < 4:
                raise ValueError("R/C/I line too short: " + line)
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


def build_node_map(components):
    """
    Create:
      node_index: mapping node_name -> 1..N (0 is ground)
      vsrc_names: ordered list of voltage source / VCVS names
      N:          number of non-ground nodes
      M:          number of voltage sources
    """
    nodes = set()
    vsrc_names = []

    for c in components:
        t = c["type"].upper()
        if "n1" in c and c["n1"] != "0":
            nodes.add(c["n1"])
        if "n2" in c and c["n2"] != "0":
            nodes.add(c["n2"])

        if t == "E":
            if c["ctrl_p"] != "0":
                nodes.add(c["ctrl_p"])
            if c["ctrl_n"] != "0":
                nodes.add(c["ctrl_n"])
            vsrc_names.append(c["name"])
        elif t == "V":
            vsrc_names.append(c["name"])

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}  # 1-based
    N = len(nodes)
    M = len(vsrc_names)
    return node_index, vsrc_names, N, M


def get_index(node, node_map):
    """Return 0-based index for a node name, or None for ground (0)."""
    if node == "0":
        return None
    return node_map[node] - 1

#linear MNA matrices (G, C, G2, Ectrl)
def build_linear_matrices(components, node_map, vsrc_names, N, M):
    """
    Build time-independent matrices:
        G: NxN conductance
        C: NxN capacitance
        G2: NxM source-connection matrix
        Ectrl:MxN control matrix for VCVS
    Diodes and current sources do NOT get stamped here
    (they are handled in b(t) or nonlinear part).
    """
    G = np.zeros((N, N))
    C = np.zeros((N, N))
    G2 = np.zeros((N, M))
    Ectrl = np.zeros((M, N))

    for c in components:
        t = c["type"].upper()

        #VCVS
        if t == "E":
            n1 = get_index(c["n1"], node_map)
            n2 = get_index(c["n2"], node_map)
            cp = get_index(c["ctrl_p"], node_map)
            cn = get_index(c["ctrl_n"], node_map)
            k = vsrc_names.index(c["name"])

            if n1 is not None:
                G2[n1, k] = 1.0
            if n2 is not None:
                G2[n2, k] = -1.0

            gain = c["gain"]
            if cp is not None:
                Ectrl[k, cp] -= gain
            if cn is not None:
                Ectrl[k, cn] += gain

        #resistor
        elif t == "R":
            n1 = get_index(c["n1"], node_map)
            n2 = get_index(c["n2"], node_map)
            g = 1.0 / c["value"]
            if n1 is not None:
                G[n1, n1] += g
            if n2 is not None:
                G[n2, n2] += g
            if n1 is not None and n2 is not None:
                G[n1, n2] -= g
                G[n2, n1] -= g

        #capacitor
        elif t == "C":
            n1 = get_index(c["n1"], node_map)
            n2 = get_index(c["n2"], node_map)
            cv = c["value"]
            if n1 is not None:
                C[n1, n1] += cv
            if n2 is not None:
                C[n2, n2] += cv
            if n1 is not None and n2 is not None:
                C[n1, n2] -= cv
                C[n2, n1] -= cv

        #voltage source (just connectivity, no RHS)
        elif t == "V":
            n1 = get_index(c["n1"], node_map)
            n2 = get_index(c["n2"], node_map)
            k = vsrc_names.index(c["name"])
            if n1 is not None:
                G2[n1, k] = 1.0
            if n2 is not None:
                G2[n2, k] = -1.0

        #current sources and diodes handled elsewhere
        elif t in ("I", "D"):
            continue

        else:
            raise ValueError("Unsupported type in linear matrices: " + t)

    return G, C, G2, Ectrl

#time-dependent RHS vector b(t)
def build_b_time(components, node_map, vsrc_names, N, M, t):
    """
    Build b(t) for time t (size N+M x 1):
      voltage sources stamp values in the bottom block (source equations)
      current sources stamp into the top node block
    """
    b = np.zeros((N + M, 1))

    for c in components:
        tt = c["type"].upper()

        #voltage source values
        if tt == "V":
            k = vsrc_names.index(c["name"])
            if "waveform" in c and c["waveform"].upper() == "COS":
                val = c["ampl"] * math.cos(2 * math.pi * c["freq"] * t)
            else:
                val = c["value"]
            b[N + k, 0] = val

        #current source (if ever used)
        elif tt == "I":
            n1 = get_index(c["n1"], node_map)
            n2 = get_index(c["n2"], node_map)
            val = c["value"]
            if n1 is not None:
                b[n1, 0] -= val
            if n2 is not None:
                b[n2, 0] += val

    return b

#linear transient (backward euler)
def transient_linear(components, node_map, vsrc_names,
                     G, C, G2, Ectrl,
                     N, M, t_stop, dt):
    n_steps = int(round(t_stop / dt)) + 1
    times = np.linspace(0.0, t_stop, n_steps)
    x_hist = np.zeros((n_steps, N + M))

    v_prev = np.zeros((N, 1))

    for i, t in enumerate(times):
        b = build_b_time(components, node_map, vsrc_names, N, M, t)

        A_tl = G + C / dt
        top = np.hstack((A_tl, G2))
        bl = G2.T + Ectrl
        bottom = np.hstack((bl, np.zeros((M, M))))
        A = np.vstack((top, bottom))

        rhs = b.copy()
        rhs[:N, 0] += (C @ v_prev)[:, 0] / dt

        x = np.linalg.solve(A, rhs)
        x_hist[i, :] = x[:, 0]

        v_prev = x[:N].copy()

    return times, x_hist

#non-linear transient with diode (Newton–Raphson)
def transient_nonlinear(components, node_map, vsrc_names,
                        G0, C, G2, Ectrl,
                        N, M, t_stop, dt,
                        Is=1e-14, Vt=0.025,
                        max_iter=50, tol=1e-6):
    n_steps = int(round(t_stop / dt)) + 1
    times = np.linspace(0.0, t_stop, n_steps)
    x_hist = np.zeros((n_steps, N + M))

    x_prev = np.zeros((N + M, 1))

    for step, t in enumerate(times):
        v_prev = x_prev[:N].copy()
        x = x_prev.copy() #initial guess from previous time step

        for _ in range(max_iter):
            G = G0.copy()
            b = build_b_time(components, node_map, vsrc_names, N, M, t)

            #diode contribution (conductance + equivalent current)
            for c in components:
                if c["type"].upper() == "D":
                    n1 = get_index(c["n1"], node_map)
                    n2 = get_index(c["n2"], node_map)

                    Vd = 0.0
                    if n1 is not None:
                        Vd += x[n1, 0]
                    if n2 is not None:
                        Vd -= x[n2, 0]

                    Id = Is * (math.exp(Vd / Vt) - 1.0)
                    Gd = (Is / Vt) * math.exp(Vd / Vt)
                    Ieq = Id - Gd * Vd

                    if n1 is not None:
                        G[n1, n1] += Gd
                        b[n1, 0] -= Ieq
                    if n2 is not None:
                        G[n2, n2] += Gd
                        b[n2, 0] += Ieq
                    if n1 is not None and n2 is not None:
                        G[n1, n2] -= Gd
                        G[n2, n1] -= Gd

            #backward Euler for capacitors
            A_tl = G + C / dt
            top = np.hstack((A_tl, G2))
            bl = G2.T + Ectrl
            bottom = np.hstack((bl, np.zeros((M, M))))
            A = np.vstack((top, bottom))

            rhs = b.copy()
            rhs[:N, 0] += (C @ v_prev)[:, 0] / dt

            x_new = np.linalg.solve(A, rhs)

            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                x = x_new
                break

            x = x_new

        x_hist[step, :] = x[:, 0]
        x_prev = x.copy()

    return times, x_hist

def main():
    netlist_path = "circuit.txt"

    lines, t_stop, dt = load_netlist(netlist_path)
    if t_stop is None or dt is None:
        # Fallback if .tran is missing
        t_stop = 0.05
        dt = 1e-4

    components = parse_components(lines)
    node_index, vsrc_names, N, M = build_node_map(components)
    G0, C, G2, Ectrl = build_linear_matrices(components, node_index, vsrc_names, N, M)

    #Linear transient (for comparison)
    times_lin, x_lin = transient_linear(
        components, node_index, vsrc_names,
        G0, C, G2, Ectrl, N, M, t_stop, dt
    )

    #nonlinear transient with diode
    times_nl, x_nl = transient_nonlinear(
        components, node_index, vsrc_names,
        G0, C, G2, Ectrl, N, M, t_stop, dt
    )

    #get indices for nodes n1 and n2
    n1_idx = node_index["n1"] - 1
    n2_idx = node_index["n2"] - 1

    v_n1_nl = x_nl[:, n1_idx]
    v_n2_nl = x_nl[:, n2_idx]

    print("Nonlinear transient results at final time:")
    print(f"t = {times_nl[-1]:.6f} s")
    print(f"V(n1) = {v_n1_nl[-1]:.6f} V")
    print(f"V(n2) = {v_n2_nl[-1]:.6f} V")

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(times_nl, v_n1_nl, label="Voltage at node n1 (Vin)")
        plt.plot(times_nl, v_n2_nl, label="Voltage at node n2 (Vout)")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title("Half-wave rectifier – Nonlinear transient (Backward Euler)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()