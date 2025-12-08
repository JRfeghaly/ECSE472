import math
import numpy as np
import matplotlib.pyplot as plt

IS = 1e-14
VT = 0.025

def diode_I(v: float) -> float:
    """
    Diode I–V law:
        I_D(v) = IS * (exp(v / VT) - 1)
    where v is the diode voltage (anode – cathode).
    """
    return IS * (math.exp(v / VT) - 1.0)

def parse_time(token: str) -> float:
    """
    Parse a time string like:
        "1ms", "0.1us", "10ns", "1e-3"
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


def load_netlist(path: str):
    """
    Read the netlist file and return:

        lines  : list of element lines (no .tran/.end/comments)
        t_stop : transient stop time (seconds) from .tran (or None)
        dt     : time step (seconds) from .tran (or None)

    Netlist format (simplified):
        * comment lines
        Rname n1 n2 value
        Cname n1 n2 value
        Iname n1 n2 [DC val] | [COS ampl freq] | val
        Vname n1 n2 [DC val] | [COS ampl freq] | val
        Dname n1 n2
        .tran t_stop dt
        .end
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
    Supported elements:

      - Resistors: R name n1 n2 value
      - Capacitors: C name n1 n2 value
      - Diodes: D name n1 n2
      - Current sources: I name n1 n2 [DC/AC val] | [COS ampl freq] | val
      - Voltage sources: V name n1 n2 [DC/AC val] | [COS ampl freq] | val

    Node names are strings; ground is "0".
    Returns a list of dicts with keys:
      "type", "name", "n1", "n2", and various fields depending on type.
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
    Extract parameters for the *special* 1-node current-source test circuit:
        I1 n1 0 COS I0 f
        D1 n1 0
        R1 n1 0 R
        C1 n1 0 C
    Returns: (node_name, I0, freq, R, C)
    or raises ValueError if this topology is not detected.
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
                raise ValueError("Expected current source to ground for special test.")
            I0 = c["ampl"]
            freq = c["freq"]

    if node is None:
        raise ValueError("No cosine current source found for special test.")

    for c in components:
        if c["type"] == "R" and c["n1"] == node and c["n2"] == "0":
            R = c["value"]
        if c["type"] == "C" and c["n1"] == node and c["n2"] == "0":
            C = c["value"]
        if c["type"] == "D" and ((c["n1"] == node and c["n2"] == "0") or
                                 (c["n2"] == node and c["n1"] == "0")):
            has_diode = True

    if R is None or C is None or not has_diode:
        raise ValueError("Missing R, C, or D to ground for the special test circuit.")

    return node, I0, freq, R, C


def extract_half_wave_params(components):
    """
    Extract parameters for a half-wave rectifier:
        Vin n1 0 COS A f
        D1  n1 n2   (or n2 n1)
        R1  n2 0 R
        C1  n2 0 C
    Returns:
        (src_node, out_node, ampl, freq, R, C, diode_from_src_to_out)
    or raises ValueError if this topology is not detected.
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

    Indexing convention for a single node:
      c[0]       = a0
      c[2k - 1]  = ak
      c[2k]      = bk
    for k = 1..K.
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
    """
    Evaluate v(t) from Fourier coefficients c on an array of times.
    """
    v = np.zeros_like(times)
    for i, t in enumerate(times):
        v[i] = v_and_dvdt_from_coeffs(c, t, w)[0]
    return v

# 1-node HB / BE solvers (your original special cases)


def harmonic_balance_current_test(I0, freq, R, C, num_harmonics=9, max_iter=40, tol=1e-10):
    """
    Harmonic balance for the 1-node current source test:

        I0 cos(ω t) = v/R + C dv/dt + I_D(v)
    """
    w = 2.0 * math.pi * freq
    K = num_harmonics
    N_unknowns = 2 * K + 1
    N_col = N_unknowns
    T = 1.0 / freq

    # Collocation times over one period
    t_col = np.linspace(0.0, T, N_col, endpoint=False)

    # Initial guess: small DC value
    c = np.zeros(N_unknowns)
    c[0] = 0.2

    for _ in range(max_iter):
        F = np.zeros(N_col)
        for i, t in enumerate(t_col):
            v, dvdt = v_and_dvdt_from_coeffs(c, t, w)
            i_src = I0 * math.cos(w * t)
            F[i] = i_src - (v / R + C * dvdt + diode_I(v))

        if np.linalg.norm(F, ord=2) < tol:
            break

        # Numerical Jacobian wrt Fourier coefficients
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


def harmonic_balance_half_wave(ampl, freq, R, C, diode_from_src_to_out=True,
                               num_harmonics=9, max_iter=40, tol=1e-10):
    """
    Harmonic balance for node v2 in the half-wave rectifier:

        Vin -- D --> v2 -- R // C --> ground

    KCL at v2:
      v2/R + C dv2/dt - I_D(Vin - v2) = 0   (if diode from src->out)
      v2/R + C dv2/dt + I_D(v2 - Vin) = 0   (if diode from out->src)
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
    Transient solve (Backward Euler) for the 1-node current test:

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
    Transient (Backward Euler) for node v2 in the half-wave rectifier.
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

# General multi-node BACKWARD EULER (I/R/C/D only)

def build_node_mapping(components):
    """
    Collect all non-ground node names and build an index mapping.

    Returns:
      nodes      : sorted list of node names (strings, excluding "0")
      node_index : dict: node_name -> index 0..N-1
    """
    nodes = set()
    for c in components:
        for key in ("n1", "n2"):
            n = c[key]
            if n != "0":
                nodes.add(n)
    nodes = sorted(nodes)
    node_index = {name: i for i, name in enumerate(nodes)}
    return nodes, node_index


def classify_components_for_nodal(components, node_index):
    """
    Split components into lists usable by the general nodal BE/HB solvers.

    Each list element encodes nodes as integers:
      -1 means ground, 0..N-1 are node indices from node_index.
    """
    resistors = []
    capacitors = []
    diodes = []
    currents = []
    other = []

    for c in components:
        ctype = c["type"]
        if ctype == "R":
            i = node_index.get(c["n1"], -1)
            j = node_index.get(c["n2"], -1)
            resistors.append((i, j, c["value"]))
        elif ctype == "C":
            i = node_index.get(c["n1"], -1)
            j = node_index.get(c["n2"], -1)
            capacitors.append((i, j, c["value"]))
        elif ctype == "D":
            an = node_index.get(c["n1"], -1)
            ca = node_index.get(c["n2"], -1)
            diodes.append((an, ca))
        elif ctype == "I":
            i = node_index.get(c["n1"], -1)
            j = node_index.get(c["n2"], -1)
            currents.append((i, j, c))
        else:
            other.append(c)

    return resistors, capacitors, diodes, currents, other


def current_source_value(comp, t: float) -> float:
    """
    Evaluate a current source at time t.
    Supports:
      - COS amplitude/frequency
      - DC (constant value)
    """
    if comp.get("waveform", "").upper() == "COS":
        ampl = comp["ampl"]
        freq = comp["freq"]
        return ampl * math.cos(2.0 * math.pi * freq * t)
    else:
        # DC or unspecified treated as constant
        return comp.get("value", 0.0)


def residual_BE_general(v, v_prev, t, dt,
                        resistors, capacitors, diodes, currents, Nn):
    """
    KCL residual at each node (multi-node Backward Euler).

    Convention:
      F_i = sum(passive currents leaving node i) - sum(current injections at i)
      We want F_i = 0 for all i.

    v, v_prev: arrays of node voltages [V] (size Nn).
    """
    F = np.zeros(Nn)

    # Resistors: I = (Vi - Vj)/R
    for (i, j, R) in resistors:
        vi = v[i] if i >= 0 else 0.0
        vj = v[j] if j >= 0 else 0.0
        I = (vi - vj) / R
        if i >= 0:
            F[i] += I
        if j >= 0:
            F[j] -= I

    # Capacitors (Backward Euler):
    #   I = C * ((Vi - Vj) - (Vi_prev - Vj_prev))/dt
    for (i, j, Cval) in capacitors:
        vi = v[i] if i >= 0 else 0.0
        vj = v[j] if j >= 0 else 0.0
        vi_prev = v_prev[i] if i >= 0 else 0.0
        vj_prev = v_prev[j] if j >= 0 else 0.0
        I = Cval * ((vi - vj) - (vi_prev - vj_prev)) / dt
        if i >= 0:
            F[i] += I
        if j >= 0:
            F[j] -= I

    # Diodes: I = I_D(Va - Vc)
    for (an, ca) in diodes:
        va = v[an] if an >= 0 else 0.0
        vc = v[ca] if ca >= 0 else 0.0
        vd = va - vc
        Id = diode_I(vd)
        if an >= 0:
            F[an] += Id
        if ca >= 0:
            F[ca] -= Id

    # Current sources: injection:
    # For a source between i and j, we take +I into node i, -I into node j.
    for (i, j, comp) in currents:
        Iinj = current_source_value(comp, t)
        if i >= 0:
            F[i] -= Iinj
        if j >= 0:
            F[j] += Iinj

    return F


def jacobian_BE_general(v, v_prev, t, dt,
                        resistors, capacitors, diodes, currents, Nn):
    """
    Jacobian matrix dF/dv for the general Backward Euler system.
    """
    J = np.zeros((Nn, Nn))

    # Resistors: g = 1/R
    for (i, j, R) in resistors:
        g = 1.0 / R
        if i >= 0:
            J[i, i] += g
        if j >= 0:
            J[j, j] += g
        if i >= 0 and j >= 0:
            J[i, j] -= g
            J[j, i] -= g

    # Capacitors: g = C/dt, same pattern as resistor
    for (i, j, Cval) in capacitors:
        g = Cval / dt
        if i >= 0:
            J[i, i] += g
        if j >= 0:
            J[j, j] += g
        if i >= 0 and j >= 0:
            J[i, j] -= g
            J[j, i] -= g

    # Diodes: Gd = dI/dV
    for (an, ca) in diodes:
        va = v[an] if an >= 0 else 0.0
        vc = v[ca] if ca >= 0 else 0.0
        vd = va - vc
        Gd = (IS / VT) * math.exp(vd / VT)
        if an >= 0:
            J[an, an] += Gd
        if ca >= 0:
            J[ca, ca] += Gd
        if an >= 0 and ca >= 0:
            J[an, ca] -= Gd
            J[ca, an] -= Gd

    # Current sources do not depend on v -> no contribution

    return J


def backward_euler_general_current(components, t_stop, dt,
                                   max_iter=40, tol=1e-9):
    """
    General multi-node Backward Euler solver for I/R/C/D circuits
    (no ideal voltage sources).

    Returns:
      times : (N_t,) array of time samples
      v_all : (N_t, N_nodes) array, each column is a node voltage
      nodes : list of node names in the same order as v_all columns
    """
    nodes, node_index = build_node_mapping(components)
    Nn = len(nodes)

    resistors, capacitors, diodes, currents, other = \
        classify_components_for_nodal(components, node_index)

    if other:
        raise ValueError("General BE solver supports only I, R, C, D elements.")

    if Nn == 0:
        n_steps = int(round(t_stop / dt)) + 1
        times = np.linspace(0.0, t_stop, n_steps)
        return times, np.zeros((n_steps, 0)), nodes

    n_steps = int(round(t_stop / dt)) + 1
    times = np.linspace(0.0, t_stop, n_steps)
    v = np.zeros((n_steps, Nn))

    for k in range(1, n_steps):
        t_k = times[k]
        v_prev = v[k - 1, :].copy()
        v_k = v_prev.copy()

        for _ in range(max_iter):
            F = residual_BE_general(v_k, v_prev, t_k, dt,
                                    resistors, capacitors, diodes, currents, Nn)
            if np.linalg.norm(F, ord=2) < tol:
                break

            J = jacobian_BE_general(v_k, v_prev, t_k, dt,
                                    resistors, capacitors, diodes, currents, Nn)

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J, -F, rcond=None)

            v_k += delta
            if np.linalg.norm(delta, ord=2) < tol:
                break

        v[k, :] = v_k

    return times, v, nodes

# General multi-node HARMONIC BALANCE (I/R/C/D only)


def eval_all_v_and_dvdt(c, t, w, Nn, K):
    """
    Evaluate v_j(t) and dv_j/dt for all nodes from the global coefficient vector c.

    Packing:
      For each node n, we store 2K+1 coeffs:
        [a0, a1, b1, a2, b2, ..., aK, bK]
      consecutively in c.
    """
    Ns = 2 * K + 1
    v = np.zeros(Nn)
    dvdt = np.zeros(Nn)

    for n in range(Nn):
        c_node = c[n * Ns:(n + 1) * Ns]
        vn, dvn = v_and_dvdt_from_coeffs(c_node, t, w)
        v[n] = vn
        dvdt[n] = dvn

    return v, dvdt


def residual_HB_general(c, t_col, w,
                        resistors, capacitors, diodes, currents,
                        Nn, K):
    """
    Collocation-based HB residual for all nodes and all collocation points.

    We build one KCL equation per node per collocation time.
    """
    N_col = len(t_col)
    F = np.zeros(Nn * N_col)

    for ti, t in enumerate(t_col):
        v, dvdt = eval_all_v_and_dvdt(c, t, w, Nn, K)
        Fnodes = np.zeros(Nn)

        # Resistors
        for (i, j, R) in resistors:
            vi = v[i] if i >= 0 else 0.0
            vj = v[j] if j >= 0 else 0.0
            I = (vi - vj) / R
            if i >= 0:
                Fnodes[i] += I
            if j >= 0:
                Fnodes[j] -= I

        # Capacitors
        for (i, j, Cval) in capacitors:
            dvi = dvdt[i] if i >= 0 else 0.0
            dvj = dvdt[j] if j >= 0 else 0.0
            I = Cval * (dvi - dvj)
            if i >= 0:
                Fnodes[i] += I
            if j >= 0:
                Fnodes[j] -= I

        # Diodes
        for (an, ca) in diodes:
            va = v[an] if an >= 0 else 0.0
            vc = v[ca] if ca >= 0 else 0.0
            vd = va - vc
            Id = diode_I(vd)
            if an >= 0:
                Fnodes[an] += Id
            if ca >= 0:
                Fnodes[ca] -= Id

        # Current sources
        for (i, j, comp) in currents:
            Iinj = current_source_value(comp, t)
            if i >= 0:
                Fnodes[i] -= Iinj
            if j >= 0:
                Fnodes[j] += Iinj

        F[ti * Nn:(ti + 1) * Nn] = Fnodes

    return F


def jacobian_HB_general(c, t_col, w,
                        resistors, capacitors, diodes, currents,
                        Nn, K):
    """
    Numerical Jacobian dF/dc for the HB system (finite differences).
    """
    N_col = len(t_col)
    Ns = 2 * K + 1
    N_unknowns = Nn * Ns

    F = residual_HB_general(c, t_col, w,
                            resistors, capacitors, diodes, currents,
                            Nn, K)
    J = np.zeros((Nn * N_col, N_unknowns))
    eps = 1e-6

    for m in range(N_unknowns):
        c2 = c.copy()
        c2[m] += eps
        Fp = residual_HB_general(c2, t_col, w,
                                 resistors, capacitors, diodes, currents,
                                 Nn, K)
        J[:, m] = (Fp - F) / eps

    return J


def harmonic_balance_general_current(components, freq,
                                     num_harmonics=9,
                                     max_iter=40,
                                     tol=1e-10):
    """
    General multi-node harmonic balance for I/R/C/D circuits driven
    by cosine current source(s) at a single fundamental frequency.

    Returns:
      c      : global coefficient vector
      nodes  : list of node names
      K      : number of harmonics used
    """
    nodes, node_index = build_node_mapping(components)
    Nn = len(nodes)

    resistors, capacitors, diodes, currents, other = \
        classify_components_for_nodal(components, node_index)

    if other:
        raise ValueError("General HB solver supports only I, R, C, D elements.")

    if Nn == 0:
        return np.zeros(0), nodes, num_harmonics

    K = num_harmonics
    Ns = 2 * K + 1
    N_unknowns = Nn * Ns
    N_col = Ns

    T = 1.0 / freq
    w = 2.0 * math.pi * freq
    t_col = np.linspace(0.0, T, N_col, endpoint=False)

    # Initial guess: small DC offset per node
    c = np.zeros(N_unknowns)
    for n in range(Nn):
        c[n * Ns] = 0.2

    for _ in range(max_iter):
        F = residual_HB_general(c, t_col, w,
                                resistors, capacitors, diodes, currents,
                                Nn, K)
        if np.linalg.norm(F, ord=2) < tol:
            break

        J = jacobian_HB_general(c, t_col, w,
                                resistors, capacitors, diodes, currents,
                                Nn, K)
        delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        c += delta

        if np.linalg.norm(delta, ord=2) < tol:
            break

    return c, nodes, K


def eval_all_nodes_from_coeffs(c, times, freq, Nn, K):
    """
    Evaluate all node voltages for all times from the global HB coefficients.

    Returns:
      v_all : array (len(times), Nn)
    """
    w = 2.0 * math.pi * freq
    v_all = np.zeros((len(times), Nn))

    for ti, t in enumerate(times):
        v, _ = eval_all_v_and_dvdt(c, t, w, Nn, K)
        v_all[ti, :] = v

    return v_all

def compute_resistive_equiv_and_factors(components, root_node):
    """
    For the half-wave rectifier:
      - Find all resistors connected (by resistor-only paths) to `root_node`
      - Compute an equivalent R_eq to ground as seen from `root_node`
      - Compute, for each resistor node k in that subnetwork, a factor alpha_k
        such that v(k)(t) = alpha_k * v(root_node)(t) for any waveform v(root_node).

    Assumptions:
      - Only resistors in this subnetwork (no capacitors/diodes on those extra nodes)
      - At least one resistive path from `root_node` to ground '0'.
    """
    # Collect all resistors as (n1, n2, R)
    resistors = []
    for c in components:
        if c["type"] == "R":
            resistors.append((c["n1"], c["n2"], c["value"]))
    if not resistors:
        raise ValueError("No resistors found when computing R_eq.")

    # Build adjacency by resistor only (include ground '0' in the graph)
    adj = {}
    for n1, n2, R in resistors:
        adj.setdefault(n1, []).append((n2, R))
        adj.setdefault(n2, []).append((n1, R))

    # BFS from root_node to find the resistor-only connected component
    visited = set()
    queue = [root_node]
    visited.add(root_node)
    while queue:
        u = queue.pop(0)
        for v, _R in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Subnetwork nodes reachable from root via resistors
    sub_nodes = visited  # may include '0'
    if "0" not in sub_nodes:
        raise ValueError(f"No resistive path from node {root_node} to ground.")

    # Collect resistors within this subnetwork
    sub_resistors = [
        (n1, n2, R) for (n1, n2, R) in resistors
        if n1 in sub_nodes and n2 in sub_nodes
    ]

    # Unknown resistor nodes = all sub_nodes except root and ground
    unknown_nodes = sorted(n for n in sub_nodes if n not in (root_node, "0"))
    M = len(unknown_nodes)

    node_factors = {}

    # Special case: only root and ground in the resistor network
    if M == 0:
        G_eq = 0.0
        for n1, n2, R in sub_resistors:
            if {n1, n2} == {root_node, "0"}:
                G_eq += 1.0 / R
        if G_eq <= 0.0:
            raise ValueError(f"No resistors directly from {root_node} to ground.")
        R_eq = 1.0 / G_eq
        node_factors[root_node] = 1.0
        return R_eq, node_factors

    # General case: build nodal equations for unknown resistor nodes, with v(root)=1, v(0)=0
    idx = {n: i for i, n in enumerate(unknown_nodes)}
    A = np.zeros((M, M))
    b = np.zeros(M)
    V_root = 1.0

    for n1, n2, R in sub_resistors:
        g = 1.0 / R
        # Contribution to n1's equation
        for u, v in ((n1, n2), (n2, n1)):
            if u in idx:
                ui = idx[u]
                A[ui, ui] += g
                if v in idx:
                    A[ui, idx[v]] -= g
                elif v == root_node:
                    b[ui] += g * V_root
                elif v == "0":
                    # + g * 0 => no change
                    pass

    # Solve A x = b for unknown node voltages with v(root)=1, v(0)=0
    x = np.linalg.solve(A, b)

    # Node potentials in the resistor subnetwork (relative to v(root)=1)
    node_pot = {root_node: V_root}
    for n, i in idx.items():
        node_pot[n] = x[i]

    # Compute equivalent conductance seen from root into the resistor network
    I_root = 0.0
    for n1, n2, R in sub_resistors:
        if n1 == root_node or n2 == root_node:
            other = n2 if n1 == root_node else n1
            V_other = node_pot.get(other, 0.0 if other == "0" else 0.0)
            I_root += (V_root - V_other) / R

    G_eq = I_root
    if G_eq <= 0.0:
        raise ValueError("Equivalent conductance is non-positive.")
    R_eq = 1.0 / G_eq

    # Factors: v(k) = alpha_k * v(root) for each resistor node k
    for n, Vn in node_pot.items():
        node_factors[n] = Vn

    return R_eq, node_factors

def find_cos_current_freq(components) -> float:
    """
    Find the fundamental frequency from COS current sources.
    Require that all such sources share the same frequency.
    """
    freq = None
    for c in components:
        if c["type"] == "I" and c.get("waveform", "").upper() == "COS":
            if freq is None:
                freq = c["freq"]
            elif abs(freq - c["freq"]) > 1e-12:
                raise ValueError(
                    "Multiple cosine current sources with different frequencies are not supported."
                )

    if freq is None:
        raise ValueError("No cosine current source (I ... COS A f) found for general HB.")
    return freq

# MAIN: detect mode and plot HB vs BE for every voltage


def main():
    """
    Main entry point.

    - Reads "circuit.txt"
    - Tries to detect:
        1. Special current-source test circuit (only 1 non-ground node)
        2. Half-wave rectifier
        3. General current-driven circuit (I/R/C/D)
    - Runs Backward Euler transient + Harmonic Balance
    - Plots node voltages: HB vs BE
    """
    netlist_path = "circuit.txt"

    lines, t_stop, dt = load_netlist(netlist_path)
    components = parse_components(lines)

    #Build node list once (for deciding which mode to use)
    nodes_all, _ = build_node_mapping(components)

    # Try special current-source topology (1-node)
    special_current_ok = False
    try:
        node_cs, I0, freq_cs, R_cs, C_cs = extract_current_test_params(components)
        # Only accept "special" mode if there is EXACTLY ONE non-ground node
        if len(nodes_all) == 1:
            special_current_ok = True
    except ValueError:
        special_current_ok = False

    # Try special half-wave rectifier
    half_wave_ok = False
    if not special_current_ok:
        try:
            src_node, out_node, ampl_hw, freq_hw, R_hw, C_hw, diode_dir_hw = \
                extract_half_wave_params(components)
            half_wave_ok = True
        except ValueError:
            half_wave_ok = False

    # ------------------ MODE 1: 1-node current-source test ------------------
    if special_current_ok:
        print("Detected CURRENT-SOURCE TEST circuit.")
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

        # Transient (Backward Euler)
        t_tran, v_tran = backward_euler_current_test(
            I0, freq, R, C, t_stop=t_tran_stop, dt=dt_tran
        )

        # Harmonic balance
        coeffs = harmonic_balance_current_test(
            I0, freq, R, C, num_harmonics=30
        )
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

    # ------------------ MODE 2: Half-wave rectifier ------------------
    elif half_wave_ok:
        print("Detected HALF-WAVE RECTIFIER circuit.")
        print(f"  Vin node : {src_node}")
        print(f"  Vout node: {out_node}")

        freq = freq_hw
        ampl = ampl_hw
        C = C_hw
        diode_dir = diode_dir_hw

        if freq <= 0.0:
            raise ValueError("Frequency must be positive.")
        T = 1.0 / freq
        w = 2.0 * math.pi * freq

        # --- Collapse all resistors connected to out_node into an equivalent R_eq ---
        try:
            R_eq, node_factors = compute_resistive_equiv_and_factors(components, out_node)
            print(f"  Found resistive subnetwork at node {out_node}:")
            print(f"    Equivalent R_eq = {R_eq:.3g} Ω to ground")
            for n, a in sorted(node_factors.items()):
                print(f"    v({n}) = {a:.3f} * v({out_node})")
        except ValueError as e:
            # Fall back to the single R from extract_half_wave_params
            print(f"  [Warning] {e}")
            print("  Falling back to single load resistor R only.")
            R_eq = R_hw
            node_factors = {out_node: 1.0}

        if t_stop is None:
            t_tran_stop = 3.0 * T
        else:
            t_tran_stop = t_stop

        dt_tran = dt if dt is not None else T / 1000.0

        # --- Transient (Backward Euler) for the dynamic node v(out_node) ---
        t_tran, v_root_tran = backward_euler_half_wave(
            ampl, freq, R_eq, C,
            t_stop=t_tran_stop,
            dt=dt_tran,
            diode_from_src_to_out=diode_dir
        )

        # --- Harmonic Balance for v(out_node) ---
        coeffs = harmonic_balance_half_wave(
            ampl, freq, R_eq, C,
            diode_from_src_to_out=diode_dir,
            num_harmonics=25
        )
        v_root_hb_full = eval_periodic(coeffs, t_tran, w)

        # --- Source node (Vin) is known analytically ---
        v_src_tran = ampl * np.cos(w * t_tran)
        v_src_hb_full = v_src_tran.copy()

        # --- Build voltages for ALL resistor nodes: v(k) = alpha_k * v(out_node) ---
        # node_factors contains at least {out_node: 1.0}
        node_names_res = sorted(node_factors.keys())

        plt.figure(figsize=(8, 4))

        # Plot source node (Vin)
        plt.plot(t_tran, v_src_tran, label=f"v({src_node}) (BE)")
        plt.plot(t_tran, v_src_hb_full, "--", label=f"v({src_node}) (HB)")

        # Plot each resistor node (including out_node) using its factor alpha_k
        for name in node_names_res:
            alpha = node_factors[name]
            v_tran_node = alpha * v_root_tran
            v_hb_node = alpha * v_root_hb_full
            plt.plot(t_tran, v_tran_node, label=f"v({name}) (BE)")
            plt.plot(t_tran, v_hb_node, "--", label=f"v({name}) (HB)")

        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title("Half-wave rectifier: node voltages (HB vs BE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # ------------------ MODE 3: General I/R/C/D circuit ------------------
    else:
        print("Detected GENERAL CURRENT-DRIVEN CIRCUIT.")
        freq = find_cos_current_freq(components)
        print(f"  Fundamental frequency: {freq} Hz")

        if freq <= 0.0:
            raise ValueError("Frequency must be positive.")
        T = 1.0 / freq

        if t_stop is None:
            t_tran_stop = 3.0 * T
        else:
            t_tran_stop = t_stop

        dt_tran = dt if dt is not None else T / 1000.0

        # Transient (Backward Euler) for all nodes
        t_tran, v_tran_all, nodes_order = backward_euler_general_current(
            components,
            t_stop=t_tran_stop,
            dt=dt_tran
        )

        # Harmonic Balance for all nodes
        coeffs, nodes_hb, K = harmonic_balance_general_current(
            components,
            freq=freq,
            num_harmonics=9
        )

        if nodes_order != nodes_hb:
            raise RuntimeError("Internal error: node order mismatch between BE and HB solvers.")

        v_hb_all = eval_all_nodes_from_coeffs(
            coeffs, t_tran, freq,
            Nn=len(nodes_order),
            K=K
        )

        # Plot every node: HB vs BE
        plt.figure(figsize=(8, 4))
        for idx, name in enumerate(nodes_order):
            plt.plot(t_tran, v_tran_all[:, idx], label=f"v({name}) (BE)")
            plt.plot(t_tran, v_hb_all[:, idx], "--", label=f"v({name}) (HB)")

        plt.xlabel("Time [s]")
        plt.ylabel("Node voltages [V]")
        plt.title("General current-driven circuit: Harmonic Balance vs Backward Euler")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()