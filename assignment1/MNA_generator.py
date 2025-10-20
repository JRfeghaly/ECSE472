import numpy as np
np.set_printoptions(precision=6, suppress=True)

def get_lines(file_path):
    netlist = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            #skip empty or comment lines
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            #stop at .end
            if line.lower() == '.end':
                break
            netlist.append(line)
    return netlist

def parse_components(lines):
    """
    Returns a list of component dicts.
    For E (VCVS): format  Ename n+ n- nc+ nc- gain
    For V: Vname n+ n- [DC|AC] value
    Others (R, I, C): name n1 n2 value
    """
    comps = []
    for line in lines:
        tks = line.split()
        name = tks[0]
        ctype = name[0].upper()

        if ctype == "E":
            #VCVS: Ename n+ n- nc+ nc- gain
            n1, n2, nc1, nc2 = tks[1], tks[2], tks[3], tks[4]
            gain = float(tks[5])
            comps.append({
                "type": "E",
                "name": name,
                "n1": n1, "n2": n2,
                "ctrl_p": nc1, "ctrl_n": nc2,
                "gain": gain
            })
            continue

        #common 2-nodes forms
        n1, n2 = tks[1], tks[2]

        if ctype == "V":
            #Vname n+ n- [DC|AC] value
            if len(tks) >= 5 and tks[3].upper() in ("DC", "AC"):
                value = float(tks[4])
            else:
                value = float(tks[3])
            comps.append({"type": "V", "name": name, "n1": n1, "n2": n2, "value": value})

        elif ctype in ("R", "I", "C"):
            value = float(tks[3])
            comps.append({"type": ctype, "name": name, "n1": n1, "n2": n2, "value": value})
        else:
            raise ValueError(f"Unsupported element type on line: {line}")
    return comps

def build_node_map(components):
    nodes = set()
    vsrc_names = []#includes independent V and E (VCVS) because both add a source current variable

    for c in components:
        if c["n1"] != "0": nodes.add(c["n1"])
        if c["n2"] != "0": nodes.add(c["n2"])

        #control nodes for VCVS
        if c["type"] == "E":
            if c["ctrl_p"] != "0": nodes.add(c["ctrl_p"])
            if c["ctrl_n"] != "0": nodes.add(c["ctrl_n"])
            vsrc_names.append(c["name"])  #E behaves like a voltage source in MNA
        elif c["type"] == "V":
            vsrc_names.append(c["name"])

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}  #starts at 1 for readability
    N = len(nodes) #number of non-ground nodes (number of voltage variables)
    M = len(vsrc_names) #number of voltage sources (number of current variables)
    return node_index, vsrc_names, N, M
    
def get_index(node, node_map):
    return node_map[node] - 1 if node != '0' else None

def stamp_components(components, node_map, voltage_sources, N, M):
    G = np.zeros((N, N))
    C = np.zeros((N, N))
    B = np.zeros((N, M))
    b = np.zeros((N + M, 1))
    Ectrl = np.zeros((M, N))

    for c in components:
        n1 = get_index(c['n1'], node_map)
        n2 = get_index(c['n2'], node_map)
        t = c['type']
        if t != 'E': val = c['value']

        if t == "R":
            g = 1 / val
            if n1 is not None:
                G[n1, n1] += g
            if n2 is not None:
                G[n2, n2] += g
            if n1 is not None and n2 is not None:
                G[n1, n2] -= g
                G[n2, n1] -= g

        elif t == "I":
            if n1 is not None:
                b[n1, 0] -= val
            if n2 is not None:
                b[n2, 0] += val

        elif t == "V":
            k = voltage_sources.index(c['name'])
            if n1 is not None:
                B[n1, k] = 1
            if n2 is not None:
                B[n2, k] = -1
            b[N + k, 0] = val

        elif t == "C":
            cval = val
            if n1 is not None:
                C[n1, n1] += cval
            if n2 is not None:
                C[n2, n2] += cval
            if n1 is not None and n2 is not None:
                C[n1, n2] -= cval
                C[n2, n1] -= cval
        
        elif t == "E":#VCVS
            k = voltage_sources.index(c["name"])
            #main terminals behave like a V source (value enforced by extra equation)
            if n1 is not None: B[n1, k] = 1.0
            if n2 is not None: B[n2, k] = -1.0
            #control relationship: v(n1)-v(n2) - gain*(v(cp)-v(cn)) = 0
            cp = get_index(c["ctrl_p"], node_map)
            cn = get_index(c["ctrl_n"], node_map)
            gain = c["gain"]
            #row k of the bottom-left block gets additional coefficients on control nodes
            if cp is not None: Ectrl[k, cp] += -gain
            if cn is not None: Ectrl[k, cn] += +gain

    return G, C, B, Ectrl, b

def build_system(G, B, Ectrl, N, M):
    #A = [[G, B], line 1 
    #[B^T + Ectrl, 0]] line 2
    top = np.hstack((G, B))
    bottom_left = B.T + Ectrl
    bottom = np.hstack((bottom_left, np.zeros((M, M))))
    A = np.vstack((top, bottom))
    return A

def main():
    file_path = "circuit.txt"
    lines = get_lines(file_path)
    comps = parse_components(lines)
    node_index, vsrc_names, N, M = build_node_map(comps) #identify all nodes and voltage sources
    G, C, B, Ectrl, b = stamp_components(comps, node_index, vsrc_names, N, M)
    A = build_system(G, B, Ectrl, N, M)

    try:
        x = np.linalg.solve(A, b) #solve Ax = b for x
    except np.linalg.LinAlgError:
        print("Singular system: check for ungrounded nodes or conflicting sources.")
        return

    #print G C b
    print("G matrix (conductance):\n", G)
    print("\nC matrix (capacitors):\n", C)
    print("\nb vector (sources):\n", b)

    print("\nNode Voltages:")
    for name, one_based in node_index.items():
        print(f"{name}: {x[one_based - 1, 0]:.6f} V")

    if M:
        print("\nCurrents through Voltage/VCVS sources:")
        for k, sname in enumerate(vsrc_names):
            print(f"{sname}: {x[N + k, 0]:.6f} A")

if __name__ == "__main__":
    main()