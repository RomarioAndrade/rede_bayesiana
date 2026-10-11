"""2º Trabalho de IA (UFAM/IComp) - rede bayesiana de diagnóstico:
farol de bicicleta movido a dínamo.

1ª Questão:
  (a) estrutura da rede, checada contra as independências do enunciado e
      desenhada em PNG (Str -> R <- Flw ; R -> V ; V -> Li <- B ; K -> Li);
  (b) CPTs faltantes;  (c) valores plausíveis para elas;
  (d) a rede não precisa da aresta (Str, Li);
  (e) P(V | Str = snow_covered).
Inferência exata por enumeração (soma sobre a distribuição conjunta).
"""
import shutil
import subprocess
from itertools import product
from pathlib import Path

# ---------- Domínios ----------
DOM = {
    "Str": ["dry", "wet", "snow_covered"],  # condição da rua
    "Flw": [True, False],                   # volante do dínamo desgastado
    "R":   [True, False],                   # dínamo deslizante
    "V":   [True, False],                   # dínamo mostra tensão
    "B":   [True, False],                   # lâmpada ok
    "K":   [True, False],                   # cabo ok
    "Li":  [True, False],                   # luz ligada
}

# ---------- (a) Estrutura: pais de cada nó ----------
PAIS = {
    "Str": [], "Flw": [], "B": [], "K": [],
    "R":   ["Str", "Flw"],
    "V":   ["R"],
    "Li":  ["V", "B", "K"],
}

# ---------- (b)/(c) CPTs (valores escolhidos) ----------
P_STR = {"dry": 0.70, "wet": 0.20, "snow_covered": 0.10}
P_FLW = 0.10                              # P(Flw = t)
P_B = 0.95                                # P(B = t)
P_K = 0.97                                # P(K = t)
P_R = {                                   # P(R = t | Str, Flw)
    ("dry", False): 0.01, ("dry", True): 0.20,
    ("wet", False): 0.10, ("wet", True): 0.50,
    ("snow_covered", False): 0.40, ("snow_covered", True): 0.90,
}
P_V = {True: 0.10, False: 0.99}           # P(V = t | R)
P_LI = {                                  # P(Li = t | V, B, K)  -- dada no enunciado
    (True, True, True): 0.99, (True, True, False): 0.01,
    (True, False, True): 0.01, (True, False, False): 0.001,
    (False, True, True): 0.3, (False, True, False): 0.005,
    (False, False, True): 0.005, (False, False, False): 0.0,
}
assert abs(sum(P_STR.values()) - 1) < 1e-12


def bern(p, valor):
    return p if valor else 1 - p


def tf(valor):
    return valor if isinstance(valor, str) else ("t" if valor else "f")


def conjunta(a):
    """P(Str,Flw,R,V,B,K,Li) = produto de P(no | pais), seguindo PAIS."""
    return (P_STR[a["Str"]] * bern(P_FLW, a["Flw"]) * bern(P_B, a["B"]) * bern(P_K, a["K"])
            * bern(P_R[(a["Str"], a["Flw"])], a["R"])
            * bern(P_V[a["R"]], a["V"])
            * bern(P_LI[(a["V"], a["B"], a["K"])], a["Li"]))


def prob(consulta, evidencia=None):
    """P(consulta | evidencia) por enumeração. Ambos são dicts {variavel: valor}."""
    evidencia = evidencia or {}
    nomes = list(DOM)
    num = den = 0.0
    for valores in product(*(DOM[n] for n in nomes)):
        a = dict(zip(nomes, valores))
        if any(a[k] != v for k, v in evidencia.items()):
            continue
        p = conjunta(a)
        den += p
        if all(a[k] == v for k, v in consulta.items()):
            num += p
    return num / den


def independentes(x, y, dado=()):
    """X ⊥ Y | dado  <=>  P(x | y, z) = P(x | z) para todos os valores de x, y, z."""
    for vx, vy, *vz in product(DOM[x], DOM[y], *(DOM[z] for z in dado)):
        z = dict(zip(dado, vz))
        if abs(prob({x: vx}, {y: vy, **z}) - prob({x: vx}, z)) > 1e-9:
            return False
    return True


def desenha_rede(arquivo):
    """Gera o desenho da rede (item a) com o Graphviz; devolve o código DOT."""
    arestas = "\n".join(f"  {p} -> {n};" for n, ps in PAIS.items() for p in ps)
    dot = ("digraph rede {\n  rankdir=TB;\n  node [shape=ellipse, fontname=Helvetica];\n"
           "  {rank=same; Str; Flw;}\n  {rank=same; V; B; K;}\n" + arestas + "\n}\n")
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", "-o", str(arquivo)], input=dot, text=True, check=True)
    return dot


if __name__ == "__main__":
    def mostra(txt, c, e=None):
        print(f"{txt:<48} = {prob(c, e):.4f}")

    def titulo(txt):
        print(f"\n{'=' * 64}\n{txt}\n{'=' * 64}")

    # ----- (a) -----
    titulo("(a) Rede bayesiana de causalidade")
    print("""
      Str     Flw
        \\     /
         v   v
           R
           |
           v
           V      B      K
            \\     |     /
             v    v    v
                 Li
""")
    for no, ps in PAIS.items():
        print(f"  pais({no:<3}) = {{{', '.join(ps)}}}")
    png = Path(__file__).with_name("rede_farol_dinamo.png")
    dot = desenha_rede(png)
    print(f"\n  Desenho salvo em {png.name}" if png.exists()
          else f"\n  Graphviz não encontrado; código DOT da rede:\n{dot}")

    print("\n  A rede satisfaz as independências do enunciado?")
    checagens = [(f"{x} ⊥ {y}", x, y, ()) for i, x in enumerate(["Str", "Flw", "B", "K"])
                 for y in ["Str", "Flw", "B", "K"][i + 1:]]
    checagens += [(f"{x} ⊥ {y}", x, y, ()) for x, y in [("R", "B"), ("R", "K"), ("V", "B"), ("V", "K")]]
    checagens += [("P(Li | V, R)   = P(Li | V)", "Li", "R", ("V",)),
                  ("P(V | R, Str)  = P(V | R)", "V", "Str", ("R",)),
                  ("P(V | R, Flw)  = P(V | R)", "V", "Flw", ("R",))]
    for txt, x, y, dado in checagens:
        print(f"    {txt:<28} {'ok' if independentes(x, y, dado) else 'FALHOU'}")

    # ----- (b) e (c) -----
    titulo("(b) e (c) CPTs  (em todas, P(f) = 1 - P(t))")
    print("  P(Str): " + ", ".join(f"{s} = {p:.2f}" for s, p in P_STR.items()))
    print(f"  P(Flw = t) = {P_FLW:.2f}    P(B = t) = {P_B:.2f}    P(K = t) = {P_K:.2f}")
    print(f"\n  {'Str':<14}{'Flw':<5}P(R = t | Str, Flw)")
    for s in DOM["Str"]:
        for f in [False, True]:
            print(f"  {s:<14}{tf(f):<5}{P_R[(s, f)]:.2f}")
    print(f"\n  {'R':<5}P(V = t | R)")
    for r in DOM["R"]:
        print(f"  {tf(r):<5}{P_V[r]:.2f}")
    print(f"\n  {'V':<4}{'B':<4}{'K':<4}P(Li = t | V, B, K)   (dada no enunciado)")
    for (v, b, k), p in P_LI.items():
        print(f"  {tf(v):<4}{tf(b):<4}{tf(k):<4}{p:g}")

    # ----- (d) -----
    titulo("(d) A rede não contém a aresta (Str, Li)")
    print("  O único caminho entre Str e Li é a cadeia Str -> R -> V -> Li; dado V ele")
    print("  fica bloqueado (d-separação). Como B, K ⊥ V, Str:")
    print("    P(Li | V, Str) = Σ_b Σ_k P(Li | V, b, k) P(b) P(k),  que não depende de Str.\n")
    for v in DOM["V"]:
        analitico = sum(P_LI[(v, b, k)] * bern(P_B, b) * bern(P_K, k)
                        for b in DOM["B"] for k in DOM["K"])
        for s in DOM["Str"]:
            mostra(f"  P(Li | V={tf(v)}, Str={s})", {"Li": True}, {"V": v, "Str": s})
        print(f"{f'  Σ_b Σ_k P(Li|V={tf(v)},b,k) P(b) P(k)':<48} = {analitico:.4f}\n")
    print(f"  Li ⊥ Str | V : {independentes('Li', 'Str', ('V',))}")
    print(f"  Li ⊥ Str     : {independentes('Li', 'Str')}   "
          "(Str influencia Li, mas só através de V)")

    # ----- (e) -----
    titulo("(e) P(V | Str = snow_covered)")
    s = "snow_covered"
    p_r = sum(P_R[(s, f)] * bern(P_FLW, f) for f in [True, False])
    p_v = P_V[True] * p_r + P_V[False] * (1 - p_r)
    print("  Passo 1: P(R | s) = P(R | s, Flw) P(Flw) + P(R | s, ~Flw) P(~Flw)   (Flw ⊥ Str)")
    print(f"                    = {P_R[(s, True)]:.2f}·{P_FLW:.2f} + {P_R[(s, False)]:.2f}·"
          f"{1 - P_FLW:.2f} = {p_r:.4f}")
    print("  Passo 2: P(V | s) = P(V | R) P(R | s) + P(V | ~R) P(~R | s)   (V ⊥ Str | R)")
    print(f"                    = {P_V[True]:.2f}·{p_r:.2f} + {P_V[False]:.2f}·{1 - p_r:.2f}"
          f" = {p_v:.4f}")
    mostra("  Conferência por enumeração: P(V | Str=snow)", {"V": True}, {"Str": s})

    # ----- Extra: uso da rede para diagnóstico -----
    titulo("Extra: diagnóstico quando a luz NÃO acende")
    e = {"Li": False}
    mostra("P(lampada queimada | ~Li)", {"B": False}, e)
    mostra("P(cabo rompido | ~Li)", {"K": False}, e)
    mostra("P(dinamo deslizando | ~Li)", {"R": True}, e)
    mostra("P(volante desgastado | ~Li)", {"Flw": True}, e)
    e = {"Li": False, "Str": "snow_covered"}
    mostra("P(deslizando | ~Li, snow)", {"R": True}, e)
    mostra("P(lampada queimada | ~Li, snow)", {"B": False}, e)
    e = {"Li": False, "V": True}
    mostra("P(lampada queimada | ~Li, V)", {"B": False}, e)
    mostra("P(cabo rompido | ~Li, V)", {"K": False}, e)
    mostra("P(deslizando | ~Li, V)", {"R": True}, e)
    e = {"Li": False, "V": True, "K": True}
    mostra("P(lampada queimada | ~Li, V, K)", {"B": False}, e)
