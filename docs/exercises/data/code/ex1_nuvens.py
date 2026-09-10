"""Exercício 1 - Nuvens de pontos: geometria e espalhamento em 2D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURAS = Path(__file__).resolve().parent.parent / "figures"
rng = np.random.default_rng(42)

# Parâmetros das 4 classes: uma média (x, y) e um desvio padrão por eixo.
MEDIAS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
DESVIOS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_POR_CLASSE = 100
CORES = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ESCALAS = [0.5, 1.0, 2.0, 4.0]


def gerar(escala=1.0):
    """400 pontos, 100 por classe. As médias são fixas; só os desvios escalam."""
    partes = [
        rng.normal(MEDIAS[k], DESVIOS[k] * escala, size=(N_POR_CLASSE, 2))
        for k in range(4)
    ]
    return np.vstack(partes), np.repeat(np.arange(4), N_POR_CLASSE)


def taxa_de_mistura(X, y):
    """Fração de pontos cujo centro de classe mais próximo não é o da própria classe."""
    # X[:, None, :] tem shape (400, 1, 2) e MEDIAS[None] tem (1, 4, 2): a subtração
    # faz broadcast e produz (400, 4, 2), a distância de cada ponto a cada centro.
    distancias = np.linalg.norm(X[:, None, :] - MEDIAS[None, :, :], axis=2)
    return (distancias.argmin(axis=1) != y).mean()


def desenhar_classes(ax, X, y, titulo):
    """Scatter colorido por classe, com o centro de cada nuvem marcado."""
    for k in range(4):
        ax.scatter(*X[y == k].T, s=12, alpha=0.6, color=CORES[k], label=f"Classe {k}")
        ax.scatter(*MEDIAS[k], marker="X", s=180, color=CORES[k],
                   edgecolor="black", linewidth=1.5, zorder=3)
    ax.set(title=titulo, xlabel="$x_1$", ylabel="$x_2$")


# --- A: as quatro nuvens em s = 1 --------------------------------------------
X, y = gerar()
fig, ax = plt.subplots(figsize=(8, 6))
desenhar_classes(ax, X, y, "Figura 1 - Quatro nuvens gaussianas ($s = 1$)\nX = centro (média) de cada classe")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig1.png", dpi=150)

# --- C: fronteiras do centro mais próximo sobre a Figura 1 -------------------
# Mesma regra geométrica da taxa de mistura, desenhada como regiões (Voronoi).
gx, gy = np.meshgrid(np.linspace(-4, 20, 400), np.linspace(-6, 14, 400))
grade = np.column_stack([gx.ravel(), gy.ravel()])
regiao = np.linalg.norm(grade[:, None, :] - MEDIAS[None, :, :], axis=2).argmin(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(gx, gy, regiao.reshape(gx.shape), levels=[-.5, .5, 1.5, 2.5, 3.5],
            colors=CORES, alpha=0.15)
ax.contour(gx, gy, regiao.reshape(gx.shape), levels=[.5, 1.5, 2.5],
           colors="black", linewidths=1.2)
desenhar_classes(ax, X, y, "Figura 1b - Esboço das fronteiras de decisão ($s = 1$)\nregiões do centro mais próximo")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig1b.png", dpi=150)

# --- B: o mesmo experimento em quatro escalas de espalhamento ----------------
datasets = {s: gerar(s) for s in ESCALAS}

# Eixos compartilhados: os limites do dataset mais espalhado servem para todos.
X_maior = datasets[4.0][0]
limites = dict(xlim=(X_maior[:, 0].min() - 1, X_maior[:, 0].max() + 1),
               ylim=(X_maior[:, 1].min() - 1, X_maior[:, 1].max() + 1))

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), ESCALAS):
    desenhar_classes(ax, *datasets[s], f"$s = {s}$")
    ax.set(**limites)
axes[0, 0].legend()
fig.suptitle("Figura 2 - As mesmas 4 classes sob quatro fatores de espalhamento", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURAS / "fig2.png", dpi=150)

# Razão de separação r_ij = ||mu_i - mu_j|| / (sigma_i + sigma_j), só em s = 1.
sigma_barra = DESVIOS.mean(axis=1)  # média dos dois eixos, por classe
print("Razão de separação em s = 1")
razoes = {}
for i in range(4):
    for j in range(i + 1, 4):
        r = np.linalg.norm(MEDIAS[i] - MEDIAS[j]) / (sigma_barra[i] + sigma_barra[j])
        razoes[(i, j)] = r
        print(f"  r_{i}{j} = {r:.3f}")
par_minimo = min(razoes, key=razoes.get)
print(f"  menor: r_{par_minimo[0]}{par_minimo[1]} = {razoes[par_minimo]:.3f}"
      f"  ->  em s = 2 vale {razoes[par_minimo] / 2:.3f}")

# Taxa de mistura por escala.
print("\nTaxa de mistura")
misturas = [taxa_de_mistura(*datasets[s]) for s in ESCALAS]
for s, m in zip(ESCALAS, misturas):
    print(f"  s = {s}: {m:.4f}  ({m * 400:.0f} de 400 pontos)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ESCALAS, misturas, marker="o", color="tab:purple", label="taxa de mistura")
for s, m in zip(ESCALAS, misturas):
    ax.annotate(f"{m:.3f}", (s, m), textcoords="offset points", xytext=(0, 9), ha="center")
ax.set(title="Figura 3 - Taxa de mistura em função do espalhamento",
       xlabel="fator de escala $s$", ylabel="fração de pontos mal atribuídos")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig3.png", dpi=150)
