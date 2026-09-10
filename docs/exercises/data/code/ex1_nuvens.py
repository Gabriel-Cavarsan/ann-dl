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
    partes = []
    for k in range(4):
        partes.append(rng.normal(MEDIAS[k], DESVIOS[k] * escala, size=(N_POR_CLASSE, 2)))
    X = np.vstack(partes)
    y = np.repeat(np.arange(4), N_POR_CLASSE)
    return X, y


def taxa_de_mistura(X, y):
    """Fração de pontos cujo centro de classe mais próximo não é o da própria classe."""
    # X[:, None, :] tem shape (400, 1, 2) e MEDIAS[None, :, :] tem (1, 4, 2). A subtração
    # faz broadcast e vira (400, 4, 2): a diferença de cada ponto para cada centro.
    diferencas = X[:, None, :] - MEDIAS[None, :, :]
    distancias = np.linalg.norm(diferencas, axis=2)   # (400, 4)
    mais_proximo = distancias.argmin(axis=1)
    return (mais_proximo != y).mean()


def desenhar_classes(ax, X, y, titulo):
    """Scatter colorido por classe, com o centro de cada nuvem marcado com um X preto."""
    for k in range(4):
        pontos = X[y == k]
        ax.scatter(pontos[:, 0], pontos[:, 1], s=12, alpha=0.6,
                   color=CORES[k], label=f"Classe {k}")
        ax.scatter(MEDIAS[k, 0], MEDIAS[k, 1], marker="X", s=150, color="black")
    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# --- A: as quatro nuvens em s = 1 --------------------------------------------
X, y = gerar()
fig, ax = plt.subplots(figsize=(8, 6))
desenhar_classes(ax, X, y, "Figura 1 - Quatro nuvens gaussianas (s = 1)\nX preto = centro de cada classe")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig1.png", dpi=150)

# --- C: fronteiras do centro mais próximo, desenhadas sobre a Figura 1 -------
# Mesma regra da taxa de mistura: para cada ponto da grade, qual centro é o mais perto.
gx, gy = np.meshgrid(np.linspace(-4, 20, 400), np.linspace(-6, 14, 400))
grade = np.column_stack([gx.ravel(), gy.ravel()])
distancias_grade = np.linalg.norm(grade[:, None, :] - MEDIAS[None, :, :], axis=2)
regiao = distancias_grade.argmin(axis=1).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(gx, gy, regiao, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=CORES, alpha=0.15)
ax.contour(gx, gy, regiao, levels=[0.5, 1.5, 2.5], colors="black")
desenhar_classes(ax, X, y, "Figura 1b - Fronteiras do centro mais próximo (s = 1)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig1b.png", dpi=150)

# --- B: o mesmo experimento em quatro escalas de espalhamento ----------------
datasets = {}
for s in ESCALAS:
    datasets[s] = gerar(s)

# Eixos compartilhados: os limites do dataset mais espalhado servem para todos.
X_maior = datasets[4.0][0]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), ESCALAS):
    X_s, y_s = datasets[s]
    desenhar_classes(ax, X_s, y_s, f"s = {s}")
    ax.set_xlim(X_maior[:, 0].min() - 1, X_maior[:, 0].max() + 1)
    ax.set_ylim(X_maior[:, 1].min() - 1, X_maior[:, 1].max() + 1)
axes[0, 0].legend()
fig.suptitle("Figura 2 - As mesmas 4 classes sob quatro fatores de espalhamento")
fig.tight_layout()
fig.savefig(FIGURAS / "fig2.png", dpi=150)

# Razão de separação r_ij = ||mu_i - mu_j|| / (sigma_i + sigma_j), só em s = 1.
sigma_barra = DESVIOS.mean(axis=1)  # média dos dois eixos, por classe
print("Razão de separação em s = 1")
razoes = {}
for i in range(4):
    for j in range(i + 1, 4):
        distancia = np.linalg.norm(MEDIAS[i] - MEDIAS[j])
        razoes[(i, j)] = distancia / (sigma_barra[i] + sigma_barra[j])
        print(f"  r_{i}{j} = {razoes[(i, j)]:.3f}")

par_minimo = min(razoes, key=razoes.get)
menor = razoes[par_minimo]
print(f"  menor: r_{par_minimo[0]}{par_minimo[1]} = {menor:.3f} -> em s = 2 vale {menor / 2:.3f}")

# Taxa de mistura por escala.
print("\nTaxa de mistura")
misturas = []
for s in ESCALAS:
    X_s, y_s = datasets[s]
    m = taxa_de_mistura(X_s, y_s)
    misturas.append(m)
    print(f"  s = {s}: {m:.4f}  ({m * 400:.0f} de 400 pontos)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ESCALAS, misturas, marker="o", label="taxa de mistura")
ax.set_title("Figura 3 - Taxa de mistura em função do espalhamento")
ax.set_xlabel("fator de escala s")
ax.set_ylabel("fração de pontos mal atribuídos")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig3.png", dpi=150)
