"""Exercício 2 - Não-linearidade em dimensões maiores (5D)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURAS = Path(__file__).resolve().parent.parent / "figures"
rng = np.random.default_rng(42)
N_POR_CLASSE = 500
CORES = ["tab:blue", "tab:red"]

# --- A: Dataset I - duas gaussianas 5D com médias e covariâncias diferentes --
MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [ 1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7,  1.5, 0.4, 0.0, 0.0],
    [ 0.2,  0.4, 1.5, 0.6, 0.0],
    [ 0.0,  0.0, 0.6, 1.5, 0.3],
    [ 0.0,  0.0, 0.0, 0.3, 1.5],
])

X1 = np.vstack([rng.multivariate_normal(MU_A, SIGMA_A, N_POR_CLASSE),
                rng.multivariate_normal(MU_B, SIGMA_B, N_POR_CLASSE)])
y1 = np.repeat([0, 1], N_POR_CLASSE)


# --- B: Dataset II - duas cascas esféricas concêntricas ---------------------
def casca(raio_medio, desvio=0.4, n=N_POR_CLASSE):
    """Direção uniforme na esfera unitária de R^5, multiplicada por um raio gaussiano."""
    v = rng.normal(size=(n, 5))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)  # dividir pela norma deixa ||u|| = 1
    raio = rng.normal(raio_medio, desvio, size=(n, 1))
    return raio * u


X2 = np.vstack([casca(2.0), casca(5.0)])
y2 = np.repeat([0, 1], N_POR_CLASSE)

DATASETS = [
    ("Dataset I - gaussianas deslocadas", X1, y1, ["Classe A", "Classe B"]),
    ("Dataset II - cascas concêntricas", X2, y2, ["Classe C", "Classe D"]),
]

# --- C: projeção PCA, distância entre centros e histograma dos raios --------
fig_pca, axes_pca = plt.subplots(1, 2, figsize=(13, 5.5))
fig_raio, axes_raio = plt.subplots(1, 2, figsize=(13, 5))

for i in range(2):
    nome, X, y, rotulos = DATASETS[i]
    ax_pca = axes_pca[i]
    ax_raio = axes_raio[i]

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    variancia = pca.explained_variance_ratio_

    centro_0 = X[y == 0].mean(axis=0)
    centro_1 = X[y == 1].mean(axis=0)
    distancia = np.linalg.norm(centro_0 - centro_1)
    raios = np.linalg.norm(X, axis=1)

    print(f"\n{nome}")
    print(f"  distância entre os centros em 5D: {distancia:.4f}")
    print(f"  variância explicada: PC1 {variancia[0]:.1%}, PC2 {variancia[1]:.1%}"
          f" -> soma {variancia.sum():.1%}")

    for k in range(2):
        raios_da_classe = raios[y == k]
        print(f"  raio médio {rotulos[k]}: {raios_da_classe.mean():.3f}"
              f" (desvio {raios_da_classe.std():.3f})")

        Z_da_classe = Z[y == k]
        ax_pca.scatter(Z_da_classe[:, 0], Z_da_classe[:, 1], s=10, alpha=0.5,
                       color=CORES[k], label=rotulos[k])
        ax_raio.hist(raios_da_classe, bins=40, alpha=0.6, color=CORES[k], label=rotulos[k])

    ax_pca.set_title(f"{nome}\nPC1+PC2 explicam {variancia.sum():.1%} da variância")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend()

    ax_raio.set_title(nome)
    ax_raio.set_xlabel("raio ||x||")
    ax_raio.set_ylabel("frequência")
    ax_raio.legend()

fig_pca.suptitle("Figura 4 - Projeção PCA em 2D dos dois datasets 5D")
fig_pca.tight_layout()
fig_pca.savefig(FIGURAS / "fig4.png", dpi=150)

fig_raio.suptitle("Figura 5 - Distribuição do raio ||x|| por classe (calculado em 5D)")
fig_raio.tight_layout()
fig_raio.savefig(FIGURAS / "fig5.png", dpi=150)

# --- D: a regra do raio separa o Dataset II sem nenhum treinamento ----------
# Limiar no meio do caminho entre as duas cascas: (2 + 5) / 2 = 3.5.
raio_ao_quadrado = np.sum(X2 ** 2, axis=1)
predito = (raio_ao_quadrado > 3.5 ** 2).astype(int)
print(f"\nRegra ||x||^2 > 3.5^2 no Dataset II: {(predito == y2).mean():.2%} de acerto")
