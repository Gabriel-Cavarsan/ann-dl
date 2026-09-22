"""Exercício 1 - Dados separáveis: o caso para o qual o perceptron foi projetado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import gerar_duas_classes, prever, treinar

FIGURAS = Path(__file__).resolve().parent.parent / "figures"
rng = np.random.default_rng(42)
CORES = ["tab:blue", "tab:red"]

# --- A: duas nuvens bem separadas -------------------------------------------
X, y = gerar_duas_classes(rng, media_0=[1.5, 1.5], media_1=[5.0, 5.0], variancia=0.5)


def desenhar_pontos(ax):
    """Scatter das duas classes, usado em todas as figuras deste exercício."""
    for k in (0, 1):
        pontos = X[y == k]
        ax.scatter(pontos[:, 0], pontos[:, 1], s=8, alpha=0.5,
                   color=CORES[k], label=f"Classe {k}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def desenhar_fronteira(ax, w, b, cor, rotulo):
    """A fronteira é a reta w0*x1 + w1*x2 + b = 0, isolada em x2."""
    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    x2 = -(w[0] * x1 + b) / w[1]
    ax.plot(x1, x2, color=cor, linewidth=2, label=rotulo)


fig, ax = plt.subplots(figsize=(7, 6))
desenhar_pontos(ax)
ax.set_title("Figura 1 - Duas classes linearmente separáveis")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig1.png", dpi=150)

# --- C: treino com a taxa pedida --------------------------------------------
# O w inicial é sorteado uma única vez e reaproveitado na execução com eta = 1.0,
# para que a comparação do item D isole o efeito da taxa de aprendizado.
w_inicial = rng.normal(0, 0.01, size=2)
print(f"w inicial sorteado: [{w_inicial[0]:.6f}, {w_inicial[1]:.6f}]")

r = treinar(X, y, w_inicial, taxa=0.01)
print(f"\n--- eta = 0.01 ---")
print(f"  w final   = [{r['w'][0]:.4f}, {r['w'][1]:.4f}]")
print(f"  b final   = {r['b']:.4f}")
print(f"  épocas    = {r['epocas']}")
print(f"  acurácia  = {r['acuracia']:.4f}  ({int(r['acuracia'] * len(X))} de {len(X)})")
print(f"  atualizações por época: {r['atualizacoes_por_epoca']}")

errados = prever(X, r["w"], r["b"]) != y
fig, ax = plt.subplots(figsize=(7, 6))
desenhar_pontos(ax)
desenhar_fronteira(ax, r["w"], r["b"], "black", "fronteira aprendida")
ax.scatter(X[errados][:, 0], X[errados][:, 1], s=90, facecolors="none",
           edgecolors="black", linewidth=1.5, label=f"mal classificados ({errados.sum()})")
ax.set_title("Figura 2 - Fronteira de decisão e pontos mal classificados")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig2.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 5))
epocas = range(1, r["epocas"] + 1)
ax.plot(epocas, r["historico_acc"], marker="o", label="acurácia no dataset")
ax.set_title("Figura 3 - Acurácia por época (dados separáveis)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig3.png", dpi=150)

# --- D2: mesma inicialização, taxa 100 vezes maior --------------------------
r_alto = treinar(X, y, w_inicial, taxa=1.0)
print(f"\n--- eta = 1.0 (mesma inicialização) ---")
print(f"  w final   = [{r_alto['w'][0]:.4f}, {r_alto['w'][1]:.4f}]")
print(f"  b final   = {r_alto['b']:.4f}")
print(f"  épocas    = {r_alto['epocas']}")
print(f"  acurácia  = {r_alto['acuracia']:.4f}")


def direcao(w):
    """Vetor unitário na direção de w: é ele que define a inclinação da fronteira."""
    return w / np.linalg.norm(w)


d1, d2 = direcao(r["w"]), direcao(r_alto["w"])
angulo = np.degrees(np.arccos(np.clip(d1 @ d2, -1, 1)))
print(f"\n  direção com eta=0.01: [{d1[0]:.4f}, {d1[1]:.4f}]")
print(f"  direção com eta=1.00: [{d2[0]:.4f}, {d2[1]:.4f}]")
print(f"  ângulo entre as duas fronteiras: {angulo:.2f} graus")
print(f"  ||w|| eta=0.01: {np.linalg.norm(r['w']):.4f} | ||w|| eta=1.0: {np.linalg.norm(r_alto['w']):.4f}")

# A reta não é definida só pela direção de w: a distância dela até a origem
# vale |b| / ||w||. Duas retas paralelas com offsets diferentes são retas diferentes.
off1 = abs(r["b"]) / np.linalg.norm(r["w"])
off2 = abs(r_alto["b"]) / np.linalg.norm(r_alto["w"])
print(f"  distância da fronteira à origem: eta=0.01 -> {off1:.4f} | eta=1.0 -> {off2:.4f}")

# --- D3: verificação numérica de que, partindo do zero, eta só reescala -----
zero = np.zeros(2)
z1 = treinar(X, y, zero, taxa=0.01)
z2 = treinar(X, y, zero, taxa=1.0)
print(f"\n--- partindo de w = 0 (o que o enunciado proíbe, testado para o item D3) ---")
print(f"  eta=0.01: w = [{z1['w'][0]:.6f}, {z1['w'][1]:.6f}], b = {z1['b']:.6f}, épocas = {z1['epocas']}")
print(f"  eta=1.00: w = [{z2['w'][0]:.6f}, {z2['w'][1]:.6f}], b = {z2['b']:.6f}, épocas = {z2['epocas']}")
print(f"  razão w (deveria ser 100): [{z2['w'][0] / z1['w'][0]:.4f}, {z2['w'][1] / z1['w'][1]:.4f}]")
print(f"  razão b (deveria ser 100): {z2['b'] / z1['b']:.4f}")
print(f"  as duas fronteiras são a mesma reta: {np.allclose(z2['w'] / 100, z1['w']) and np.isclose(z2['b'] / 100, z1['b'])}")

# Com eta = 1.0 cada passo vale ~5, contra um w inicial de ~0.009: a inicialização
# vira irrelevante e o treino se comporta como se tivesse partido do zero.
print(f"\n  eta=1.0 partindo do w sorteado: w = [{r_alto['w'][0]:.4f}, {r_alto['w'][1]:.4f}], {r_alto['epocas']} épocas")
print(f"  eta=1.0 partindo do zero:        w = [{z2['w'][0]:.4f}, {z2['w'][1]:.4f}], {z2['epocas']} épocas")
print(f"  norma do w inicial: {np.linalg.norm(w_inicial):.6f} | norma de um passo com eta=1.0: ~{np.linalg.norm(X, axis=1).mean():.2f}")
