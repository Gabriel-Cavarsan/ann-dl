"""Exercício 2 - Dados sobrepostos: o caso que o perceptron não resolve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import gerar_duas_classes, prever, treinar

FIGURAS = Path(__file__).resolve().parent.parent / "figures"
rng = np.random.default_rng(42)
CORES = ["tab:blue", "tab:red"]

# --- A: duas nuvens próximas e três vezes mais espalhadas -------------------
X, y = gerar_duas_classes(rng, media_0=[3.0, 3.0], media_1=[4.0, 4.0], variancia=1.5)


def desenhar_pontos(ax):
    for k in (0, 1):
        pontos = X[y == k]
        ax.scatter(pontos[:, 0], pontos[:, 1], s=8, alpha=0.4,
                   color=CORES[k], label=f"Classe {k}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def desenhar_fronteira(ax, w, b, cor, rotulo, estilo="-"):
    x1 = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    x2 = -(w[0] * x1 + b) / w[1]
    ax.plot(x1, x2, color=cor, linewidth=2, linestyle=estilo, label=rotulo)


fig, ax = plt.subplots(figsize=(7, 6))
desenhar_pontos(ax)
ax.set_title("Figura 4 - Duas classes sobrepostas")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig4.png", dpi=150)

# --- B: mesmo treino do Exercício 1, agora acompanhando o pocket ------------
w_inicial = rng.normal(0, 0.01, size=2)
r = treinar(X, y, w_inicial, taxa=0.01, max_epocas=100)

print(f"épocas executadas: {r['epocas']} (o laço não parou sozinho)")
print(f"atualizações na última época: {r['atualizacoes_por_epoca'][-1]} de {len(X)} amostras")
print()
print("--- pesos finais ---")
print(f"  w = [{r['w'][0]:.4f}, {r['w'][1]:.4f}]  b = {r['b']:.4f}")
print(f"  acurácia = {r['acuracia']:.4f}")
print("--- pesos do pocket ---")
print(f"  w = [{r['pocket_w'][0]:.4f}, {r['pocket_w'][1]:.4f}]  b = {r['pocket_b']:.4f}")
print(f"  acurácia = {r['pocket_acuracia']:.4f}  (melhor visto na época {r['pocket_epoca']})")

# --- material para a análise do item D --------------------------------------
# Uma reta w.x + b = 0 fica a |b|/||w|| da origem. Os dados estão centrados em
# [3.5, 3.5], a uma distância ||[3.5, 3.5]|| da origem: se a fronteira ficar
# muito mais perto da origem do que isso, ela passa fora da nuvem.
centro = X.mean(axis=0)
print()
print(f"centro dos dados: [{centro[0]:.2f}, {centro[1]:.2f}], a {np.linalg.norm(centro):.2f} da origem")
for nome, w, b in (("final", r["w"], r["b"]), ("pocket", r["pocket_w"], r["pocket_b"])):
    print(f"  fronteira {nome:6}: |b|/||w|| = {abs(b) / np.linalg.norm(w):6.2f}")

# Quanto cada erro move w e quanto move b: ||x|| vezes mais no caso de w.
print(f"  ||x|| médio = {np.linalg.norm(X, axis=1).mean():.2f}  "
      f"-> cada erro move w em ~{0.01 * np.linalg.norm(X, axis=1).mean():.3f} e b em 0.010")

# A predição da fronteira final é quase constante: ela joga quase tudo numa classe.
pred_final = prever(X, r["w"], r["b"])
print(f"  a fronteira final prediz classe 1 em {pred_final.sum()} das {len(X)} amostras")
print(f"  é praticamente uma constante: acerta {int((pred_final == y).sum())} de {len(X)},"
      f" que é o tamanho da classe 1 ({int((y == 1).sum())}) mais um ponto")

# --- C: figuras -------------------------------------------------------------
errados_pocket = prever(X, r["pocket_w"], r["pocket_b"]) != y
fig, ax = plt.subplots(figsize=(7.5, 6))
desenhar_pontos(ax)
ax.scatter(X[errados_pocket][:, 0], X[errados_pocket][:, 1], s=28, facecolors="none",
           edgecolors="dimgray", linewidth=0.6,
           label=f"errados pelo pocket ({errados_pocket.sum()})")
desenhar_fronteira(ax, r["pocket_w"], r["pocket_b"], "black", "fronteira pocket")
desenhar_fronteira(ax, r["w"], r["b"], "darkorange", "fronteira final", estilo="--")
ax.set_title("Figura 5 - Fronteiras final e pocket sobre os dados")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURAS / "fig5.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
epocas = range(1, r["epocas"] + 1)
ax.plot(epocas, r["historico_acc"], linewidth=1, color="tab:orange", label="acurácia dos pesos atuais")
ax.plot(epocas, r["historico_melhor"], linewidth=2, color="black", label="melhor até agora (pocket)")
ax.set_title("Figura 6 - Acurácia por época (dados sobrepostos)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "fig6.png", dpi=150)
