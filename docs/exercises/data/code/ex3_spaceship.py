"""Exercício 3 - Preparando dados do mundo real (Spaceship Titanic) para uma rede tanh."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

BASE = Path(__file__).resolve().parent.parent
FIGURAS = BASE / "figures"

GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAS = ["Age"] + GASTOS
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DESCARTADAS = ["PassengerId", "Cabin", "Name"]

df = pd.read_csv(BASE / "dataset" / "train.csv")

# --- A: conhecer os dados ---------------------------------------------------
print(f"shape do dataset: {df.shape}")
print("\nBalanceamento de Transported:")
print(df["Transported"].value_counts(normalize=True).to_string())

print("\nValores faltantes por coluna:")
faltantes = pd.DataFrame({"faltantes": df.isna().sum(),
                          "percentual": (df.isna().mean() * 100).round(2)})
print(faltantes[faltantes["faltantes"] > 0].to_string())

print("\nColunas de gasto (dataset completo):")
print(df[GASTOS].agg(["mean", "median", "max"]).round(2).T.to_string())

# --- B: separar antes de qualquer estatística -------------------------------
X = df.drop(columns=DESCARTADAS + ["Transported"])
y = df["Transported"].astype(int)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\ntreino {X_treino.shape} | teste {X_teste.shape}")
print(f"proporção da classe positiva - treino {y_treino.mean():.4f} | teste {y_teste.mean():.4f}")
print(f"FoodCourt no treino antes de transformar: média {X_treino['FoodCourt'].mean():.2f}"
      f" | mediana {X_treino['FoodCourt'].median():.2f}")

foodcourt_bruto = X_treino["FoodCourt"].copy()  # guardado para o "antes" da Figura 6

# --- C: pré-processamento, sempre com fit no treino e transform no teste ----
# 1. Imputação: mediana nas numéricas (resiste à cauda pesada), moda nas categóricas.
imputador_num = SimpleImputer(strategy="median").fit(X_treino[NUMERICAS])
imputador_cat = SimpleImputer(strategy="most_frequent").fit(X_treino[CATEGORICAS])

X_treino[NUMERICAS] = imputador_num.transform(X_treino[NUMERICAS])
X_teste[NUMERICAS] = imputador_num.transform(X_teste[NUMERICAS])
X_treino[CATEGORICAS] = imputador_cat.transform(X_treino[CATEGORICAS])
X_teste[CATEGORICAS] = imputador_cat.transform(X_teste[CATEGORICAS])

# 2. Engenharia de features: TotalSpend soma os gastos ainda na escala original,
#    porque log(a) + log(b) não é log(a + b).
X_treino["TotalSpend"] = X_treino[GASTOS].sum(axis=1)
X_teste["TotalSpend"] = X_teste[GASTOS].sum(axis=1)

# 3. Cauda pesada: log(1 + x) nos gastos e no total derivado deles.
COLUNAS_LOG = GASTOS + ["TotalSpend"]
X_treino[COLUNAS_LOG] = np.log1p(X_treino[COLUNAS_LOG])
X_teste[COLUNAS_LOG] = np.log1p(X_teste[COLUNAS_LOG])

# 4. Categóricas -> one-hot. handle_unknown="ignore" faz uma categoria inédita no
#    teste virar uma linha de zeros, em vez de quebrar o transform.
codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
codificador.fit(X_treino[CATEGORICAS])

print("\nCategorias vistas no teste e ausentes do treino:")
for coluna in CATEGORICAS:
    novas = set(X_teste[coluna].unique()) - set(X_treino[coluna].unique())
    print(f"  {coluna}: {novas or 'nenhuma'}")

# 5. Escalonamento das numéricas para [-1, 1], que é a imagem da tanh.
COLUNAS_NUM = NUMERICAS + ["TotalSpend"]
escalador = MinMaxScaler(feature_range=(-1, 1)).fit(X_treino[COLUNAS_NUM])

# Matriz final: numéricas escaladas à esquerda, categóricas codificadas à direita.
Xt_treino = np.hstack([escalador.transform(X_treino[COLUNAS_NUM]),
                       codificador.transform(X_treino[CATEGORICAS])])
Xt_teste = np.hstack([escalador.transform(X_teste[COLUNAS_NUM]),
                      codificador.transform(X_teste[CATEGORICAS])])

# --- D: verificação e visualização ------------------------------------------
print(f"\nNaN remanescentes: treino {np.isnan(Xt_treino).sum()} | teste {np.isnan(Xt_teste).sum()}")
print(f"shape final: treino {Xt_treino.shape} | teste {Xt_teste.shape}")
print(f"intervalo treino: [{Xt_treino.min():.4f}, {Xt_treino.max():.4f}]")
print(f"intervalo teste:  [{Xt_teste.min():.4f}, {Xt_teste.max():.4f}]")

# FoodCourt é a segunda coluna numérica (Age vem primeiro), logo índice 1.
indice_foodcourt = COLUNAS_NUM.index("FoodCourt")

fig, (ax_antes, ax_depois) = plt.subplots(1, 2, figsize=(13, 5))
# Eixo y em escala log nos dois: sem isso, só a barra dos zeros seria visível.
ax_antes.hist(foodcourt_bruto.dropna(), bins=60, color="tab:red", label="bruto")
ax_antes.set_title("Antes: FoodCourt bruto")
ax_antes.set_xlabel("gasto (unidades monetárias)")
ax_antes.set_ylabel("frequência")
ax_antes.set_yscale("log")
ax_antes.legend()

ax_depois.hist(Xt_treino[:, indice_foodcourt], bins=60, color="tab:green", label="transformado")
ax_depois.set_title("Depois: log(1+x) e escala [-1, 1]")
ax_depois.set_xlabel("valor escalado")
ax_depois.set_ylabel("frequência")
ax_depois.set_yscale("log")
ax_depois.legend()

fig.suptitle("Figura 6 - FoodCourt no conjunto de treino, antes e depois do pré-processamento")
fig.tight_layout()
fig.savefig(FIGURAS / "fig6.png", dpi=150)
