# ANN-DL — Gabriel Cavarsan

Entregas da disciplina **Redes Neurais Artificiais e Deep Learning** — Insper, 2026.2.

Site publicado: **https://gabriel-cavarsan.github.io/ann-dl**

## Entregas

| # | Exercício | Pasta |
|---|-----------|-------|
| 1 | Data — preparação e análise de dados | [`docs/exercises/data/`](docs/exercises/data/) |

Cada entrega tem o relatório em `index.md`, os scripts que realmente rodaram em `code/` e as figuras que o relatório exibe em `figures/`.

## Rodando localmente

```shell
python -m venv env
.\env\Scripts\activate          # Windows
python -m pip install -r requirements.txt
mkdocs serve
```

Os scripts de cada exercício rodam de forma independente e regeram as figuras:

```shell
python docs/exercises/data/code/ex1_nuvens.py
python docs/exercises/data/code/ex2_dimensoes.py
python docs/exercises/data/code/ex3_spaceship.py
```

Todos fixam `np.random.default_rng(42)`, então os resultados são reproduzíveis.

---

Baseado no [template da disciplina](https://github.com/hsandmann/documentation.template) (MkDocs + Material).
