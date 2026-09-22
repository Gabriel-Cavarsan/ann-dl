"""Perceptron de camada única, escrito do zero.

Este módulo é usado pelos dois exercícios sem nenhuma alteração: o Exercício 2
importa exatamente a mesma função de treino do Exercício 1.
"""

import numpy as np


def degrau(z):
    """Ativação do perceptron: 1 se z >= 0, senão 0."""
    return (z >= 0).astype(int)


def prever(X, w, b):
    """Predição para todas as amostras de X de uma vez."""
    return degrau(X @ w + b)


def acuracia(X, y, w, b):
    """Fração de acertos no dataset inteiro."""
    return (prever(X, w, b) == y).mean()


def treinar(X, y, w_inicial, taxa=0.01, max_epocas=100):
    """Treina o perceptron amostra a amostra e devolve o histórico do treino.

    A cada amostra calcula o erro (y - y_chapeu), que vale 0 quando a predição
    está correta, +1 num falso negativo e -1 num falso positivo. Só os erros
    movem os pesos.

    O "pocket" guarda os melhores pesos já vistos: sempre que uma atualização
    deixa a acurácia acima da melhor até então, copiamos (w, b) para o bolso.
    Em dados separáveis o bolso termina igual aos pesos finais; em dados
    sobrepostos os dois se separam, e é essa diferença que o Exercício 2 estuda.
    """
    w = w_inicial.copy()
    b = 0.0

    melhor_w, melhor_b = w.copy(), b
    melhor_acc = acuracia(X, y, w, b)
    melhor_epoca = 0

    historico_acc = []       # acurácia dos pesos atuais ao fim de cada época
    historico_melhor = []    # melhor acurácia vista até cada época
    atualizacoes_por_epoca = []

    for epoca in range(1, max_epocas + 1):
        atualizacoes = 0
        for i in range(len(X)):
            erro = y[i] - degrau(X[i] @ w + b)
            if erro != 0:
                w = w + taxa * erro * X[i]
                b = b + taxa * erro
                atualizacoes += 1

                acc = acuracia(X, y, w, b)
                if acc > melhor_acc:
                    melhor_w, melhor_b = w.copy(), b
                    melhor_acc, melhor_epoca = acc, epoca

        atualizacoes_por_epoca.append(atualizacoes)
        historico_acc.append(acuracia(X, y, w, b))
        historico_melhor.append(melhor_acc)

        if atualizacoes == 0:      # uma passagem inteira sem erro: convergiu
            break

    return {
        "w": w, "b": b,
        "acuracia": acuracia(X, y, w, b),
        "epocas": epoca,
        "pocket_w": melhor_w, "pocket_b": melhor_b,
        "pocket_acuracia": melhor_acc, "pocket_epoca": melhor_epoca,
        "historico_acc": historico_acc,
        "historico_melhor": historico_melhor,
        "atualizacoes_por_epoca": atualizacoes_por_epoca,
    }


def gerar_duas_classes(rng, media_0, media_1, variancia, n_por_classe=1000):
    """Duas nuvens gaussianas 2D com covariância diagonal igual para as duas."""
    cov = np.array([[variancia, 0.0], [0.0, variancia]])
    X = np.vstack([rng.multivariate_normal(media_0, cov, n_por_classe),
                   rng.multivariate_normal(media_1, cov, n_por_classe)])
    y = np.repeat([0, 1], n_por_classe)
    return X, y
