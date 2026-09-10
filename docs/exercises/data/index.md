---
exercise: data
ai_use: "Claude (Opus 5) atuou como par de programação: escrita dos três scripts em code/, geração das figuras e redação do relatório, a partir do enunciado e de decisões discutidas comigo. Revisei o código linha a linha e sei explicar cada etapa. Todos os números citados vêm da execução dos scripts versionados nesta pasta."
---

# Data — Preparação e Análise de Dados para Redes Neurais

Preparação e análise de dados para redes neurais. O fio condutor é o **espalhamento**: quanto uma nuvem de pontos se abre, em que direção, e como isso muda a dificuldade da classificação.

!!! note "Reprodutibilidade"
    Cada script abre com `rng = np.random.default_rng(42)` e usa esse mesmo gerador do início ao fim. Nenhum modelo é treinado — tudo aqui é geometria e preparo de dados. Bibliotecas: `numpy`, `pandas`, `matplotlib` e `scikit-learn` (só PCA e pré-processamento).

    Os três scripts estão em `code/` e rodam a partir de um clone limpo, com `pip install -r requirements.txt`. O `train.csv` do Spaceship Titanic está versionado em `dataset/`.

---

## Exercício 1

**Nuvens de Pontos: Geometria e Espalhamento em 2D**

Abordagem: gerar as quatro nuvens com uma função só, parametrizada pelo fator de escala, e medir a separação de duas formas independentes — uma analítica (a razão $r_{ij}$, que só olha para os parâmetros) e uma empírica (a taxa de mistura, que conta pontos). As duas contam a mesma história, e é essa concordância que sustenta a análise do item C.

### A — Gere as nuvens

400 amostras, 100 por classe, cada classe uma gaussiana com desvio padrão independente por eixo:

| Classe | Média | Desvio padrão | $\bar\sigma$ |
|---|---|---|---|
| 0 | [2, 3] | [0,8 ; 2,5] | 1,65 |
| 1 | [5, 6] | [1,2 ; 1,9] | 1,55 |
| 2 | [8, 1] | [0,9 ; 0,9] | 0,90 |
| 3 | [15, 4] | [0,5 ; 2,0] | 1,25 |

![Figura 1](figures/fig1.png)

A geometria já se lê no gráfico: a classe 2 é a mais compacta ($\bar\sigma = 0{,}90$) e a classe 0 é a mais alongada na vertical ($\sigma_y = 2{,}5$, mais de três vezes o seu $\sigma_x$). As classes 0 e 1 são vizinhas próximas; a classe 3 está isolada à direita.

### B — Mais ou menos espalhado

As mesmas quatro classes, geradas quatro vezes, com todos os desvios multiplicados por $s \in \{0{,}5\ ;\ 1{,}0\ ;\ 2{,}0\ ;\ 4{,}0\}$. **As médias nunca mudam** — só o espalhamento.

![Figura 2](figures/fig2.png)

Os quatro subplots compartilham os mesmos limites de eixo, então a comparação é honesta: o que cresce na imagem é a nuvem, não a escala.

#### Razão de separação em $s = 1$

$$r_{ij} = \frac{\|\mu_i - \mu_j\|}{\bar\sigma_i + \bar\sigma_j}, \qquad \bar\sigma_k = \frac{\sigma_{k,x} + \sigma_{k,y}}{2}$$

| Par $(i,j)$ | Distância entre centros | $\bar\sigma_i + \bar\sigma_j$ | $r_{ij}$ |
|---|---|---|---|
| (0, 1) | 4,243 | 3,20 | **1,326** ← menor |
| (0, 2) | 6,325 | 2,55 | 2,480 |
| (0, 3) | 13,038 | 2,90 | 4,496 |
| (1, 2) | 5,831 | 2,45 | 2,380 |
| (1, 3) | 10,198 | 2,80 | 3,642 |
| (2, 3) | 7,616 | 2,15 | 3,542 |

O menor é o par **(0, 1)**, com $r_{01} = 1{,}326$ — as duas nuvens vizinhas do canto esquerdo. Como as médias são fixas e só os desvios escalam por $s$, o numerador de $r_{ij}$ não muda e o denominador é multiplicado por $s$: portanto $r_{ij}(s) = r_{ij}(1)/s$. Em $s = 2$, sem gerar nada de novo, $r_{01} = 1{,}326 / 2 = \mathbf{0{,}663}$.

#### Taxa de mistura

Fração de pontos cujo centro de classe mais próximo não é o da própria classe — uma medida puramente geométrica, sem treinar nada.

| $s$ | Taxa de mistura | Pontos mal atribuídos |
|---|---|---|
| 0,5 | 0,0000 | 0 de 400 |
| 1,0 | 0,0675 | 27 de 400 |
| 2,0 | 0,2250 | 90 de 400 |
| 4,0 | 0,4175 | 167 de 400 |

![Figura 3](figures/fig3.png)

**A partir de qual fator de escala as nuvens deixam de poder ser separadas por retas?** A partir de **$s = 2$**. O critério não é visual, é numérico: em $s = 2$ o menor par cruza o limiar $r_{ij} = 1$, caindo para $r_{01} = 0{,}663$. Quando $r_{ij} < 1$, a distância entre os centros passa a ser *menor* que a soma dos espalhamentos típicos das duas nuvens — elas se interpenetram, e não existe reta que as separe. A taxa de mistura confirma: salta de 6,75% em $s = 1$ para 22,5% em $s = 2$, e em $s = 4$ chega a 41,75%, não muito longe dos 75% que se obteria chutando entre quatro classes.

Em $s = 0{,}5$ e $s = 1$ as retas ainda servem: $r_{01}$ vale 2,652 e 1,326, ambos acima de 1.

### C — Análise

**Sobreposição em $s = 1$ e separabilidade linear.** A sobreposição existe mas não é uniforme: os 6,75% de pontos mal atribuídos concentram-se quase todos no par (0, 1), que é o único com $r_{ij}$ abaixo de 1,5. A classe 3 está isolada (seu menor $r$ é 3,542) e a classe 2 é compacta e bem destacada.

Uma **única** fronteira linear não separa as quatro classes — uma reta divide o plano em apenas duas regiões, e precisamos de quatro. São necessárias no mínimo três retas. Já um **conjunto** de fronteiras lineares resolve quase tudo: é exatamente o que uma rede com uma camada escondida constrói, combinando vários semiplanos em regiões poligonais.

**Esboço das fronteiras.**

![Figura 1b](figures/fig1b.png)

Esbocei as fronteiras do **centro mais próximo** (as mediatrizes entre cada par de médias, que formam um diagrama de Voronoi) por um motivo deliberado: é precisamente a regra geométrica que a taxa de mistura do item B mede. Assim o desenho e o número falam da mesma coisa — os 27 pontos mal atribuídos são visivelmente os que caem do lado errado dessas retas, quase todos azuis invadindo o território laranja. São três segmentos de reta, coerente com o mínimo teórico apontado acima.

**Espalhamento e a região de erro inevitável.** Aqui está o ponto que conecta o esboço ao item B: **as fronteiras não se movem quando $s$ cresce**. Elas dependem apenas das médias, que são fixas por construção. O que muda é a densidade de pontos em torno delas.

Conforme $s$ aumenta, cada nuvem se derrama sobre o território das vizinhas, e a faixa em que as densidades de duas classes se cruzam engorda. Nessa faixa, pontos das duas classes ocupam literalmente as mesmas coordenadas — nenhuma fronteira, linear ou não, pode acertar os dois. É o **erro de Bayes** do problema: um piso de erro imposto pelos dados, não pela arquitetura. Trocar o modelo por um mais expressivo não ajuda; o que ajudaria seria uma feature nova capaz de distinguir os pontos sobrepostos. A progressão 0% → 6,75% → 22,5% → 41,75% é esse piso subindo.

---

## Exercício 2

**Não-Linearidade em Dimensões Maiores**

Abordagem: construir dois datasets com a *mesma* dimensão (5D) e o *mesmo* tamanho (500 por classe), mudando apenas a geometria — um separável por deslocamento de médias, outro separável só por raio. Depois medir os dois com as mesmas três ferramentas (PCA, distância entre centros, histograma de raio) e deixar o contraste aparecer.

!!! note "Suposição declarada"
    O enunciado escreve o raio como $\rho \sim N(2{,}0\ ;\ 0{,}4)$. Interpretei o segundo parâmetro como **desvio padrão** (não variância), que é a convenção de `rng.normal(loc, scale)` do NumPy. Com $\sigma = 0{,}4$, as duas cascas ficam a 7,5 desvios uma da outra, o que é consistente com o enunciado ao antecipar que os histogramas de raio ficam bem separados.

### A — Dataset I: gaussianas deslocadas

500 amostras por classe, geradas com `rng.multivariate_normal` a partir das médias e matrizes de covariância especificadas. A classe A tem $\mu_A = \mathbf{0}$ e correlação **positiva** (0,8) entre as duas primeiras features; a classe B tem $\mu_B = [1{,}5]^5$, variâncias maiores (1,5 na diagonal) e correlação **negativa** (−0,7) nas mesmas features. As duas classes diferem, portanto, em posição *e* em espalhamento.

### B — Dataset II: cascas concêntricas

Também 500 por classe, também em 5D, mas com estrutura radial. Cada ponto é $x = \rho \cdot u$, com direção e raio sorteados de forma independente:

- **direção**: $v \sim N(0, I_5)$, depois $u = v / \|v\|$. Normalizar leva $u$ para a esfera unitária; como a gaussiana isotrópica não tem direção preferida, $u$ fica uniforme sobre a esfera;
- **raio**: $\rho \sim N(2{,}0\ ;\ 0{,}4)$ para a classe C (núcleo) e $\rho \sim N(5{,}0\ ;\ 0{,}4)$ para a classe D (casca).

Raios medidos nos dados gerados: **C = 1,972 ± 0,396** e **D = 5,005 ± 0,409**, próximos dos valores nominais.

### C — Visualize e compare

![Figura 4](figures/fig4.png)

| | Dataset I | Dataset II |
|---|---|---|
| Distância entre os centros em 5D | **3,2282** | **0,2662** |
| Variância explicada por PC1 | 50,0% | 21,6% |
| Variância explicada por PC2 | 15,9% | 21,3% |
| **PC1 + PC2** | **66,0%** | **42,9%** |

**Em qual dataset a projeção 2D preserva melhor a informação relevante para a classificação?** No **Dataset I**, e com folga. Ali PC1 sozinho captura metade da variância e se alinha com a direção que separa as classes — o deslocamento entre as médias, de comprimento 3,23. Na Figura 4 as duas classes aparecem escalonadas ao longo do eixo horizontal.

No Dataset II a variância é praticamente isotrópica: 21,6% e 21,3% são quase exatamente $1/5$ cada, o que é o que se espera de uma nuvem sem direção preferencial em 5 dimensões. Não há direção privilegiada para a PCA encontrar, e o resultado é o painel da direita — o núcleo azul afogado no meio da casca vermelha.

![Figura 5](figures/fig5.png)

O histograma de raio inverte completamente o veredito: os suportes das duas classes **nem se tocam**. A informação que a PCA não conseguiu mostrar estava lá o tempo todo, só que codificada de forma não-linear.

### D — Análise

**Centros coincidentes + raios separados: o que isso diz sobre hiperplanos?** Diz que nenhum hiperplano funciona. Um hiperplano decide por $w \cdot x + b$, ou seja, projeta cada ponto sobre uma única direção $w$ e aplica um limiar. Se os dois centros praticamente coincidem (0,266 contra raios médios de 1,97 e 5,00), então para **qualquer** escolha de $w$ vale $w \cdot \mu_C \approx w \cdot \mu_D$: as duas projeções ficam centradas no mesmo ponto. Como as duas nuvens são simétricas em torno da origem, a projeção da casca D é apenas mais larga que a do núcleo C, e ambas centradas em zero — uma envolvendo a outra sobre a reta. Nenhum limiar separa duas distribuições aninhadas assim.

O raio, por outro lado, separa perfeitamente, porque $\|x\|$ não é uma função linear de $x$.

**Por que mais dados não resolvem.** Porque o obstáculo é topológico, não estatístico. A classe C ocupa a região $\|x\| \approx 2$ e está **inteiramente contida** na bola que a classe D delimita — está envolvida por ela em todas as direções. Um semiespaço (o que um hiperplano produz) é convexo e ilimitado: qualquer semiespaço que contenha todo o núcleo C necessariamente contém também a metade da casca D que está daquele lado. Coletar mais amostras apenas preenche melhor as mesmas duas cascas; a relação de "uma dentro da outra" não muda. É o obstáculo do XOR em versão contínua — e a saída é a mesma: uma transformação não-linear das entradas.

**Uma projeção 2D ruim prova inseparabilidade?** Não, e o Dataset II é o contraexemplo, com os meus próprios números. Na Figura 4 as classes C e D parecem irremediavelmente misturadas, com PC1 + PC2 retendo só 42,9% da variância. Ainda assim, a função

$$f(x) = \|x\|^2 = \sum_{i=1}^{5} x_i^2, \qquad \text{classifique como D se } f(x) > 3{,}5^2 = 12{,}25$$

separa o Dataset II com **100,00% de acerto** nos 1000 pontos — sem treinar nada, com um limiar escolhido a olho no meio do caminho entre os raios 2 e 5.

A lição é sobre o que a PCA é: uma transformação **linear**, que busca direções de máxima variância. Ela só consegue exibir uma separação que já esteja disponível em alguma combinação linear das coordenadas. No Dataset II essa separação não existe em nenhuma direção — ela mora no raio, uma função quadrática. Uma projeção linear que mistura as classes prova apenas que **aquela família de transformações** não as separa, e nada além disso. É exatamente esse o espaço em que redes profundas trabalham: aprender a transformação não-linear que torna os dados separáveis.

---

## Exercício 3

**Preparando Dados do Mundo Real para uma Rede Neural**

Dataset: **Spaceship Titanic** (Kaggle), arquivo `train.csv`, 8.693 linhas e 14 colunas, versionado em `dataset/`. O alvo é uma rede neural com ativação **tanh** nas camadas ocultas, o que condiciona as decisões de escala.

Abordagem: seguir a ordem *separar → ajustar no treino → aplicar no teste*, sem exceção. Todos os objetos que aprendem algo dos dados (imputadores, codificador, escalador) recebem `.fit()` apenas em `X_treino`.

### A — Conheça os dados

**Objetivo.** A coluna `Transported` é o alvo binário: indica se o passageiro foi transportado para outra dimensão durante a colisão da nave com a anomalia espaço-temporal. A tarefa é classificação binária.

**Balanceamento.** Praticamente perfeito — **50,36% `True`** contra 49,64% `False`. Não há problema de desbalanceamento aqui: acurácia é uma métrica utilizável e não há motivo para reponderar classes ou aplicar SMOTE.

**Tipos de feature.**

| Tipo | Colunas |
|---|---|
| Numéricas | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |
| Categóricas | `HomePlanet`, `CryoSleep`, `Destination`, `VIP` |
| Identificadores / texto | `PassengerId`, `Cabin`, `Name` |
| Alvo | `Transported` |

**Valores faltantes.** Todas as 12 colunas de feature têm faltantes, e em proporção notavelmente uniforme — entre 2,06% e 2,50%. Essa uniformidade sugere ausência aleatória (perfil MCAR), sem relação com o valor que falta, o que torna a imputação simples uma escolha segura.

| Coluna | Faltantes | % |
|---|---|---|
| `CryoSleep` | 217 | 2,50 |
| `ShoppingMall` | 208 | 2,39 |
| `VIP` | 203 | 2,34 |
| `HomePlanet` | 201 | 2,31 |
| `Name` | 200 | 2,30 |
| `Cabin` | 199 | 2,29 |
| `VRDeck` | 188 | 2,16 |
| `FoodCourt` | 183 | 2,11 |
| `Spa` | 183 | 2,11 |
| `Destination` | 182 | 2,09 |
| `RoomService` | 181 | 2,08 |
| `Age` | 179 | 2,06 |

**Colunas de gasto.**

| Coluna | Média | Mediana | Máximo |
|---|---|---|---|
| `RoomService` | 224,69 | 0,00 | 14.327 |
| `FoodCourt` | 458,08 | 0,00 | 29.813 |
| `ShoppingMall` | 173,73 | 0,00 | 23.492 |
| `Spa` | 311,14 | 0,00 | 22.408 |
| `VRDeck` | 304,85 | 0,00 | 24.133 |

**O que a diferença média × mediana indica.** A mediana é **zero em todas as cinco colunas**, enquanto as médias vão de 174 a 458. Isso é o retrato de uma distribuição extremamente assimétrica à direita: mais da metade dos passageiros não gastou absolutamente nada em cada serviço, e a média é inteiramente puxada por uma minoria de gastadores pesados — o máximo de `FoodCourt`, 29.813, está a 65 vezes a própria média. O espalhamento não é simétrico em torno de um centro; é uma massa concentrada em zero com uma cauda longuíssima. Duas consequências práticas, que voltam nos itens C e D: imputar pela média inventaria gastos altos em passageiros dos quais nada se sabe, e escalar linearmente esmagaria a maioria dos pontos num canto do intervalo.

### B — Separe antes de transformar

Split **80/20 estratificado** por `Transported`, com `random_state=42`:

- treino: **6.954 linhas**, proporção positiva 0,5036
- teste: **1.739 linhas**, proporção positiva 0,5037

A estratificação preservou o balanceamento nas duas partes até a quarta casa decimal.

**Por que a separação vem antes da imputação e do escalonamento.** Porque toda transformação usada aqui *aprende parâmetros a partir dos dados*: o imputador aprende uma mediana, o escalador aprende um mínimo e um máximo, o codificador aprende a lista de categorias. Se esses parâmetros forem estimados sobre o dataset inteiro, informação do conjunto de teste entra na transformação que depois será aplicada ao próprio teste — ele deixa de ser dado inédito e a performance reportada fica otimista, sem que nada no código pareça errado.

Este relatório tem a **prova numérica** disso no item C: como ajustei o `MinMaxScaler` apenas no treino, o teste terminou com máximo **1,1383**, fora do intervalo [−1, 1]. Se eu tivesse ajustado no dataset completo, treino e teste cairiam ambos exatamente em [−1, 1] — e essa aparência de perfeição seria justamente o sintoma do vazamento.

### C — Pré-processe

**1. Dados faltantes.** Estratégia por tipo de coluna, com `SimpleImputer` ajustado no treino:

- **numéricas → mediana.** Justificativa direta do item A: com mediana 0 e média nas centenas, imputar pela média atribuiria um gasto alto a um passageiro sobre o qual não temos informação nenhuma, criando um dado falso e distorcendo a cauda. A mediana é a hipótese conservadora e representa o que a maioria de fato fez. `Age` usa mediana pelo mesmo critério de robustez;
- **categóricas → moda** (`most_frequent`). Não existe "média" de `Europa`; a categoria mais frequente é o palpite de menor risco. Com apenas ~2,3% de faltantes, o efeito sobre a distribuição é desprezível.

**2. Features categóricas → one-hot.** `HomePlanet`, `CryoSleep`, `Destination` e `VIP` convertidas com `OneHotEncoder`. Escolhi one-hot em vez de codificação inteira porque nenhuma dessas variáveis é ordinal — atribuir 0, 1, 2 a `Earth`, `Europa`, `Mars` faria a rede assumir que Mars é "o dobro" de Europa e que Europa está "entre" os outros dois, uma ordem que não existe. A cardinalidade é baixa (2 ou 3 categorias), então o custo em número de colunas é mínimo.

**Categoria vista no teste mas ausente do treino.** Tratada por `handle_unknown="ignore"`: uma categoria inédita produz uma linha de zeros em todas as dummies daquela feature, em vez de levantar exceção no `transform`. Verifiquei explicitamente e, neste split, **nenhuma** das quatro colunas apresenta categoria nova no teste. A proteção fica no código de qualquer forma, porque é uma garantia estrutural do pipeline — não uma sorte desta semente.

**3. Engenharia de features.** `TotalSpend` = soma das cinco colunas de gasto, calculada **antes** da transformação logarítmica. A ordem importa: $\log(a) + \log(b) \neq \log(a+b)$, e o que interessa como feature é o gasto total real do passageiro. Descartei `PassengerId` (identificador puro, sem sinal), `Name` (texto de cardinalidade quase igual ao número de linhas) e `Cabin` (código composto `deck/número/lado`; extrair deck e lado seria possível e provavelmente útil, mas foge do escopo desta atividade).

**4. Cauda pesada: $\log(1+x)$.** Aplicado às cinco colunas de gasto e a `TotalSpend`.

Por que isso ajuda uma rede com tanh: a tanh é aproximadamente linear perto de zero e satura nas pontas — $\tanh(3) = 0{,}995$, e a derivada ali já é praticamente nula. Sem a transformação, ao escalar `FoodCourt` linearmente para [−1, 1] usando o máximo de 29.813, mais de metade dos passageiros (os de gasto zero) ficaria exatamente em −1 e a grande maioria dos demais a poucos milésimos dali. A rede veria quase todos os passageiros como o **mesmo ponto de entrada**, produziria ativações quase idênticas e receberia gradientes minúsculos: não haveria como aprender a distinguir quem gastou 50 de quem gastou 500. O $\log(1+x)$ comprime a cauda e redistribui a massa ao longo do intervalo útil, como a Figura 6 mostra.

Uso $\log(1+x)$ e não $\log(x)$ porque zero é o valor mais comum nessas colunas e $\log(0)$ é indefinido; o deslocamento de 1 mapeia $0 \mapsto 0$ e preserva o significado do "não gastou nada".

**5. Escalonamento.** `MinMaxScaler(feature_range=(-1, 1))` nas sete colunas numéricas (`Age`, os cinco gastos e `TotalSpend`).

Escolhi **normalização para [−1, 1]** em vez de padronização por uma razão específica a esta arquitetura: o intervalo alvo é exatamente a imagem da tanh. Com padronização (média 0, desvio 1), os valores extremos cairiam a 4 ou 5 desvios da média — região onde $\tanh$ já saturou e o gradiente morreu. Com [−1, 1], todo o conjunto de treino fica garantidamente na faixa de maior derivada da ativação, onde o aprendizado é mais eficiente. A objeção usual ao min-max é sua sensibilidade a outliers, mas o $\log(1+x)$ do passo anterior já domou a cauda, o que remove a maior parte do problema.

Valores resultantes: treino **[−1,0000 ; 1,0000]** (exato, por construção do scaler) e teste **[−1,0000 ; 1,1383]**.

### D — Verifique e visualize

![Figura 6](figures/fig6.png)

O painel esquerdo mostra `FoodCourt` bruto no treino: uma barra dominante em zero e uma cauda que se arrasta até quase 30.000 (eixo vertical em escala log, sem o qual nada além da primeira barra seria visível). O painel direito mostra a mesma coluna após $\log(1+x)$ e escala para [−1, 1]: a massa dos que não gastaram continua concentrada em −1, mas todo o restante dos passageiros agora ocupa o intervalo de forma distribuída, com um corpo bem definido entre −0,7 e 1,0. É essa segunda distribuição que uma rede com tanh consegue usar.

**Checagens finais.**

| Verificação | Resultado |
|---|---|
| NaN remanescentes | **0** no treino, **0** no teste |
| Shape final — treino | **(6954, 17)** |
| Shape final — teste | **(1739, 17)** |
| Intervalo do treino | [−1,0000 ; 1,0000] |
| Intervalo do teste | [−1,0000 ; 1,1383] |
| Compatível com tanh | **Sim** |

As 17 features são 7 numéricas (`Age`, os cinco gastos, `TotalSpend`) mais 10 colunas one-hot: `HomePlanet` (3), `CryoSleep` (2), `Destination` (3) e `VIP` (2).

Sobre o intervalo do teste ultrapassar 1: isso é **esperado e correto**. Significa que o teste contém ao menos um passageiro cujo gasto excede o máximo observado no treino. O escalador não foi ajustado nele e portanto não sabia desse valor — que é exatamente a situação de um modelo em produção diante de um dado novo. O excesso é pequeno (13,8% acima do limite) e mantém as entradas na faixa onde a tanh ainda tem derivada apreciável, então não há necessidade de recorte.

**Quais decisões mais afetariam o treinamento, e por quê.** O $\log(1+x)$, sem competição. É a única transformação do pipeline que muda a **forma** da distribuição, e não apenas sua posição e escala. A imputação toca só ~2% das linhas. A escolha entre padronização e min-max, por mais que eu a tenha justificado acima, é uma transformação afim — a primeira camada da rede poderia em princípio reaprendê-la ajustando pesos e viés, ainda que isso custe épocas. O $\log(1+x)$ não é afim: nenhuma camada linear o reproduz, então ele entrega à rede uma representação que ela não conseguiria construir sozinha sem gastar capacidade e profundidade. Somado a isso, é ele que impede a saturação da tanh — é a diferença entre a rede enxergar 90% dos passageiros como um ponto só e enxergá-los espalhados. Em segundo lugar eu colocaria o descarte de `Cabin`: é uma perda de informação assumida conscientemente, já que o deck e o lado da nave plausivelmente carregam sinal sobre quem foi transportado.

---

## Resumo dos resultados

| # | Item | Seu valor |
|---|---|---|
| 1 | Taxa de mistura em $s = 0{,}5$ | **0,0000** (0 de 400) |
| 2 | Taxa de mistura em $s = 1{,}0$ | **0,0675** (27 de 400) |
| 3 | Taxa de mistura em $s = 2{,}0$ | **0,2250** (90 de 400) |
| 4 | Taxa de mistura em $s = 4{,}0$ | **0,4175** (167 de 400) |
| 5 | Menor $r_{ij}$ em $s = 1{,}0$ e qual é o par | **1,326** — par **(0, 1)** |
| 6 | Distância entre os centros — Dataset I | **3,2282** |
| 7 | Distância entre os centros — Dataset II | **0,2662** |
| 8 | Variância explicada PC1 + PC2 — Dataset I | **66,0%** (50,0% + 15,9%) |
| 9 | Variância explicada PC1 + PC2 — Dataset II | **42,9%** (21,6% + 21,3%) |
| 10 | Proporção da classe positiva em `Transported` | **0,5036** (50,36%) |
| 11 | Média e mediana de `FoodCourt` no treino, antes de transformar | média **452,61** / mediana **0,00** |
| 12 | `shape` final da matriz de features de treino | **(6954, 17)** |
| 13 | Mínimo e máximo do treino e do teste após o escalonamento | treino **[−1,0000 ; 1,0000]** / teste **[−1,0000 ; 1,1383]** |

---

## Código

### `code/ex1_nuvens.py`

```python
--8<-- "docs/exercises/data/code/ex1_nuvens.py"
```

### `code/ex2_dimensoes.py`

```python
--8<-- "docs/exercises/data/code/ex2_dimensoes.py"
```

### `code/ex3_spaceship.py`

```python
--8<-- "docs/exercises/data/code/ex3_spaceship.py"
```
