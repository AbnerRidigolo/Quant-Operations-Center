# Quant Operations Center

Dashboard de análise quantitativa de portfólio com quatro módulos integrados: otimização por HRP, simulação Monte Carlo, modelo fatorial e backtesting histórico. O usuário entra com tickers e um intervalo de datas; o sistema baixa os dados do Yahoo Finance, roda todos os modelos no backend FastAPI e exibe os resultados no frontend Streamlit em tempo real.

---

## O que o projeto faz, na prática

Você monta uma carteira digitando os tickers na sidebar (ex: `AAPL,MSFT,GOOGL,NVDA`) e clica em **RUN FULL ANALYSIS**. O dashboard responde com quatro perspectivas complementares sobre esse portfólio, cada uma representada por uma aba.

---

## Aba 1 — HRP Optimizer: *quanto colocar em cada ativo?*

O primeiro problema de qualquer portfólio é a alocação. A abordagem clássica de Markowitz exige estimar retornos esperados — algo notoriamente difícil. O **Hierarchical Risk Parity** (HRP, Lopez de Prado 2016) resolve isso sem precisar prever retornos: ele olha apenas para o risco.
-
### Gráfico: Weight Allocation (%)

Barras verticais mostrando o percentual alocado em cada ativo. O algoritmo usa a estrutura de correlação para agrupar ativos similares e depois distribui o risco de forma hierárquica: ativos muito correlacionados entre si recebem menos peso coletivamente, evitando concentração disfarçada. O resultado é uma alocação diversificada que nenhum ativo domina — diferente de uma carteira igualmente ponderada, mas também sem os excessos do portfólio de mínima variância.

### Gráfico: Hierarchical Clustering (Dendrograma)

O dendrograma revela *por que* os pesos ficaram como ficaram. Cada galho do diagrama representa uma fusão de ativos: quanto mais alta a junção, menos correlacionados eles são. Ativos que se juntam na base (distância curta) se movem juntos — o algoritmo os trata como um bloco e distribui o risco entre os blocos, não entre ativos individualmente. Ler o dendrograma é entender a estrutura oculta de dependência da carteira.

### Gráfico: Correlation Matrix (Heatmap)

Matriz de calor mostrando a correlação de Pearson entre todos os pares de ativos. Azul ciano intenso = correlação positiva forte (se um sobe, o outro tende a subir). Vermelho = correlação negativa (hedge natural). Diagonal principal sempre igual a 1. Um portfólio saudável tem muitos quadrantes em tom neutro — correlações altas em todo lugar significam diversificação ilusória.

---

## Aba 2 — Monte Carlo: *o que pode acontecer no futuro?*

Com os pesos definidos pelo HRP, o Monte Carlo projeta o portfólio para frente no tempo. O modelo usa **Geometric Brownian Motion** com decomposição de Cholesky para preservar a estrutura de correlação entre os ativos — os choques aleatórios não são independentes, eles refletem como os ativos se movem juntos.

### Gráfico: Simulated Paths (Caminhos Simulados)

Centenas de linhas translúcidas representam trajetórias possíveis do portfólio ao longo do horizonte escolhido (padrão: 252 dias úteis = 1 ano). Sobre o espaguete de simulações, cinco percentis coloridos se destacam:

- **P5 (vermelho)** — cenário pessimista: só 5% das simulações terminam abaixo desta linha.
- **P25 (âmbar)** — cenário conservador.
- **Mediana (verde)** — o resultado mais provável: metade acima, metade abaixo.
- **P75 (âmbar)** — cenário otimista.
- **P95 (ciano)** — cenário touro: só 5% das simulações terminam acima.

A linha horizontal tracejada branca marca o capital inicial. Qualquer trajetória acima dela é lucro; abaixo, prejuízo. A densidade de linhas acima vs. abaixo da linha inicial é o primeiro sinal visual de se o portfólio tem expectativa positiva.

### Gráfico: Final Value Distribution (Histograma)

Distribuição de todos os valores finais das simulações. O histograma conta quantas simulações terminaram em cada faixa de valor. Barras azuis = simulações lucrativas; barras vermelhas = simulações com prejuízo. Duas linhas verticais orientam a leitura: a branca tracejada marca o capital inicial; a vermelha pontilhada marca o **VaR 95%** — o pior resultado esperado com 95% de confiança.

### Tabela: Scenario Analysis

Cinco cenários extraídos dos percentis: Bear (P5), Conservative (P25), Base (Mediana), Optimistic (P75), Bull (P95). Para cada cenário: valor final projetado e retorno percentual sobre o capital inicial. É a tradução do gráfico de simulações em números concretos para tomada de decisão.

---

## Aba 3 — Factor Exposure: *de onde vem o retorno?*

Um portfólio que rendeu 25% ao ano pode ter feito isso simplesmente por estar exposto ao mercado em alta — nenhuma habilidade real, só beta. O modelo fatorial decompõe o retorno do portfólio em exposição a quatro fatores sistemáticos (usando proxies de ETFs) e um alfa residual.

| Fator | Proxy | Interpretação |
|-------|-------|---------------|
| Mkt-RF | SPY − RF | Exposição ao mercado (beta de mercado) |
| SMB | IWM − SPY | Viés small cap vs. large cap |
| HML | IVE − IVW | Viés value vs. growth |
| MOM | MTUM − SPY | Momentum |

### Gráfico: Factor Betas (Barras Horizontais)

Cada barra representa o beta do portfólio naquele fator. Beta positivo em Mkt-RF > 1 significa que o portfólio amplifica os movimentos do mercado. Beta positivo em SMB significa exposição a small caps. Beta negativo em HML indica viés growth. O comprimento e a cor (verde = positivo, vermelho = negativo) traduzem a sensibilidade do portfólio a cada risco sistemático.

### Gráfico: Portfolio vs. Market Returns (Scatter)

Cada ponto é um dia de retorno: eixo X é o retorno excedente do mercado (SPY − RF), eixo Y é o retorno excedente do portfólio. A linha diagonal âmbar é a reta de regressão — sua inclinação é o beta de mercado e seu intercepto (alpha diário) indica o retorno não explicado pelos fatores. Uma nuvem de pontos acima da reta significa que o portfólio superou o mercado com frequência; abaixo, sub-desempenhou.

### Gráfico: Rolling Market Beta (63 dias)

Beta de mercado calculado em janelas móveis de 63 dias úteis (~3 meses). Revela como a sensibilidade do portfólio ao mercado *muda com o tempo*. Um beta crescendo em mercado de baixa é sinal de risco aumentando na hora errada; um beta estável próximo de 1 indica comportamento previsível. A linha de referência β=1 marca neutralidade em relação ao mercado.

---

## Aba 4 — Backtesting: *como teria performado no passado?*

O backtest aplica os pesos HRP sobre os retornos históricos reais do período selecionado e compara a performance com o SPY como benchmark. Todos os cálculos usam retornos diários realizados — sem look-ahead bias.

### Gráfico: Cumulative Performance vs. Benchmark

Retorno acumulado do portfólio HRP (linha ciano com preenchimento suave) versus SPY (linha cinza pontilhada) ao longo de todo o período histórico. Quando a linha ciano está acima da cinza, o portfólio superou o mercado; quando cruza para baixo, ficou atrás. A área preenchida embaixo da curva do portfólio enfatiza a magnitude do crescimento acumulado.

### Gráfico: Portfolio Drawdown

Queda percentual do portfólio em relação ao seu pico histórico em cada momento. A área vermelha representa o "buraco" que o investidor estaria vivenciando — quanto mais funda, mais doloroso o período. O ponto de mínimo é o **Maximum Drawdown**: a pior sequência de perdas do período. Drawdowns longos e profundos testam a disciplina de qualquer estratégia.

### Gráfico: Rolling Sharpe (252 dias)

Índice de Sharpe calculado em janelas móveis de 252 dias (1 ano). Responde à pergunta: *o portfólio estava entregando retorno ajustado a risco consistente, ou foi sorte pontual?* Linha verde = Sharpe acima de 1 (excelente). Âmbar = entre 0 e 1 (razoável). Vermelho = negativo (destruição de valor ajustado a risco). Períodos prolongados abaixo de zero indicam que a estratégia precisaria ser revisada.

### Tabela: Portfolio vs. Benchmark Breakdown

Comparação linha a linha de 11 métricas: retorno anualizado, volatilidade, Sharpe, Sortino, Max Drawdown, Calmar Ratio, VaR diário, Win Rate (% de dias positivos), melhor dia, pior dia e retorno total. Permite identificar exatamente em que dimensão o portfólio HRP ganhou ou perdeu para o benchmark passivo.

---

## Stack Técnica

| Camada | Tecnologia |
|--------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Dados | yfinance (Yahoo Finance) |
| Otimização | SciPy (clustering hierárquico) |
| Fator Model | scikit-learn (OLS) |
| Simulação | NumPy (Cholesky GBM) |
| Deploy | Docker Compose |

## Como rodar

```bash
# Com Docker
docker compose up --build

# Sem Docker (dois terminais)
cd backend && uvicorn main:app --reload --port 8000
cd frontend && streamlit run app.py --server.port 8501
```

Frontend: http://localhost:8501 · API docs: http://localhost:8000/docs
