# Relatório de Análise Estatística: Ruído Sísmico & COVID-19

## 1. Introdução

### Descrição do Problema
Este estudo analisa a variação do ruído sísmico de alta frequência durante o período de lockdown da COVID-19. O foco é identificar se houve uma redução estatisticamente significativa ("Global Quieting").

### Processamento dos Dados
*   **Dados Brutos**: Séries de RMS (Root Mean Square) do deslocamento na banda 4-20Hz.
*   **Agregação Temporal**: Cálculo da **Mediana Diária** para cada estação (minimizando efeito de transientes).
*   **Normalização**: Variação percentual (%) em relação à Baseline (Mediana dos dias Pre-Lockdown).
*   **Estações Analisadas**: 145
*   **Total de Dias-Estação**: 27,432
*   **Integração de Mobilidade**: Dados de mobilidade do Google (Community Mobility Reports) integrados.

### Natureza das Variáveis
*   **Variável Independente**: Status de Mobilidade (Proxy: Data de Início do Lockdown, binária Pre/Post).
*   **Variável Dependente**: Variação Percentual do Ruído (%) (Contínua).
*   **Covariáveis**: Densidade Populacional, Índices de Mobilidade Google.

### 1.1. Quadro Metodológico dos Testes Estatísticos
Abaixo detalhamos a bateria de testes estatísticos aplicados neste estudo, seus objetivos e a justificativa para sua escolha, garantindo rigor e robustez metodológica.

| Teste Estatístico | Objetivo | Justificativa de Utilização |
| :--- | :--- | :--- |
| **Shapiro-Wilk** | Verificar Normalidade | Teste padrão para verificar a distribuição dos dados. Fundamental para orientar a decisão inicial entre testes paramétricos e não-paramétricos. |
| **Kolmogorov-Smirnov (KS)** | Verificar Normalidade | Utilizado como confirmação do Shapiro-Wilk em grandes amostras ($N > 5000$). Valida a aderência à distribuição normal com alta sensibilidade. |
| **Correlação de Pearson** ($r$) | Medir Relação Linear | Padrão da indústria para quantificar a força e direção da relação linear entre o Ruído Sísmico e a Mobilidade. |
| **Correlação de Spearman** ($\rho$) | Medir Relação Monotônica | **Validação de Robustez**: Alternativa não-paramétrica ao Pearson. Utilizado para confirmar a correlação independentemente da normalidade dos dados, provando que a relação não é um artefato de outliers. |
| **Teste T (1 Amostra)** | Comparar Médias | Verifica se a redução média do ruído é estatisticamente diferente de 0. Válido para grandes amostras devido ao **Teorema Central do Limite (CLT)**, que garante a normalidade da média amostral. |
| **Teste de Wilcoxon** | Comparar Medianas | **Validação de Robustez**: Equivalente não-paramétrico do Teste T. Utilizado para confirmar a significância da redução ("quieting") sem depender de premissas sobre a distribuição dos dados. |

## 2. Probabilidade

### Teste de Normalidade
A variável analisada neste teste é a **Variação Percentual do Ruído** (`noise_change_pct`), que representa o desvio diário do nível de ruído (RMS) em relação à baseline pré-lockdown.

Foram aplicados testes de **Shapiro-Wilk** e **Kolmogorov-Smirnov (KS)** na distribuição desta variável. O teste KS foi realizado nos dados padronizados (Z-scores).

**Definição das Hipóteses:**
*   **Hipótese Nula ($H_0$):** A distribuição dos dados segue uma distribuição Normal.
*   **Hipótese Alternativa ($H_1$):** A distribuição dos dados **não** segue uma distribuição Normal.
*   **Nível de Significância ($\alpha$):** 0.05
*   **Critério de Decisão:** Rejeitar $H_0$ se o $p$-value $< \alpha$.

### Teste de Normalidade (Por População)
A análise foi segregada em duas populações distintas para verificar se o comportamento da distribuição muda entre os períodos:
1.  **Pré-Lockdown**: Período de referência.
2.  **Lockdown**: Período de restrição.

**Resultados dos Testes (Shapiro-Wilk & KS):**

| População | N (Amostra) | Shapiro-Wilk (W) | P-Valor (SW) | Kolmogorov-Smirnov (D) | P-Valor (KS) | Resultado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pré-Lockdown** | 4000 | 0.0154 | < 0.001 | 0.4835 | < 0.001 | **Não Normal** |
| **Lockdown** | 4000 | 0.7904 | < 0.001 | 0.1444 | < 0.001 | **Não Normal** |

**Análise Visual:**
O histograma comparativo abaixo mostra claramente a mudança de regime.
*   **Pré-Lockdown (Cinza)**: Centrado em 0 (Baseline), com distribuição mais estreita.
*   **Lockdown (Vermelho)**: Deslocamento massivo para a esquerda (negativo), indicando a redução de ruído.

![Comparação das Distribuições](distribution_comparison.png)
*> **Nota**: Ambos os períodos falham no teste de normalidade, mas a diferença visual nas médias e variâncias confirma o impacto do lockdown.*

## 3. Estatísticas Descritivas

## 3. Comparative Statistics (Hypothesis Testing)

We performed a **1-Sample T-Test** to determine if the noise levels during the lockdown significantly deviated from the baseline (0).

### Hypothesis Definition
*   **Null Hypothesis ($H_0$):** The mean percentage change in seismic noise is equal to 0 ($\mu = 0$).
*   **Alternative Hypothesis ($H_1$):** The mean percentage change in seismic noise is different from 0 ($\mu \neq 0$).
*   **Significance Level ($\alpha$):** 0.05

### Test Results
*   **Degrees of Freedom:** 8271
*   **Calculated T-Statistic:** -61.5024
*   **Critical T-Value:** +/- 1.9603
*   **P-Value (T-Test):** 0.0000e+00

### Conclusion & Validation
**Reject H0**.

**Nota sobre o Tamanho da Amostra:**
O $N=8272$ refere-se exclusivamente aos dias monitorados durante o período de **Lockdown**. O total mencionado na introdução (27,432) engloba todo o período do estudo (incluindo a fase Pré-Lockdown usada para calcular a baseline). Testamos aqui se o subconjunto "Lockdown" difere de zero.

**Justificativa Metodológica:**
Embora os dados não sigam uma distribuição normal (conforme Seção 2), o **Teste T** é válido devido ao grande tamanho da amostra (N=8272), o que garante a normalidade da distribuição das médias pelo **Teorema Central do Limite (CLT)**.

**Prova Visual do CLT:**
Para validar a afirmação acima, realizamos uma simulação de Bootstrap com 1000 amostras. O histograma abaixo mostra a distribuição das médias amostrais. Observe como ela se ajusta perfeitamente à curva normal (linha vermelha), validando o pressuposto do Teste T.

![Prova Visual do CLT](clt_means_distribution.png)

### Validação Não-Paramétrica: Teste de Wilcoxon (Signed-Rank)

Dada a não-normalidade do conjunto de dados original, o **Teste de Wilcoxon** serve como a validação mais robusta e credível.

**Definição das Hipóteses:**
*   **$H_0$:** Mediana das diferenças = 0.
*   **$H_1$:** Mediana das diferenças $\neq$ 0.

**Tabela de Postos (Ranks):**
A tabela abaixo detalha a soma dos postos das diferenças, permitindo visualizar a magnitude da redução.

| Categoria | Descrição | N (Observações) | Soma dos Postos |
| :--- | :--- | :--- | :--- |
| **Postos Negativos ($W_-$)** | Dias com Redução de Ruído (Noise < Baseline) | 6844 | 31006302 |
| **Postos Positivos ($W_+$)** | Dias com Aumento de Ruído (Noise > Baseline) | 1428 | 3210826 |
| **Empates** | Dias sem alteração | 0 | - |
| **Total** | | 8527 | 34217128 |

**Estatística do Teste (O que é comparado?):**
*   **Estatística Z (Calculada):** -63.9852
*   **Valor Crítico Z (para $\alpha=0.05$):** 1.96
*   **P-Value:** 0.0000e+00

**Critério de Decisão:**
Para grandes amostras, a estatística $W$ do Wilcoxon converge para uma distribuição Normal (Z).
*   **Regra:** Rejeita-se $H_0$ se $|Z_{calc}| > 1.96$ (ou se $p < 0.05$).
*   **Comparação:** Como $|-63.9852| > 1.96$, rejeitamos a hipótese de que a mediana das diferenças é zero.

**Conclusão Final:**
A esmagadora preponderância de **Postos Negativos** resulta em um Z-score extremamente significativo. A rejeição de $H_0$ confirma que o silenciamento sísmico foi um fenômeno real e sistemático.

![Boxplot Daily RMS](boxplot_daily_rms.png)

## 4. Análise de Mobilidade (Google Mobility Data)

Integração dos dados de mobilidade do Google para verificar correlações com a redução sísmica.

### 4.1. Correlação (Ruído vs Mobilidade)
Calculamos a correlação de duas formas para capturar a tendência temporal:
### 4.1. Correlação (Ruído vs Mobilidade)
Calculamos a correlação de duas formas para capturar a tendência temporal:
1.  **Global Temporal (Smoothed)**: Correlação entre a mediana diária global do ruído e a mediana diária da mobilidade (suavizada por média móvel de 7 dias).
2.  **Mediana Local**: Mediana das correlações calculadas individualmente para cada estação.

**Justificativa Metodológica (Média Móvel de 7 Dias):**
A aplicação de uma janela de 7 dias é fundamental para remover a periodicidade semanal intrínseca (efeito de finais de semana) presente tanto no ruído sísmico cultural quanto nos dados de mobilidade. Isso atua como um filtro passa-baixa, isolando a **tendência de longo prazo** associada ao lockdown e removendo oscilações de alta frequência que poderiam mascarar a correlação real entre o 'quieting' e a mudança de comportamento social. Esta abordagem segue a metodologia padrão da literatura (e.g., Lecocq et al., 2020) para análise de tendências em séries temporais antropogênicas.

**Teste de Hipótese para Correlação de Pearson:**
*   **Hipótese Nula ($H_0$):** Não há correlação linear ($\rho = 0$).
*   **Hipótese Alternativa ($H_1$):** Existe correlação linear significativa ($\rho \neq 0$).
*   **Decisão:** Rejeitar $H_0$ se $|T_{calc}| > T_{crit}$.

**Detalhamento da Validação por Spearman (O que é comparado?):**
A correlação de Spearman é um teste não-paramétrico que avalia a relação monotônica.
*   **Mecânica:** Os valores brutos de Ruído e Mobilidade são convertidos em **Postos (Ranks)** (ex: o menor valor recebe posto 1, o segundo 2, etc.).
*   **Comparação:** O teste calcula a correlação de Pearson entre esses **ranks**, não entre os valores absolutos.
*   **Hipótese Nula ($H_0$):** Não há associação monotônica entre as variáveis (os ranks são independentes).
*   **Decisão (P-Value):** Como os p-valores de Spearman na tabela abaixo são extremamente baixos (muito menores que $\alpha=0.05$), rejeitamos $H_0$, confirmando que a relação é robusta e não depende da distribuição normal dos dados.

**Tabela de Correlações:**
    | Categoria de Mobilidade   |   R (Pearson) |   T-Calculado (Corr) |   Graus de Liberdade |   T-Crítico |   P-Valor (Global) | Significativo?   |   Rho (Spearman) |   R (Mediana Estações) |
|:--------------------------|--------------:|---------------------:|---------------------:|------------:|-------------------:|:-----------------|-----------------:|-----------------------:|
| Varejo e Lazer            |      0.990824 |             75.8321  |                  107 |     1.98238 |       7.75412e-95  | Sim              |         0.983706 |               0.591979 |
| Mercado e Farmácia        |      0.967439 |             39.5378  |                  107 |     1.98238 |       1.13165e-65  | Sim              |         0.95034  |               0.575524 |
| Parques                   |      0.589058 |              7.54032 |                  107 |     1.98238 |       1.61104e-11  | Sim              |         0.894328 |               0.468846 |
| Estações de Transporte    |      0.985578 |             60.2463  |                  107 |     1.98238 |       2.17193e-84  | Sim              |         0.976681 |               0.590114 |
| Locais de Trabalho        |      0.994904 |            102.072   |                  107 |     1.98238 |       1.86213e-108 | Sim              |         0.983201 |               0.559554 |
| Residencial               |     -0.988634 |            -68.0203  |                  107 |     1.98238 |       6.91206e-90  | Sim              |        -0.977795 |              -0.519659 |

### 4.2. Visualização Comparativa
O gráfico abaixo apresenta a sobreposição das curvas suavizadas (7 dias) do Ruído Sísmico Global e das principais métricas de mobilidade. Note o alinhamento quase perfeito entre a queda da mobilidade (linhas coloridas) e o quieting sísmico (linha preta).

![Comparação Ruído vs Mobilidade](noise_mobility_timeseries.png)

O scatter plot abaixo reforça esta relação linear forte (R² próximo de 1.0 para Varejo e Transporte), mostrando que a redução proporcional da atividade humana explica quase totalmente a redução do nível de ruído sísmico.

![Scatter Plot: Mobilidade vs Ruído](scatter_noise_vs_mobility.png)

### 4.3. Tabela de Distribuição de Frequências Relativas
A tabela abaixo compara a distribuição das variações percentuais. Os valores representam a **Frequência Relativa** (proporção de dias) em cada intervalo de 10%. Isso permite comparar diretamente a distribuição estatística do "Quieting" com a queda de mobilidade.

|    | index       |   Ruído Global |    Varejo |   Transporte |   Residencial |
|---:|:------------|---------------:|----------:|-------------:|--------------:|
|  0 | -100 a -90% |      0         | 0         |    0         |      0        |
|  1 | -90 a -80%  |      0         | 0         |    0         |      0        |
|  2 | -80 a -70%  |      0         | 0.146789  |    0         |      0        |
|  3 | -70 a -60%  |      0.0642202 | 0.238532  |    0.321101  |      0        |
|  4 | -60 a -50%  |      0.266055  | 0.12844   |    0.302752  |      0        |
|  5 | -50 a -40%  |      0.110092  | 0.110092  |    0.0366972 |      0        |
|  6 | -40 a -30%  |      0.183486  | 0.0458716 |    0.0366972 |      0        |
|  7 | -30 a -20%  |      0.0550459 | 0.0733945 |    0.0642202 |      0        |
|  8 | -20 a -10%  |      0.0733945 | 0.0366972 |    0.0275229 |      0        |
|  9 | -10 a 0%    |      0.0183486 | 0.100917  |    0.0917431 |      0        |
| 10 | 0 a 10%     |      0.0275229 | 0.119266  |    0.119266  |      0.302752 |
| 11 | 10 a 20%    |      0.201835  | 0         |    0         |      0.40367  |
| 12 | 20 a 30%    |      0         | 0         |    0         |      0.293578 |
| 13 | 30 a 40%    |      0         | 0         |    0         |      0        |
| 14 | 40 a 50%    |      0         | 0         |    0         |      0        |
| 15 | 50 a 60%    |      0         | 0         |    0         |      0        |
| 16 | Total       |      1         | 1         |    1         |      1        |

### 4.4. Confirmação de Alteração na Mobilidade (Teste de Wilcoxon)

Para confirmar se a pandemia causou uma alteração estatisticamente significativa nos padrões de mobilidade (e não apenas flutuações aleatórias), aplicamos o **Teste de Wilcoxon (Signed-Rank)** detalhado.

**Definição das Hipóteses:**
*   **Hipótese Nula ($H_0$):** A mediana da variação da mobilidade é igual a 0 (Padrão Normal).
*   **Hipótese Alternativa ($H_1$):** A mediana da variação da mobilidade é diferente 0 (Alteração Comportamental).
*   **Nível de Significância ($lpha$):** 0.05

**Tabela de Detalhamento dos Postos (Evidência da Alteração):**
A tabela abaixo exibe a **Soma dos Postos** positivos ($W_+$) e negativos ($W_-$). A gigantesca discrepância entre eles comprova a direção e a magnitude da mudança.

| Categoria              |   Mediana (%) |    N |   W+ (Aumento) |   W- (Queda) |   Z-Score | Significativo?   |
|:-----------------------|--------------:|-----:|---------------:|-------------:|----------:|:-----------------|
| Varejo e Lazer         |           -40 | 8572 |        1771868 |     34947478 |    -72.45 | Sim              |
| Mercado e Farmácia     |            -8 | 8545 |        7509972 |     28965685 |    -47.12 | Sim              |
| Parques                |           -11 | 8572 |        9983363 |     26750504 |    -36.61 | Sim              |
| Estações de Transporte |           -47 | 8572 |        1811131 |     34917696 |    -72.28 | Sim              |
| Locais de Trabalho     |           -34 | 8572 |        3076010 |     33642668 |    -66.76 | Sim              |
| Residencial            |            14 | 8527 |       34832400 |      1227577 |     73.26 | Sim              |

**Interpretação dos Resultados:**
*   **Varejo, Transporte e Trabalho**: A soma dos postos negativos ($W_-$) é massivamente superior à dos positivos ($W_+$), resultando em Z-scores altamente negativos. Isso confirma uma **redução drástica e sistemática**.
*   **Residencial**: O padrão se inverte ($W_+ \gg W_-$), com um Z-score positivo altíssimo, confirmando o **aumento significativo** da permanência em casa (o "ficar em casa").
*   Todos os p-valores são extremos (< 0.001), rejeitando $H_0$ categoricamente. **A mobilidade humana não apenas mudou; ela sofreu um choque estrutural.**

## 5. Análise de Regressão

Comparamos dois modelos para explicar a variação do ruído sísmico ("Quieting"): 
1.  **Densidade Populacional (Estático)**: Explica variações espaciais baseadas na ocupação fixa.
2.  **Mobilidade Urbana - Varejo e Lazer (Dinâmico)**: Explica variações temporais baseadas no comportamento humano durante a pandemia.

### Modelo 1: Densidade Populacional (Estático)
*   **Variável Explicativa:** Densidade Populacional (hab/km²).
*   **Equação:** $y = 0.0470x + -64.6873$
*   **R² (Coeficiente de Determinação):** 0.0022
*   **Interpretação:** A densidade populacional explica apenas **0.22%** da variância na redução de ruído. Isso indica que apenas estar em uma área densa não garantiu maior silenciamento; a mudança de comportamento foi o fator chave.

### Modelo 2: Queda de Mobilidade - Varejo e Lazer (Dinâmico)
Utilizando os dados de mobilidade do Google ("Retail and Recreation") como preditor da redução de ruído.

*   **Variável Explicativa:** Variação de Mobilidade em Varejo e Lazer (Tendência Suavizada 7 dias).
*   **Equação:** $y = 0.9819x + 11.5065$
*   **R² (Coeficiente de Determinação):** 0.9840
*   **Interpretação:** A variação na mobilidade explica **98.40%** da variação na tendência do ruído sísmico. 
*   **Conclusão:** Este modelo é imensamente superior ao demográfico, provando que a **dinâmica da atividade humana** (e não apenas a presença estática de pessoas) é o driver direto do ruído sísmico de alta frequência.

Ver gráfico comparativo: `regression_pop_vs_drop.png` (Modelo 1). O Modelo 2 é visualizado no Scatter Plot da Seção 4.2.

## 6. Análise Espacial

O mapa `spatial_map_change.png` ilustra a variação percentual do ruído em cada estação. Cores azuis indicam redução (quieting), enquanto vermelhos indicam aumento.

## 7. Conclusão

A análise refinada (Mediana Diária) confirma que houve uma alteração no campo de ondas sísmicas global.
*   A variação média global foi de **-60.6%**.
*   A estatística prova que o "silêncio" sísmico foi um fenômeno real e mensurável.
*   **Mobilidade**: Observou-se correlação significativa com métricas de mobilidade urbana (Retail, Transit, Workplaces).

---
**Arquivos Gerados**:
*   `processed_data.pkl`: Dados de Mediana Diária.
*   `Statistical_Analysis_Report.md`: Este relatório.
*   Figuras: `distribution_change_pct.png`, `boxplot_daily_rms.png`, `regression_pop_vs_drop.png`, `spatial_map_change.png`.
