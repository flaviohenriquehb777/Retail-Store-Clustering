import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid', palette='viridis')

# ============================================================
# Carregar tabela de lojas ja preparada
# ============================================================
df = pd.read_csv('data/03_processed/tabela_por_loja_fase3.csv')

# ============================================================
# ITEM 1: Assimetria de todas as colunas numericas (ordenada)
# ============================================================
print('=' * 70)
print('ITEM 1: ASSIMETRIA (SKEW) DE TODAS AS COLUNAS NUMERICAS (ORDENADA)')
print('=' * 70)

# Pegar colunas numericas, excluir id e flags binarias de auditoria
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
excluir = ['loja']  # mantemos as binarias para medir tambem o skew delas
cols_num = [c for c in cols_num if c not in excluir]

skew_df = pd.DataFrame({
    'coluna': cols_num,
    'assimetria': df[cols_num].skew().round(4).values,
    'curtose': df[cols_num].kurtosis().round(4).values,
}).set_index('coluna')

skew_df['abs_skew'] = skew_df['assimetria'].abs()
skew_df = skew_df.sort_values('abs_skew', ascending=False).drop(columns=['abs_skew'])

# Classificacao de severidade
def classificacao(s):
    if abs(s) < 0.5: return 'baixa (simetrica ok)'
    elif abs(s) < 1.0: return 'moderada'
    elif abs(s) < 2.0: return 'ALTA'
    else: return 'EXTREMA'
skew_df['classe'] = skew_df['assimetria'].map(classificacao)

print(skew_df.to_string())

# ============================================================
# ITEM 2: Histograma das 3 colunas mais representativas dos problemas
# ============================================================
print()
print('=' * 70)
print('ITEM 2: HISTOGRAMAS DAS 3 COLUNAS MAIS PROBLEMATICAS')
print('=' * 70)

# Selecionar 3 colunas MAIS problemáticas, de classes distintas de skew para ter vizualizacoes diferentes:
# 1. tx_domingos_abertos: bimodal, skew extremo por massa em 0
# 2. distancia_concorrente (sem tratamento ainda, ou seja a coluna original quebrada de skew): skew extrema clássica de cauda direita pesada
# 3. razao_pico_sabado ou uplift_promo_pct: escolhemos a que tem skew mais alto entre as restantes, excluindo binarias
# Para manter variedade de tipos de problema:
top3_visual = []
# Primeiro, pegar colunas com classes EXTREMA, desconsiderando binarias puras
binarias_puras = [c for c in cols_num if set(df[c].dropna().unique()).issubset({0,1})]
cols_nao_bin = [c for c in cols_num if c not in binarias_puras]
skew_naobin = skew_df.reindex(cols_nao_bin).sort_values('assimetria', key=lambda s: s.abs(), ascending=False)
print('Top colunas NAO binarias por skew (para escolher top-3 figuras):')
print(skew_naobin[['assimetria','classe']].head(6).to_string())

# Escolhas para 3 figuras (cada uma representa um tipo diferente de problema):
col_fig_a = 'tx_domingos_abertos'  # TIPO: bimodal / massa de zeros + rabo direito
col_fig_b = 'distancia_concorrente'  # TIPO: skew cauda direta (dist original)
col_fig_c = 'clientes_medio_dia'  # TIPO: skew + cauda, mas nao tao exagerado (permite ver log1p efeito)
# Confirmar que existem
for c in [col_fig_a, col_fig_b, col_fig_c]:
    assert c in df.columns, f'{c} nao encontrada'
print()
print(f'Figura 1: {col_fig_a} - bimodal massa de zeros')
print(f'Figura 2: {col_fig_b} - cauda direta extrema')
print(f'Figura 3: {col_fig_c} - skew moderado/alto mas cauda razoavel')

# Plotar os 3 histogramas em grid
fig, axes = plt.subplots(3, 1, figsize=(11, 12))
for ax, c in zip(axes, [col_fig_a, col_fig_b, col_fig_c]):
    sns.histplot(data=df, x=c, bins=50, kde=True, color='darkslateblue', ax=ax)
    s_val = skew_df.loc[c, 'assimetria']
    k_val = skew_df.loc[c, 'curtose']
    ax.axvline(df[c].mean(), color='crimson', linestyle='--', linewidth=2, label=f'Media = {df[c].mean():.2f}')
    ax.axvline(df[c].median(), color='forestgreen', linestyle='-', linewidth=2, label=f'Mediana = {df[c].median():.2f}')
    ax.set_title(f'Histograma: {c}   |   skew = {s_val:+.3f}  |   kurt = {k_val:+.2f}', fontsize=12)
    ax.set_xlabel(c)
    ax.set_ylabel('Frequencia')
    ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
out_fig = 'reports/figures/fase3_histogramas_top3_skew.png'
fig.savefig(out_fig, dpi=140, bbox_inches='tight')
plt.close(fig)
print()
print(f'Figura salva em: {out_fig}')

# BONUS: histograma das colunas ja tratadas vs nao tratadas (distancia_concorrente vs log1p)
fig2, ax2 = plt.subplots(figsize=(11, 4.5))
c1 = 'distancia_concorrente'
c2 = 'distancia_concorrente_log1p'
sns.histplot(data=df, x=c1, bins=50, kde=True, color='darkred', alpha=0.4, label=c1, ax=ax2, stat='density')
sns.histplot(data=df, x=c2, bins=50, kde=True, color='navy', alpha=0.4, label=c2, ax=ax2, stat='density')
ax2.set_title('Efeito do log1p: distancia_concorrente'
              f' (skew {skew_df.loc[c1,"assimetria"]:+.2f})'
              f' vs {c2} (skew {skew_df.loc[c2,"assimetria"]:+.2f})')
ax2.legend()
fig2.savefig('reports/figures/fase3_efeito_log1p_distancia.png', dpi=140, bbox_inches='tight')
plt.close(fig2)
print('Figura bonus (comparativo log1p distancia) salva em: reports/figures/fase3_efeito_log1p_distancia.png')

# ============================================================
# ITEM 3 + 4: Diagnostico por coluna (escala vs forma) + tratamento
# ============================================================
print()
print('=' * 70)
print('ITEM 3 + 4: DIAGNOSTICO POR COLUNA (escala vs forma) E TRATAMENTO')
print('=' * 70)

# Vamos pegar colunas NAO binarias auditoria e com skew >= 1.0 em modulo (ALTA ou EXTREMA)
cols_alvos = skew_df[skew_df['classe'].isin(['ALTA','EXTREMA'])].index.tolist()

# Definicao extra: para cada coluna, calcular razao max/mediana e P99/P50 para ajudar a diferenciar escala vs forma
linhas = []
for c in cols_alvos:
    x = df[c]
    mediana = x.median()
    if mediana == 0:
        p99_p50 = np.inf
        max_med = np.inf
    else:
        p99_p50 = x.quantile(0.99) / mediana
        max_med = x.max() / mediana
    # Regras de bolso
    skew_abs = abs(skew_df.loc[c, 'assimetria'])
    n_unique = x.nunique()
    # Heuristica: se bimodal (massa grande em 0 + valores nao zero), problema É DE FORMA
    qtd_zeros = (x == 0).sum()
    prop_zero = qtd_zeros / len(x)
    binaria_like = n_unique <= 2 or prop_zero > 0.8 or (x.eq(0).any() and (x > 0).sum() < len(x) * 0.15)

    if binaria_like:
        problema = 'FORMA (bimodal: massa de zeros + grupo raro de valores positivos)'
        diferenca = 'Nao e "mesma coisa com tamanhos diferentes". Sao DUAS populacoes no mesmo atributo: lojas da categoria 0 vs lojas da categoria != 0. A assimetria vem da distribuicao DESIGUAL entre os grupos, nao da amplitude numerica.'
        tratamento = 'BIFURCAR em DUAS features: (i) binaria "tem_valor_nao_zero?" (ex: abre_domingo = 1 sim/nao) para capturar a existencia, e (ii) continua condicional (ex: tx_domingos_abertos dado que abre domingo de vez em quando) para a intensidade DENTRO do grupo nao-zero. Se a categoria nao-zero for pequena (<= ~15% dos dados), a feature continua original SEMPRE vai dominar a distancia euclidiana apenas pelo fato de "loja que abre domingo ja e diferente do resto".'
    else:
        # Desagregar entre escala vs forma:
        # Se p99/p50 < 4 e max/p99 < 2 (mesmo com skew alto) -> problema de ESCALA
        # Se p99/p50 > 8 ou max/p99 > 1.8 -> problema de FORMA (cauda pesada em cima dos outliers)
        # Geralmente os dois aparecem juntos
        tem_forma = (p99_p50 >= 6) or (max_med / max(p99_p50, 1e-9) >= 1.5)  # max >> p99
        tem_escala = (p99_p50 >= 3) and skew_abs >= 1.0
        if tem_forma and tem_escala:
            problema = 'ESCALA + FORMA (ambos).'
            diferenca = 'Ambos: os valores ocupam uma faixa de amplitude muito grande (P99/P50 >= 6x -> ESCALA), e alem disso tem outliers extremos que fazem o max ser >> P99, criando uma cauda pesada nao-Gaussiana (FORMA).'
        elif tem_forma:
            problema = 'FORMA (cauda pesada, outliers top desproporcionais)'
            diferenca = 'Depois de corrigir a faixa de valores ainda restam pontos extremos. O problema nao e "valores sao grandes", mas sim "a cauda nao cai normalmente".'
        else:
            problema = 'ESCALA (faixa muito larga, mas sem outliers absurdos)'
            diferenca = 'A distribuicao tem assimetria mas o max e razoavelmente proximo do P99. O problema e a diferenca absoluta de escala entre lojas pequenas e grandes, nao outliers isolados.'

        # Tratamento
        if problema.startswith('ESCALA + FORMA'):
            tratamento = 'log1p(x) primeiro para comprimir escala e achatar a cauda (trata ambos juntos). Se apos log1p o skew ainda ficar > |1|, aplicar rank-Gaussiano (transforma em ranks 1..N depois mapeia para quantis normais). Como passo final de seguranca, winsorize a P1/P99 da variavel transformada (ou seja, corta valores extremos residuais).'
        elif problema.startswith('FORMA'):
            tratamento = 'rank-Gaussiano direto (transformacao monotona que achata qualquer cauda pesada para distribuição normal). Nao perde a ordem relativa entre lojas e elimina robustamente outliers.'
        else:  # so escala
            tratamento = 'log1p(x) ou StandardScaler apos log1p. Winsorize opcional so se houver suspeita de erro de medicao no topo.'

    linhas.append({
        'coluna': c,
        'skew': f'{skew_df.loc[c,"assimetria"]:+.3f}',
        'classe': skew_df.loc[c,'classe'],
        'n_valores_unicos': n_unique,
        'P99 / mediana': f'{p99_p50:.1f}x' if not np.isinf(p99_p50) else 'inf',
        'max / mediana': f'{max_med:.1f}x' if not np.isinf(max_med) else 'inf',
        '%_zeros': f'{prop_zero*100:.1f}%',
        'tipo_problema': problema,
        'qual_a_diferenca?': diferenca,
        'tratamento_recomendado': tratamento,
    })

diagnostico = pd.DataFrame(linhas)
print(diagnostico.to_string(index=False))
print()
print('NOTA SOBRE A DIFERENCA ENTRE ESCALA E FORMA:')
print('  ESCALA = problema de AMPLITUDE. Mesma forma de distribuicao, mas os numeros sao')
print('           muito grandes em modulo. Ex: distancia de 20m vs 75.000m. StandardScaler sozinho')
print('           resolve escala mas NAO resolve assimetria.')
print('  FORMA  = problema de DISTRIBUICAO. Ex: 95% das lojas em 0-10.000m e 3 lojas em 40-75km puxando')
print('           a cauda de forma nao-gaussiana. StandardScaler NAO resolve forma.')
print('  No KMeans (distancia euclidiana), escala "estica" um eixo desproporcionalmente; forma')
print('  cria poucos pontos que puxam o centroide na direcao deles. Juntos, matam qualquer clusterizacao.')

# ============================================================
# ITEM 5: tx_domingos_abertos - tratamento detalhado
# ============================================================
print()
print('=' * 70)
print('ITEM 5: TRATAMENTO DEDICADO DA tx_domingos_abertos (bimodal/binaria-like)')
print('=' * 70)

c = 'tx_domingos_abertos'
total = len(df)
zeros = (df[c] == 0).sum()
naozero = total - zeros
cut_abre = 0.50
perto100 = (df[c] >= cut_abre).sum()
print(f'Total lojas                  : {total}')
print(f'Com tx = 0 (NUNCA abre domingo): {zeros}  ({zeros/total*100:.2f}%)')
print(f'Com tx > 0  (alguma vez abriu) : {naozero}  ({naozero/total*100:.2f}%)')
print(f'Com tx >= 50% (abre frequente) : {perto100}  ({perto100/total*100:.2f}%)')
print(f'Com tx = 100% (abre SEMPRE)    : {(df[c] == 1).sum()}  ({(df[c]==1).sum()/total*100:.2f}%)')
print()
print('Tratamento recomendado para coluna tx_domingos_abertos:')
print('  (a) CRIAR coluna binaria: abre_domingo = 1 se tx >= 50%, senao 0.')
print('      Essa captura o "tipo de loja" (operacao 7 dias vs 6 dias). E binaria e resolve a bimodalidade')
print('      em grupos claros para o KMeans (distancia entre os grupos vai ser pequena no calculo,)')
print('  (b) CRIAR coluna continua intensidade_abrindo_domingo = tx_domingos_abertos APENAS nas lojas')
print('      com abre_domingo=1. Nas lojas abre_domingo=0, preencher com a MEDIANA DA INTENSIDADE')
print('      do grupo que abre domingo. NAO preencher com 0! Preencher com 0 zeraria a distancia entre a')
print('      loja que NUNCA abre e a loja que abre as vezes: a mediana do grupo mantem a representacao')
print('      neutra ("igual a media do outro grupo") e evita distorcao.')
print('  (c) NAO aplicar log1p em tx_domingos_abertos bruto. Ja e 0..1, nao tem problema de escala.')
print('      O problema e de forma (bimodal), que log1p nao corrige.')
print('  (d) O mesmo racional se aplica a QUALQUER outra feature que aparecer no futuro')
print('      com >80% de massa em um unico valor discreto + rabo pequeno no outro lado.')

# Salvar tabela auxiliar de skew e diagnostico para fase de feature engineering
skew_df.reset_index().to_csv('reports/figures/fase3_assimetrias_por_coluna.csv', index=False, encoding='utf-8')
diagnostico.to_csv('reports/figures/fase3_diagnostico_tratamentos.csv', index=False, encoding='utf-8')
print()
print(f'Tabelas auxiliares salvas: skew.csv + diagnostico.csv em reports/figures/')
