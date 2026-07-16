import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ============================================================
# Carregar bases
# ============================================================
df_loja = pd.read_csv('data/01_raw/loja.csv')
df_treino = pd.read_csv('data/01_raw/treino.csv', parse_dates=['data'])

TOTAL_DIAS = (df_treino['data'].max() - df_treino['data'].min()).days + 1

# ============================================================
# ITEM 1: 3 lojas com distancia_concorrente null
# ============================================================
print('=' * 70)
print('ITEM 1: INVESTIGACAO DAS 3 LOJAS COM distancia_concorrente NULL')
print('=' * 70)

nulas = df_loja[df_loja['distancia_concorrente'].isnull()].copy()
print(f'Lojas com distancia_concorrente nula: {sorted(nulas["loja"].tolist())}')
print()
print('Cadastro completo das 3 lojas:')
print(nulas.to_string(index=False))
print()

# Puxar metricas de desempenho das 3 lojas do treino.csv
metricas_vendas = df_treino.groupby('loja').agg(
    dias_reg=('data','count'),
    fat_total=('faturamento','sum'),
    cli_total=('clientes','sum'),
    dias_abertos=('aberta','sum'),
    dias_com_promo=('promo','sum'),
)
metricas_vendas['dias_faltantes'] = TOTAL_DIAS - metricas_vendas['dias_reg']
metricas_vendas['vm_abertos'] = np.where(metricas_vendas['dias_abertos'] > 0,
                                         metricas_vendas['fat_total'] / metricas_vendas['dias_abertos'], np.nan)
metricas_vendas['ticket_medio'] = np.where(metricas_vendas['cli_total'] > 0,
                                           metricas_vendas['fat_total'] / metricas_vendas['cli_total'], np.nan)
metricas_vendas['tx_aberto'] = metricas_vendas['dias_abertos'] / TOTAL_DIAS
metricas_vendas['tx_promo'] = np.where(metricas_vendas['dias_abertos'] > 0,
                                       metricas_vendas['dias_com_promo'] / metricas_vendas['dias_abertos'], np.nan)
metricas_vendas = metricas_vendas.reset_index()

# Merge e mostrar
nulas_com_met = nulas.merge(metricas_vendas, on='loja', how='left')
print('Metricas de venda das 3 lojas:')
cols_mostrar = ['loja','tipo_loja','sortimento','distancia_concorrente','promo_continua',
                'dias_reg','dias_faltantes','dias_abertos',
                'fat_total','vm_abertos','cli_total','ticket_medio','tx_aberto','tx_promo']
print(nulas_com_met[cols_mostrar].to_string(index=False))
print()

# Mostrar percentis da distancia_concorrente para contextualizar
pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
estat_dist = df_loja['distancia_concorrente'].describe(percentiles=pcts)
print('Distribuicao global de distancia_concorrente para comparar:')
print(estat_dist.to_string())
print()

# Procurar padrao: tipo_loja, sortimento, promo_continua?
print('As 3 lojas pertencem aos seguintes grupos cadastrais:')
for col in ['tipo_loja','sortimento','promo_continua']:
    print(f'  {col}: {nulas[col].value_counts().sort_index().to_dict()}')

# Ver se as 3 sao do grupo "incompletas" (buraco 2 sem 2014)
print()
incompletas = metricas_vendas[metricas_vendas['dias_faltantes'] > 0]['loja'].unique()
for lj in nulas['loja']:
    status = 'INCOMPLETA' if lj in incompletas else 'COMPLETA'
    print(f'  Loja {int(lj)}: {status} (faltam {int(metricas_vendas.loc[metricas_vendas.loja==lj, "dias_faltantes"].iloc[0])} dias)')

print()
print('Hipotese "distancia muito grande / nao ha concorrente proximo -> veio nulo":')
print(f'  Menor distancia existente na base: R${df_loja["distancia_concorrente"].min():,.0f}')
print(f'  Maior distancia existente na base: R${df_loja["distancia_concorrente"].max():,.0f}')
print(f'  Se nulo = "sem concorrente proximo", qual a alternativa de preenchimento? (Discutir na Fase 3)')

# ============================================================
# ITEM 2: Métricas estatísticas principais das colunas
# ============================================================
print()
print('=' * 70)
print('ITEM 2: METRICAS ESTATISTICAS - loja.csv')
print('=' * 70)

# Numericas
cols_num = df_loja.select_dtypes(include=[np.number]).columns.tolist()
if 'loja' in cols_num:
    cols_num.remove('loja')  # id nao e variavel

print('Colunas NUMERICAS (menos loja id):')
print('-' * 60)
est_num = df_loja[cols_num].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
est_num['missing'] = df_loja[cols_num].isnull().sum()
est_num['missing_pct'] = (df_loja[cols_num].isnull().sum() / len(df_loja) * 100)
print(est_num.round(2).to_string())
print()

# Categoricas
cols_cat = df_loja.select_dtypes(include=['object']).columns.tolist()
print('Colunas CATEGORICAS:')
print('-' * 60)
for c in cols_cat:
    dist = df_loja[c].value_counts(dropna=False).sort_index()
    pct = (dist / len(df_loja) * 100).round(2)
    tbl = pd.DataFrame({'n': dist, '%': pct})
    print(f'--- {c} ---')
    print(tbl.to_string())
    print()

# Binaria promo_continua
print('--- promo_continua (binaria) ---')
dist = df_loja['promo_continua'].value_counts(dropna=False).sort_index()
pct = (dist / len(df_loja) * 100).round(2)
tbl = pd.DataFrame({'n': dist, '%': pct})
print(tbl.to_string())

# ============================================================
# ITEM 3: Outliers e assimetria. Impacto para Kmeans?
# ============================================================
print()
print('=' * 70)
print('ITEM 3: OUTLIERS, ASSIMETRIA E IMPACTO NO KMEANS (distancia)')
print('=' * 70)

# Assimetria e curtose das numericas
skew_kurt = pd.DataFrame({
    'assimetria (skew)': df_loja[cols_num].skew(),
    'curtose (kurtosis)': df_loja[cols_num].kurtosis(),
})
print('Medidas de forma:')
print(skew_kurt.round(3).to_string())
print()

# Para distancia_concorrente: analisar outliers via IQR e Z-score
print('Analise de OUTLIERS em distancia_concorrente:')
x = df_loja['distancia_concorrente'].dropna()
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
n_out_baixo = (x < lower).sum()
n_out_alto = (x > upper).sum()
print(f'  Q1 = {q1:,.0f}  |  Q3 = {q3:,.0f}  |  IQR = {iqr:,.0f}')
print(f'  Limite inferior = {lower:,.0f}  -> outliers baixo: {n_out_baixo}')
print(f'  Limite superior = {upper:,.0f}  -> outliers alto : {n_out_alto} ({n_out_alto/len(x)*100:.2f}%)')
print(f'  MAIOR valor: {x.max():,.0f}  (em relacao a mediana {x.median():,.0f}: {x.max()/x.median():.1f}x maior)')

# Z-score top 10
z = (x - x.mean()) / x.std(ddof=0)
top_out = pd.DataFrame({
    'loja': df_loja.loc[x.index, 'loja'].values,
    'dist': x.values,
    'z-score': z.values,
}).sort_values('z-score', ascending=False).head(10)
print()
print('Top-10 lojas com maior distancia_concorrente (via Z-score):')
print(top_out.to_string(index=False))
print()

# Regra prática: Kmeans sensivel a escala e outliers.
# Simular impacto: a maior distancia vs std da coluna.
print('MAGNITUDE DO IMPACTO NO KMEANS (usando padronizacao z-score):')
print(f'  Menor Z (min dist)  : {z.min():+.2f}')
print(f'  Z na mediana        : {((x.median() - x.mean()) / x.std(ddof=0)):+.2f}')
print(f'  Z no P99            : {((x.quantile(0.99) - x.mean()) / x.std(ddof=0)):+.2f}')
print(f'  Maior Z (max dist)  : {z.max():+.2f}')
print(f'  Max distancia = {x.max():,.0f} tem Z = {z.max():.2f}  -> ~ {int(z.max())} desvios-padrao acima da media')

# Mesma analise para promo_continua (binaria, espera-se pouco problema)
print()
print('Colunas binarias / categoricas (outliers de FREQUENCIA):')
for c in ['promo_continua']:
    dist = df_loja[c].value_counts(dropna=False)
    print(f'  {c}: {dist.to_dict()}')
print()
print('CONCLUSAO PRELIMINAR DO ITEM 3 (antes da Fase 3 correcoes):')
print('  -> distancia_concorrente tem assimetria EXTREMA a direita')
print('     (skew =', f'{skew_kurt.loc["distancia_concorrente","assimetria (skew)"]:.2f},',
      'kurtosis =', f'{skew_kurt.loc["distancia_concorrente","curtose (kurtosis)"]:.2f})')
print('  -> Se entrassem hoje no Kmeans (ainda que padronizado por z-score),')
print('     ~2% de lojas no topo ainda puxariam o centroide com peso desproporcional.')
print('  -> Transformacao recomendada na Fase 3: log1p, rank-Gaussiano, ou winsorize c/ P99.')

# ============================================================
# ITEM 4: Seleção por pergunta de negócio
# ============================================================
print()
print('=' * 70)
print('ITEM 4: SELECAO DE COLUNAS POR PERGUNTA DE NEGOCIO')
print('  Pergunta: como agrupar lojas por comportamento comercial?')
print('=' * 70)
print()
print('Coluna loja -> NAO entra.')
print('  Justificativa (negocio): identificador unico, nao descreve')
print('  comportamento. Id nao "explica" por que duas lojas sao parecidas.')
print()
print('Coluna tipo_loja -> NAO entra no modelo.')
print('  Justificativa (negocio): explicitamente definido no case como')
print('  o "baseline" a superar na avaliacao. Entrar ela no modelo vicia')
print('  a clusterizacao a reproduzir o que a diretoria ja faz hoje, que')
print('  foi justamente o que eles pediram para MELHORAR.')
print()
print('Coluna sortimento -> NAO entra no modelo.')
print('  Justificativa (negocio): mesma regra do tipo_loja. Case diz que')
print('  sortimento tambem serve como baseline. Nosso modelo sera melhor')
print('  que sortimento? Isso so se prova se sortimento NAO for dado ao modelo.')
print()
print('Coluna distancia_concorrente -> ENTRA.')
print('  Justificativa (negocio): descreve a pressao competitiva local,')
print('  que influencia diretamente o comportamento comercial: uma loja sem')
print('  concorrente a 30km define preco/promocao mix de forma muito diferente')
print('  de uma loja com concorrente a 50m. Isso afeta os 4 consumidores do')
print('  resultado: Comercial, Trade Marketing, Supply e Expansao.')
print()
print('Coluna promo_continua -> ENTRA.')
print('  Justificativa (negocio): define se a loja opera em regime permanente')
print('  de promocao (impacta diretamente Trade Marketing na definicao da verba)')
print('  e tambem Comercial, pois loja de promo continua tem meta diferente de')
print('  loja de preco estavel. E uma caracteristica de comportamento operacional')
print('  da loja, nao um mero cadastro.')
print()
print('CONCLUSAO (entrada na modelagem - 2 de 5 colunas do cadastro):')
print('  > ENTRAM: distancia_concorrente, promo_continua')
print('  > SAO BASLINES (NAO ENTRAM): tipo_loja, sortimento')
print('  > E CHAVE TECNICA: loja')
