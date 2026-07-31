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

CODIGO_DOMINGO = 7
CODIGO_SABADO = 6

PERIODO_BURACO_INI = pd.Timestamp('2014-07-01')
PERIODO_BURACO_FIM = pd.Timestamp('2014-12-31')

print('=' * 70)
print('FASE 3 - PREPARACAO DOS DADOS')
print('=' * 70)

# ============================================================
# Passo 0: Filtrar dias realmente operando (aberta=1 & faturamento>0)
# ============================================================
linhas_antes = len(df_treino)
mask_operando = (df_treino['aberta'] == 1) & (df_treino['faturamento'] > 0)
df_op = df_treino[mask_operando].copy()
linhas_depois = len(df_op)
print()
print('PASSO 0: Filtrar dias de operacao real (aberta=1 E faturamento>0)')
print(f'  Linhas antes: {linhas_antes:>10,}')
print(f'  Linhas apos : {linhas_depois:>10,}')
print(f'  Linhas descartadas: {(linhas_antes - linhas_depois):>10,} '
      f'({(linhas_antes - linhas_depois)/linhas_antes*100:.3f}%)')
print()

# ============================================================
# BLOCO 1: Porte e modelo de negocio
# ============================================================
print('=' * 70)
print('BLOCO 1: PORTE E MODELO DE NEGOCIO')
print('=' * 70)

# ticket MEDIO por loja: calculado como soma_total_fat / soma_total_cli SOBRE OS DIAS OPERANDO
# Motivo: ticket ponderado pelo numero de clientes. Se usarmos media(fat/cli por dia), damos o mesmo peso
# para um dia de domingo com 2 clientes e um dia de black friday com 5.000 clientes - erro grave.
# Com a soma agregada, temos o ticket medio real por cliente atendido no periodo todo.
bloco1 = df_op.groupby('loja').agg(
    total_fat_bloco1_denominador=('faturamento', 'sum'),
    total_cli_bloco1_denominador=('clientes', 'sum'),
    dias_operando=('data', 'count'),
).reset_index()
bloco1['venda_media_dia'] = np.where(
    bloco1['dias_operando'] > 0,
    bloco1['total_fat_bloco1_denominador'] / bloco1['dias_operando'], np.nan
)
bloco1['clientes_medio_dia'] = np.where(
    bloco1['dias_operando'] > 0,
    bloco1['total_cli_bloco1_denominador'] / bloco1['dias_operando'], np.nan
)
bloco1['ticket_medio'] = np.where(
    bloco1['total_cli_bloco1_denominador'] > 0,
    bloco1['total_fat_bloco1_denominador'] / bloco1['total_cli_bloco1_denominador'], np.nan
)
bloco1_saida = bloco1[['loja','venda_media_dia','clientes_medio_dia','ticket_medio']].copy()

print('Calculo ticket_medio: sum(faturamento em dias operando) / sum(clientes em dias operando)')
print('Motivo: ticket ponderado por qtd de clientes real (cada cliente tem peso 1). Media(fat/dia / cli/dia)')
print('iria distorcer dando peso igual a dias com muito ou pouco fluxo.')
print()
print('Describe do BLOCO 1:')
print(bloco1_saida.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(2).to_string())

# ============================================================
# BLOCO 2: Dependencia de promocao
# ============================================================
print()
print('=' * 70)
print('BLOCO 2: DEPENDENCIA DE PROMOCAO (RAZAO PROMO / SEM PROMO)')
print('=' * 70)

por_loja_promo = df_op.groupby('loja').agg(
    fat_com_promo=('faturamento', lambda s: s[df_op.loc[s.index, 'promo'] == 1].sum()),
    dias_promo=('promo', lambda s: (s == 1).sum()),
    fat_sem_promo=('faturamento', lambda s: s[df_op.loc[s.index, 'promo'] == 0].sum()),
    dias_sem_promo=('promo', lambda s: (s == 0).sum()),
).reset_index()
por_loja_promo['vm_com_promo'] = np.where(por_loja_promo['dias_promo'] > 0,
                                          por_loja_promo['fat_com_promo'] / por_loja_promo['dias_promo'], np.nan)
por_loja_promo['vm_sem_promo'] = np.where(por_loja_promo['dias_sem_promo'] > 0,
                                          por_loja_promo['fat_sem_promo'] / por_loja_promo['dias_sem_promo'], np.nan)
por_loja_promo['razao_promo'] = np.where(
    (por_loja_promo['vm_sem_promo'] > 0),
    por_loja_promo['vm_com_promo'] / por_loja_promo['vm_sem_promo'], np.nan
)
por_loja_promo['uplift_promo_pct'] = (por_loja_promo['razao_promo'] - 1.0) * 100

bloco2_saida = por_loja_promo[['loja','razao_promo','uplift_promo_pct']].copy()
lojas_sem_promo = por_loja_promo['dias_promo'] == 0
lojas_sem_nenhum_sem_promo = por_loja_promo['dias_sem_promo'] == 0
print(f'Lojas que NUNCA tiveram dia com promo (operando): {lojas_sem_promo.sum()}')
print(f'Lojas que SEMPRE tiveram promo (todos os dias operando): {lojas_sem_nenhum_sem_promo.sum()}')
print()
print('Faixa de variacao da medida razao_promo na rede:')
print(f'  min    : {por_loja_promo["razao_promo"].min():.3f}x (uplift de {(por_loja_promo["razao_promo"].min()-1)*100:+.2f}%)')
print(f'  P1     : {por_loja_promo["razao_promo"].quantile(0.01):.3f}x')
print(f'  P5     : {por_loja_promo["razao_promo"].quantile(0.05):.3f}x')
print(f'  P25    : {por_loja_promo["razao_promo"].quantile(0.25):.3f}x')
print(f'  mediana: {por_loja_promo["razao_promo"].median():.3f}x  ({(por_loja_promo["razao_promo"].median()-1)*100:+.2f}%)')
print(f'  P75    : {por_loja_promo["razao_promo"].quantile(0.75):.3f}x')
print(f'  P95    : {por_loja_promo["razao_promo"].quantile(0.95):.3f}x')
print(f'  P99    : {por_loja_promo["razao_promo"].quantile(0.99):.3f}x')
print(f'  max    : {por_loja_promo["razao_promo"].max():.3f}x ({(por_loja_promo["razao_promo"].max()-1)*100:+.2f}%)')
print()
print('Describe do BLOCO 2:')
print(bloco2_saida.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(2).to_string())

# ============================================================
# BLOCO 3: Volatilidade dia-a-dia e sazonalidade mensal
# Aplicar mitigacao do buraco: excluir 01/07/2014 a 31/12/2014 para TODAS as lojas
# ============================================================
print()
print('=' * 70)
print('BLOCO 3: VOLATILIDADE DIA-A-DIA E SAZONALIDADE MENSAL (ADIMENSIONAIS)')
print('Mitigacao buraco reformadas: excluir periodo 01/jul a 31/dez 2014 para TODAS')
print('=' * 70)

mask_b3 = ~((df_op['data'] >= PERIODO_BURACO_INI) & (df_op['data'] <= PERIODO_BURACO_FIM))
df_b3 = df_op[mask_b3].copy()
print(f'Dias operando validos para BLOCO 3 (fora do buraco): {len(df_b3):,}')
print()

# Medida 1: Volatilidade do dia a dia
# Coeficiente de Variacao = std(fat diario) / mean(fat diario). Adimensional, robusto a porte.
volatilidade = df_b3.groupby('loja')['faturamento'].agg(
    media_fat_diaria='mean',
    std_fat_diaria='std',
).reset_index()
volatilidade['cv_volatilidade_diaria'] = np.where(
    volatilidade['media_fat_diaria'] > 0,
    volatilidade['std_fat_diaria'] / volatilidade['media_fat_diaria'], np.nan
)

# Medida 2: Sazonalidade mensal
# Passo 1: para cada loja, media de faturamento por (ano, mes)
df_b3['ano_mes'] = df_b3['data'].dt.to_period('M').astype(str)
mensais = df_b3.groupby(['loja', 'ano_mes'])['faturamento'].mean().reset_index(name='media_mensal')
# Passo 2: para cada loja, calcular CV das medias mensais
sazonal = mensais.groupby('loja')['media_mensal'].agg(
    media_geral_meses='mean',
    std_entre_meses='std',
    n_meses='count',
).reset_index()
sazonal['cv_sazonalidade_mensal'] = np.where(
    sazonal['media_geral_meses'] > 0,
    sazonal['std_entre_meses'] / sazonal['media_geral_meses'], np.nan
)
print(f'Min de meses observados por loja (apos mitigacao): {sazonal["n_meses"].min()}')
print(f'Max de meses observados por loja (apos mitigacao): {sazonal["n_meses"].max()}')

bloco3_saida = volatilidade[['loja','cv_volatilidade_diaria']].merge(
    sazonal[['loja','cv_sazonalidade_mensal']], on='loja', how='outer'
)
print()
print('Describe do BLOCO 3:')
print(bloco3_saida.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(4).to_string())

# ============================================================
# BLOCO 4: Formato, concorrencia e cadastro
# ============================================================
print()
print('=' * 70)
print('BLOCO 4: FORMATO, CONCORRENCIA E CADASTRO')
print('=' * 70)

# --- Parte A: taxa de abertura de domingo (ja usando dias operando, fechado = nao operante)
dias_domingo = df_treino[df_treino['dia_semana'] == CODIGO_DOMINGO].copy()
dias_domingo['operou'] = (dias_domingo['aberta'] == 1) & (dias_domingo['faturamento'] > 0)
domingo_por_loja = dias_domingo.groupby('loja').agg(
    domingos_tot=('data','count'),
    domingos_operou=('operou','sum'),
).reset_index()
domingo_por_loja['tx_domingos_abertos'] = np.where(
    domingo_por_loja['domingos_tot'] > 0,
    domingo_por_loja['domingos_operou'] / domingo_por_loja['domingos_tot'], np.nan
)

# --- Parte B: pico de sabado vs media de segunda a sexta
# Usa dias operando (ja filtrado)
mask_seg_sex = df_op['dia_semana'].between(1, 5)
mask_sab = df_op['dia_semana'] == CODIGO_SABADO
seg_sex = df_op[mask_seg_sex].groupby('loja')['faturamento'].mean().rename('vm_seg_sex').reset_index()
sab = df_op[mask_sab].groupby('loja')['faturamento'].mean().rename('vm_sabado').reset_index()
pico = seg_sex.merge(sab, on='loja', how='outer')
pico['razao_pico_sabado'] = np.where(pico['vm_seg_sex'] > 0, pico['vm_sabado'] / pico['vm_seg_sex'], np.nan)

# --- Parte C: Colunas do cadastro
# Tratar distancia_concorrente: imputar 3 nulos com a mediana GLOBAL da base
mediana_dist = df_loja['distancia_concorrente'].median()
n_antes_nulos = df_loja['distancia_concorrente'].isnull().sum()
cadastro = df_loja.copy()
cadastro['distancia_concorrente'] = cadastro['distancia_concorrente'].fillna(mediana_dist)
cadastro['distancia_foi_imputada'] = (df_loja['distancia_concorrente'].isnull()).astype(int)
n_depois_nulos = cadastro['distancia_concorrente'].isnull().sum()
print(f'Mediana global de distancia_concorrente usada na imputacao: {mediana_dist:,.0f}')
print(f'Nulos antes: {n_antes_nulos} | Nulos apos: {n_depois_nulos}')
print(f'Flag distancia_foi_imputada: {(cadastro["distancia_foi_imputada"]==1).sum()} marcadas')

# Log1p em distancia_concorrente (tratar assimetria)
cadastro['distancia_concorrente_log1p'] = np.log1p(cadastro['distancia_concorrente'])
# (obs: as outras colunas de outros blocos passarao por log1p em etapa de feature engineering separada,
#  aqui apenas guardamos a versao tratada do cadastro conforme decisao da Fase 2+3)

cols_cadastro_manter = [
    'loja','distancia_concorrente','distancia_concorrente_log1p','distancia_foi_imputada',
    'promo_continua','tipo_loja','sortimento'
]
cadastro_saida = cadastro[cols_cadastro_manter].copy()

# Montar bloco 4 completo
bloco4_saida = domingo_por_loja[['loja','tx_domingos_abertos']].merge(
    pico[['loja','razao_pico_sabado']], on='loja', how='outer'
).merge(cadastro_saida, on='loja', how='outer')

print()
print('Describe do BLOCO 4 (numericas apenas):')
cols_num_b4 = ['tx_domingos_abertos','razao_pico_sabado','distancia_concorrente',
               'distancia_concorrente_log1p','promo_continua','distancia_foi_imputada']
print(bloco4_saida[cols_num_b4].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).round(4).to_string())
print()
print('Contagens das categoricas baseline no BLOCO 4:')
for c in ['tipo_loja','sortimento']:
    print(f'--- {c} ---')
    print(bloco4_saida[c].value_counts(dropna=False).sort_index().to_string())
    print()

# ============================================================
# BLOCO FINAL: Consolidar todas as tabelas em um dataframe por loja final
# ============================================================
print('=' * 70)
print('TABELA FINAL CONSOLIDADA (1 linha por LOJA)')
print('=' * 70)

tabela_final = bloco1_saida.merge(bloco2_saida, on='loja', how='outer') \
    .merge(bloco3_saida, on='loja', how='outer') \
    .merge(bloco4_saida, on='loja', how='outer')

# Ordem de colunas (legibilidade)
ordem = [
    'loja',
    # Bloco 1
    'venda_media_dia','clientes_medio_dia','ticket_medio',
    # Bloco 2
    'razao_promo','uplift_promo_pct',
    # Bloco 3
    'cv_volatilidade_diaria','cv_sazonalidade_mensal',
    # Bloco 4
    'tx_domingos_abertos','razao_pico_sabado',
    'distancia_concorrente','distancia_concorrente_log1p','distancia_foi_imputada','promo_continua',
    # Baselines
    'tipo_loja','sortimento',
]
tabela_final = tabela_final[ordem]

print()
print(f'Shape final: {tabela_final.shape[0]} linhas x {tabela_final.shape[1]} colunas')
print()
print('Colunas, tipos e nulos restantes:')
tipos = tabela_final.dtypes.rename('tipo').to_frame()
tipos['nulos_restantes'] = tabela_final.isnull().sum()
tipos['%_nulos'] = (tipos['nulos_restantes'] / len(tabela_final) * 100).round(4).astype(str) + ' %'
print(tipos.to_string())

# ============================================================
# Salvar tabela final em data/03_processed
# ============================================================
tabela_final.to_csv('data/03_processed/tabela_por_loja_fase3.csv', index=False)
try:
    tabela_final.to_parquet('data/03_processed/tabela_por_loja_fase3.parquet', index=False)
    print()
    print(f'Arquivos salvos em data/03_processed/: tabela_por_loja_fase3.csv e .parquet')
except ImportError as e:
    print()
    print(f'AVISO: Nao foi possivel salvar parquet (motor ausente: {e}). CSV salvo com sucesso.')
