import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ============================================================
# Carregar base
# ============================================================
df_treino = pd.read_csv('data/01_raw/treino.csv', parse_dates=['data'])

print('=' * 70)
print('PRIMEIRO: descobrindo qual dia_semana = domingo (via calendario)')
print('=' * 70)
# Data conhecida: 01/01/2013 foi uma TERCA-FEIRA
amostra = df_treino[['data', 'dia_semana']].drop_duplicates().sort_values('data')
amostra['nome_dia_correto'] = amostra['data'].dt.day_name(locale='pt_BR.UTF-8')
amostra['dow_pandas'] = amostra['data'].dt.dayofweek  # seg=0, dom=6
tabela_mapeamento = amostra.groupby(['dia_semana', 'dow_pandas', 'nome_dia_correto']).size().reset_index(name='n')
print(tabela_mapeamento.to_string(index=False))
print()

mapa_dia = {row.dia_semana: row.nome_dia_correto
            for row in tabela_mapeamento.itertuples()}
mapa_dow = {row.dia_semana: row.dow_pandas
            for row in tabela_mapeamento.itertuples()}
print(f'Mapeamento dia_semana -> nome: {mapa_dia}')
print(f'Mapeamento dia_semana -> dow_pandas (seg=0..dom=6): {mapa_dow}')

# Determinar qual codificacao de dia_semana = domingo
codigo_domingo = tabela_mapeamento.loc[tabela_mapeamento['dow_pandas'] == 6, 'dia_semana'].iloc[0]
print(f'Código de domingo na coluna dia_semana: {codigo_domingo} ({mapa_dia[codigo_domingo]})')

# ============================================================
# ITEM 1: faturamento quebrado por aberta
# ============================================================
print()
print('=' * 70)
print('ITEM 1: FATURAMENTO QUEBRADO PELA COLUNA aberta')
print('=' * 70)

grupo = df_treino.groupby('aberta')['faturamento'].agg(
    contagem='count',
    media='mean',
    mediana='median',
    soma='sum'
).reset_index()
grupo['%_base'] = grupo['contagem'] / len(df_treino) * 100
print(grupo.to_string(index=False))
print()

# ============================================================
# ITEM 2: incoerencia aberta x faturamento (dois sentidos)
# ============================================================
print()
print('=' * 70)
print('ITEM 2: LINHAS INCOERENTES ENTRE aberta E faturamento')
print('=' * 70)

# Sentido 1: aberta=0 mas faturamento > 0 (fechada e vendeu)
inc1 = df_treino[(df_treino['aberta'] == 0) & (df_treino['faturamento'] > 0)]
print(f'Sentido 1: aberta=0 (fechada) e faturamento > 0 .................: {len(inc1)} linhas')
if len(inc1) > 0:
    print('Amostra (max 10 linhas):')
    print(inc1.head(10)[['loja','data','aberta','faturamento','clientes','promo']].to_string(index=False))

# Sentido 2: aberta=1 mas faturamento = 0 (abriu e nao vendeu nada)
inc2 = df_treino[(df_treino['aberta'] == 1) & (df_treino['faturamento'] == 0)]
print(f'Sentido 2: aberta=1 (aberta) e faturamento = 0 .................: {len(inc2)} linhas')
if len(inc2) > 0:
    print(f'  Lojas distintas envolvidas: {inc2["loja"].nunique()}')
    print('  Dias da semana (dia_semana -> nome):')
    print(inc2['dia_semana'].map(mapa_dia).value_counts().to_string())
    print()
    print('  Tem feriado_estadual != 0? Contagem:')
    print(inc2['feriado_estadual'].astype(str).value_counts().to_string())
    print()
    print('  Tem feriado_escolar = 1? Contagem:')
    print(inc2['feriado_escolar'].value_counts().to_string())
    print()
    print('  Distribuição de clientes nessas linhas:')
    print(inc2['clientes'].value_counts().sort_index().to_string())
    print()
    print('  Amostra (max 10 linhas):')
    print(inc2.head(10)[['loja','data','dia_semana','aberta','faturamento','clientes','promo','feriado_estadual','feriado_escolar']].to_string(index=False))

# Caso 2b: aberta=1 com clientes = 0 (mesma logica, mas olhando a coluna clientes)
inc2b = df_treino[(df_treino['aberta'] == 1) & (df_treino['clientes'] == 0)]
print(f'Sentido 2b: aberta=1 (aberta) e clientes = 0 ..................: {len(inc2b)} linhas')
if len(inc2b) > 0:
    interseccao = len(set(inc2.index) & set(inc2b.index))
    print(f'  Dessas, quantas tambem tem faturamento=0? {interseccao} (sobreposicao)')

# Caso 3: faturamento > 0 porem clientes = 0 (vendeu sem cliente? impossivel conceitualmente)
inc3 = df_treino[(df_treino['faturamento'] > 0) & (df_treino['clientes'] == 0)]
print(f'Outro: faturamento > 0 e clientes = 0 .........................: {len(inc3)} linhas')
# Caso 4: clientes > 0 e faturamento = 0 (cliente entrou e nao gastou nada?)
inc4 = df_treino[(df_treino['clientes'] > 0) & (df_treino['faturamento'] == 0)]
print(f'Outro: clientes > 0 e faturamento = 0 .........................: {len(inc4)} linhas')

# ============================================================
# ITEM 3: Proporcao de domingos em que a loja esteve aberta
# ============================================================
print()
print('=' * 70)
print('ITEM 3: PROPORCAO DE DOMINGOS EM QUE CADA LOJA ESTEVE ABERTA')
print('=' * 70)

domingos = df_treino[df_treino['dia_semana'] == codigo_domingo].copy()
domingos_por_loja = domingos.groupby('loja')['aberta'].agg(
    domingos_tot='count',
    domingos_abertas='sum',
).reset_index()
domingos_por_loja['tx_domingos_abertos'] = (
    domingos_por_loja['domingos_abertas'] / domingos_por_loja['domingos_tot']
)

print(f'Número médio de domingos observados por loja: {domingos_por_loja["domingos_tot"].mean():.1f}')
print()

# Classificar lojas: abre domingo (>= 50% dos domingos abertos) vs nao abre (< 50%)
domingos_por_loja['abre_domingo'] = np.where(domingos_por_loja['tx_domingos_abertos'] >= 0.5, 1, 0)
print('Classificacao por abertura de domingo (corte: tx >= 50%):')
print(domingos_por_loja['abre_domingo'].value_counts().sort_index().to_string())
print()

# Distribuicao da proporcao em faixas
faixas = pd.cut(
    domingos_por_loja['tx_domingos_abertos'],
    bins=[-0.001, 0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.001],
    labels=['0% (nunca)', '(0%,1%]', '(1%,25%]', '(25%,50%]', '(50%,75%]', '(75%,99%]', '100% (sempre)']
)
print('Distribuicao da rede por faixa de % domingos abertos:')
print(faixas.value_counts().sort_index().to_string())
print()
print('Estatisticas da taxa de domingos abertos por loja:')
print(f'  min     = {domingos_por_loja["tx_domingos_abertos"].min():.4f}')
print(f'  P25     = {domingos_por_loja["tx_domingos_abertos"].quantile(0.25):.4f}')
print(f'  mediana = {domingos_por_loja["tx_domingos_abertos"].median():.4f}')
print(f'  P75     = {domingos_por_loja["tx_domingos_abertos"].quantile(0.75):.4f}')
print(f'  max     = {domingos_por_loja["tx_domingos_abertos"].max():.4f}')
print(f'  media   = {domingos_por_loja["tx_domingos_abertos"].mean():.4f}')

# ============================================================
# ITEM 4: Venda media todos os dias vs so dias abertos
# ============================================================
print()
print('=' * 70)
print('ITEM 4: VENDA MEDIA POR LOJA (TODOS OS DIAS vs SO DIAS ABERTOS)')
print('=' * 70)

por_loja = df_treino.groupby('loja').agg(
    dias_tot=('data', 'count'),
    dias_abertos=('aberta', 'sum'),
    fat_total=('faturamento', 'sum'),
).reset_index()
por_loja['tx_aberto'] = por_loja['dias_abertos'] / por_loja['dias_tot']
por_loja['vm_todos_dias'] = por_loja['fat_total'] / por_loja['dias_tot']
por_loja['vm_somente_abertos'] = por_loja['fat_total'] / por_loja['dias_abertos']

# Integrar com a classificacao de abre domingo
por_loja = por_loja.merge(
    domingos_por_loja[['loja','tx_domingos_abertos','abre_domingo']],
    on='loja', how='left'
)

# Resumo comparativo
resumo = por_loja.groupby('abre_domingo').agg(
    n_lojas=('loja','count'),
    tx_aberto_medio=('tx_aberto','mean'),
    vm_todos_dias_media=('vm_todos_dias','mean'),
    vm_todos_dias_mediana=('vm_todos_dias','median'),
    vm_abertos_media=('vm_somente_abertos','mean'),
    vm_abertos_mediana=('vm_somente_abertos','median'),
).reset_index()
resumo['razao_media_abertos/todos'] = resumo['vm_abertos_media'] / resumo['vm_todos_dias_media']
resumo['grupo_nome'] = resumo['abre_domingo'].map({0:'NAO abre domingo (tx<50%)', 1:'ABRE domingo (tx>=50%)'})
cols = ['grupo_nome','n_lojas','tx_aberto_medio','vm_todos_dias_media','vm_todos_dias_mediana',
        'vm_abertos_media','vm_abertos_mediana','razao_media_abertos/todos']
print(resumo[cols].to_string(index=False))
print()

# Tambem a nivel de loja individual: diferencas
por_loja['razao'] = por_loja['vm_somente_abertos'] / por_loja['vm_todos_dias']
print('Razao (vm_abertos / vm_todos_dias) POR LOJA - distribuicao:')
print(f'  min     = {por_loja["razao"].min():.3f}')
print(f'  P25     = {por_loja["razao"].quantile(0.25):.3f}')
print(f'  mediana = {por_loja["razao"].median():.3f}')
print(f'  P75     = {por_loja["razao"].quantile(0.75):.3f}')
print(f'  max     = {por_loja["razao"].max():.3f}')

# Salvar por_loja para uso posterior
por_loja.to_csv('data/02_interim/resumo_por_loja_fase2_item4.csv', index=False)
print()
print('(Arquivo auxiliar salvo: data/02_interim/resumo_por_loja_fase2_item4.csv)')