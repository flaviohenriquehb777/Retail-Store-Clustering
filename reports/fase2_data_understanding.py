import pandas as pd
import warnings
warnings.filterwarnings('ignore')

SEED = 42

# ============================================================
# 1. Carregar bases com data como datetime; dims e período
# ============================================================
print('=' * 70)
print('ITEM 1: DIMENSÕES E PERÍODO COBERTO')
print('=' * 70)

df_loja = pd.read_csv('data/01_raw/loja.csv')
df_treino = pd.read_csv('data/01_raw/treino.csv', parse_dates=['data'])

print(f'loja.csv   : {df_loja.shape[0]:>8} linhas x {df_loja.shape[1]} colunas')
print(f'treino.csv : {df_treino.shape[0]:>8} linhas x {df_treino.shape[1]} colunas')
print()

data_min = df_treino['data'].min().strftime('%d/%m/%Y')
data_max = df_treino['data'].max().strftime('%d/%m/%Y')
dias_total = (df_treino['data'].max() - df_treino['data'].min()).days + 1
print('Periodo coberto por treino.csv:')
print(f'  1o dia     : {data_min}')
print(f'  Ultimo dia : {data_max}')
print(f'  Dias totais: {dias_total} dias corridos')
print()
print(f'Lojas unicas em treino.csv: {df_treino["loja"].nunique()}')
print(f'Lojas unicas em loja.csv  : {df_loja["loja"].nunique()}')
print(f'Lojas em ambos (intersecao): {len(set(df_treino["loja"]) & set(df_loja["loja"]))}')

# ============================================================
# 2. Primeiras linhas das duas bases
# ============================================================
print()
print('=' * 70)
print('ITEM 2: PRIMEIRAS LINHAS - loja.csv')
print('=' * 70)
print(df_loja.head(10).to_string(index=False))

print()
print('=' * 70)
print('ITEM 2: PRIMEIRAS LINHAS - treino.csv')
print('=' * 70)
print(df_treino.head(10).to_string(index=False))

# ============================================================
# 3. Dicionario de dados (tipo, significado, unidade)
# ============================================================
print()
print('=' * 70)
print('ITEM 3: DICIONARIO DE DADOS - loja.csv')
print('=' * 70)
print(f'{"Coluna":<25} {"Tipo pandas":<18} {"Significado":<60} {"Unidade"}')
print('-' * 130)

info_loja = [
    ("loja",
     str(df_loja["loja"].dtype),
     "Identificador unico da loja (chave primaria / link com treino.csv)",
     "Inteiro (id), sem unidade"),
    ("tipo_loja",
     str(df_loja["tipo_loja"].dtype),
     "Categoria cadastral da loja (baseline, NAO entra no modelo). Valores: "
     + str(sorted(df_loja["tipo_loja"].unique().tolist())),
     "Categoria nominal (letra)"),
    ("sortimento",
     str(df_loja["sortimento"].dtype),
     "Categoria cadastral de sortimento (baseline, NAO entra no modelo). Valores: "
     + str(sorted(df_loja["sortimento"].unique().tolist())),
     "Categoria nominal (letra)"),
    ("distancia_concorrente",
     str(df_loja["distancia_concorrente"].dtype),
     "Distancia ate o concorrente mais proximo da loja",
     "Metros (presumido - nao documentado explicitamente no caso)"),
    ("promo_continua",
     str(df_loja["promo_continua"].dtype),
     "Indica se a loja participa do programa de promocao continua. Valores: 0 = nao, 1 = sim",
     "Binaria 0/1 (flag)"),
]
for c, t, s, u in info_loja:
    print(f'{c:<25} {t:<18} {s[:58]:<60} {u}')

print()
print('=' * 70)
print('ITEM 3: DICIONARIO DE DADOS - treino.csv')
print('=' * 70)
print(f'{"Coluna":<22} {"Tipo pandas":<18} {"Significado":<62} {"Unidade"}')
print('-' * 130)

info_treino = [
    ("loja",
     str(df_treino["loja"].dtype),
     "Identificador da loja (chave estrangeira para loja.csv)",
     "Inteiro (id), sem unidade"),
    ("dia_semana",
     str(df_treino["dia_semana"].dtype),
     "Dia da semana codificado de 1 a 7. Nao ha documentacao sobre qual dia = 1 (nao obvio)",
     "Inteiro ordinal de 1 a 7"),
    ("data",
     str(df_treino["data"].dtype),
     "Data calendario do registro diario",
     "Data (YYYY-MM-DD)"),
    ("faturamento",
     str(df_treino["faturamento"].dtype),
     "Valor total de vendas registrado na loja naquele dia",
     "Reais (R$), presumido valor bruto"),
    ("clientes",
     str(df_treino["clientes"].dtype),
     "Quantidade de clientes atendidos na loja naquele dia",
     "Contagem (pessoas)"),
    ("aberta",
     str(df_treino["aberta"].dtype),
     "Indica se a loja estava aberta no dia. 0 = fechada, 1 = aberta",
     "Binaria 0/1 (flag)"),
    ("promo",
     str(df_treino["promo"].dtype),
     "Indica se havia promocao ativa na loja naquele dia. 0 = sem promo, 1 = com promo",
     "Binaria 0/1 (flag)"),
    ("feriado_estadual",
     str(df_treino["feriado_estadual"].dtype),
     "Ausencia ou tipo de feriado estadual. Valores unicos: "
     + str([str(x) for x in df_treino["feriado_estadual"].unique()])
     + ". Significado de a/b/c nao esta documentado (nao obvio)",
     "Categoria nominal: 0 (sem) / a / b / c"),
    ("feriado_escolar",
     str(df_treino["feriado_escolar"].dtype),
     "Indica ocorrencia de feriado escolar no dia. 0 = nao, 1 = sim",
     "Binaria 0/1 (flag)"),
]
for c, t, s, u in info_treino:
    print(f'{c:<22} {t:<18} {s[:60]:<62} {u}')

# ============================================================
# 4. Diferenca entre colunas de nomes parecidos
# ============================================================
print()
print('=' * 70)
print('ITEM 4: DIFERENCA ENTRE COLUNAS DE NOMES PARECIDOS')
print('=' * 70)

print()
print('Par 1: promo_continua (loja.csv) vs promo (treino.csv)')
print('-' * 70)
print(f'  promo_continua (loja.csv): flag POR LOJA, cadastral, estavel no tempo.')
print(f'    - Niveis: {sorted(df_loja["promo_continua"].unique())}')
print(f'    - Lojas em programa continuo: {df_loja["promo_continua"].sum()} de {len(df_loja)}')
print(f'  promo (treino.csv): flag POR DIA POR LOJA, operacional, varia no tempo.')
print(f'    - Niveis: {sorted(df_treino["promo"].unique())}')
print(f'    - Dias com promo: {df_treino["promo"].sum()} de {len(df_treino)} linhas')
print()
print('  Diferenca: Uma loja no programa continuo pode ter ou nao promocao ativa em um dia')
print('  especifico. Ambas as colunas existem e serao exploradas na modelagem.')

print()
print('Par 2: feriado_estadual vs feriado_escolar (ambos em treino.csv)')
print('-' * 70)
print(f'  feriado_estadual: categoria 0/a/b/c - feriado de calendario civil estadual')
vc_est = df_treino['feriado_estadual'].astype(str).value_counts().to_dict()
print(f'    - Distribuicao: {vc_est}')
print(f'  feriado_escolar: binaria 0/1 - recesso escolar (ex: ferias, semana de provas)')
print(f'    - Distribuicao: {df_treino["feriado_escolar"].value_counts().to_dict()}')
print()
print('  Diferenca: Feriado estadual fecha comercio por lei estadual; feriado escolar')
print('  altera o fluxo de pessoas mas nao necessariamente fecha lojas.')

# ============================================================
# 5. Nulos e duplicatas
# ============================================================
print()
print('=' * 70)
print('ITEM 5: VALORES NULOS E DUPLICATAS')
print('=' * 70)

print()
print('--- loja.csv ---')
nulos_loja = df_loja.isnull().sum()
print('Valores nulos por coluna:')
for col in df_loja.columns:
    pct = nulos_loja[col] / len(df_loja) * 100
    print(f'  {col:<25}: {nulos_loja[col]:>5} nulos  ({pct:>6.2f}%)')

dup_loja_chave = df_loja.duplicated(subset=['loja']).sum()
dup_loja_linha = df_loja.duplicated().sum()
print(f'Duplicatas por chave (loja)           : {dup_loja_chave}')
print(f'Duplicatas de linha inteira (todas as cols): {dup_loja_linha}')

print()
print('--- treino.csv ---')
nulos_treino = df_treino.isnull().sum()
print('Valores nulos por coluna:')
for col in df_treino.columns:
    pct = nulos_treino[col] / len(df_treino) * 100
    print(f'  {col:<22}: {nulos_treino[col]:>7} nulos  ({pct:>6.2f}%)')

dup_treino_chave = df_treino.duplicated(subset=['loja', 'data']).sum()
dup_treino_linha = df_treino.duplicated().sum()
print(f'Duplicatas por chave composta (loja, data)    : {dup_treino_chave}')
print(f'Duplicatas de linha inteira (todas as cols)   : {dup_treino_linha}')

# ============================================================
# EXTRA: dados importantes na amostra
# ============================================================
print()
print('=' * 70)
print('OBSERVACOES RELEVANTES NOS DADOS BRUTOS (APENAS DESCOBERTA)')
print('=' * 70)

print()
print(f'Faturamento em treino.csv: min R$ {df_treino["faturamento"].min():>10,} | '
      f'mediana R$ {df_treino["faturamento"].median():>10,} | '
      f'max R$ {df_treino["faturamento"].max():>10,}')
print(f'Clientes em treino.csv   : min {df_treino["clientes"].min():>7,} | '
      f'mediana {df_treino["clientes"].median():>7,} | '
      f'max {df_treino["clientes"].max():>7,}')

dias_fechados = (df_treino['aberta'] == 0).sum()
fat_zero = (df_treino['faturamento'] == 0).sum()
cli_zero = (df_treino['clientes'] == 0).sum()
print(f'Dias com aberta=0          : {dias_fechados:>8}')
print(f'Dias com faturamento = 0   : {fat_zero:>8}')
print(f'Dias com clientes = 0      : {cli_zero:>8}')

dif_fat_aberta = set(df_treino.loc[df_treino['aberta'] == 0, 'loja'].unique()) - \
                 set(df_treino.loc[df_treino['faturamento'] == 0, 'loja'].unique())
dif_aberta_fat = set(df_treino.loc[df_treino['faturamento'] == 0, 'loja'].unique()) - \
                 set(df_treino.loc[df_treino['aberta'] == 0, 'loja'].unique())
print(f'Dias fechados (aberta=0) com faturamento > 0: {df_treino[(df_treino.aberta == 0) & (df_treino.faturamento > 0)].shape[0]}')
print(f'Dias abertos (aberta=1) com faturamento = 0: {df_treino[(df_treino.aberta == 1) & (df_treino.faturamento == 0)].shape[0]}')
print(f'Dias abertos (aberta=1) com clientes = 0   : {df_treino[(df_treino.aberta == 1) & (df_treino.clientes == 0)].shape[0]}')

print()
print('Dias em que feriado_estadual != 0 e loja estava aberta=1:')
mask_feriado = df_treino['feriado_estadual'].astype(str).str.strip() != '0'
print(f'  Total linhas com feriado estadual: {mask_feriado.sum()}')
print(f'  Destas, aberta=1: {df_treino.loc[mask_feriado, "aberta"].sum()}')
print(f'  Destas, aberta=0: {(mask_feriado.sum() - df_treino.loc[mask_feriado, "aberta"].sum())}')

print()
print('Tipos cadastrais (baseline) em loja.csv:')
print(df_loja['tipo_loja'].value_counts().sort_index().to_string())
print()
print('Sortimentos (baseline) em loja.csv:')
print(df_loja['sortimento'].value_counts().sort_index().to_string())