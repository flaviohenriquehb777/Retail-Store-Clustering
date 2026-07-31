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

# Maquinao de domingo (ja descoberto na Fase 2)
CODIGO_DOMINGO = 7  # dia_semana -> domingo

# ============================================================
# ITEM 1: tabela de contingencia tipo_loja x sortimento
# ============================================================
print('=' * 70)
print('ITEM 1: TABELA DE CONTINGENCIA tipo_loja x sortimento')
print('=' * 70)

tab_contingencia = pd.crosstab(df_loja['tipo_loja'], df_loja['sortimento'], margins=True, margins_name='TOTAL')
print('Contagens absolutas:')
print(tab_contingencia.to_string())
print()

tab_pct_linha = pd.crosstab(df_loja['tipo_loja'], df_loja['sortimento'], normalize='index').round(4) * 100
tab_pct_linha['TOTAL'] = tab_pct_linha.sum(axis=1)
print('Percentual por LINHA (distribuicao do sortimento DENTRO de cada tipo_loja):')
print(tab_pct_linha.round(2).to_string())
print()

tab_pct_coluna = pd.crosstab(df_loja['tipo_loja'], df_loja['sortimento'], normalize='columns').round(4) * 100
print('Percentual por COLUNA (distribuicao do tipo_loja DENTRO de cada sortimento):')
print(tab_pct_coluna.round(2).to_string())

# ============================================================
# ITEM 2: metricas por tipo_loja
# ============================================================
print()
print('=' * 70)
print('ITEM 2: METRICAS DE NEGOCIO POR tipo_loja')
print('=' * 70)

# Primeiro, metricas POR LOJA (agregado), depois agregar por tipo_loja
metricas_loja = df_treino.groupby('loja').agg(
    dias_tot_reg=('data','count'),
    dias_abertos=('aberta','sum'),
    fat_total=('faturamento','sum'),
    cli_total=('clientes','sum'),
    domingos_tot=('dia_semana', lambda s: (s == CODIGO_DOMINGO).sum()),
    domingos_abertos=('dia_semana', lambda s: ((s == CODIGO_DOMINGO) & (df_treino.loc[s.index, 'aberta'] == 1)).sum()),
).reset_index()

metricas_loja['vm_abertos'] = np.where(
    metricas_loja['dias_abertos'] > 0,
    metricas_loja['fat_total'] / metricas_loja['dias_abertos'], np.nan
)
metricas_loja['clientes_med_por_dia'] = np.where(
    metricas_loja['dias_abertos'] > 0,
    metricas_loja['cli_total'] / metricas_loja['dias_abertos'], np.nan
)
metricas_loja['ticket_medio'] = np.where(
    metricas_loja['cli_total'] > 0,
    metricas_loja['fat_total'] / metricas_loja['cli_total'], np.nan
)
metricas_loja['tx_domingos_abertos'] = np.where(
    metricas_loja['domingos_tot'] > 0,
    metricas_loja['domingos_abertos'] / metricas_loja['domingos_tot'], np.nan
)

# Juntar cadastro
completo = metricas_loja.merge(df_loja, on='loja', how='left')

# Agregar por tipo_loja: mediana (robusta a outliers) E media, para cada metrica.
# Escolha: mediana (p), pois e a menos sensivel ao skew e as 180 lojas incompletas.
# Mas mostraremos media tb para comparar.
cols_metrica = ['vm_abertos', 'clientes_med_por_dia', 'ticket_medio', 'tx_domingos_abertos']

agreg_mediana = completo.groupby('tipo_loja')[cols_metrica].median().round(2)
agreg_media = completo.groupby('tipo_loja')[cols_metrica].mean().round(2)
n_lojas = completo.groupby('tipo_loja').size().rename('n_lojas')

print('Usamos MEDIANA para sumarizar por tipo_loja (mais robusta).')
print()
tabela = pd.concat([n_lojas, agreg_mediana], axis=1).reset_index()
tabela['%_lojas_rede'] = (tabela['n_lojas'] / tabela['n_lojas'].sum() * 100).round(2)
tabela['tx_domingos_abertos'] = (tabela['tx_domingos_abertos'] * 100).round(2).astype(str) + ' %'
mostrar_cols = ['tipo_loja','n_lojas','%_lojas_rede','vm_abertos','clientes_med_por_dia','ticket_medio','tx_domingos_abertos']
print(tabela[mostrar_cols].to_string(index=False))
print()
print('Para comparacao, a MEDIA por tipo_loja (sensivel a outliers / lojas incompletas):')
tabela_media = pd.concat([n_lojas, agreg_media], axis=1).reset_index()
tabela_media['tx_domingos_abertos'] = (tabela_media['tx_domingos_abertos'] * 100).round(2).astype(str) + ' %'
print(tabela_media.drop(columns=['n_lojas']).to_string(index=False))

# ============================================================
# ITEM 3: interpretacao do modelo de negocio de cada tipo
# ============================================================
print()
print('=' * 70)
print('ITEM 3: INTERPRETACAO DO MODELO DE NEGOCIO DE CADA tipo_loja')
print('=' * 70)

# Mediana por tipo_loja ja calculada. Vamos rankear cada metrica dentro de cada tipo.
rank_med = agreg_mediana.rank(ascending=False, method='dense').astype(int)
rank_med.columns = ['rank_' + c for c in rank_med.columns]
analise = pd.concat([agreg_mediana, rank_med], axis=1)
print('Ranking de cada tipo_loja em cada metrica (1 = maior/melhor da rede):')
print(analise.round(2).to_string())
print()

# Contar quais tipos abrem domingo "sempre" (tx >= 90%) e quais "quase nunca" (<5%)
tipos_list = sorted(completo['tipo_loja'].unique())
print('Abertura de domingo por tipo_loja (distribuicao % de lojas por faixa):')
completo['faixa_dom'] = pd.cut(
    completo['tx_domingos_abertos'],
    bins=[-0.001, 0.01, 0.10, 0.50, 0.90, 1.001],
    labels=['Nunca (<=1%)','Raro (<=10%)','Ate 50%','Frequent. (50-90%)','Sempre (>=90%)']
)
tab_faixa_dom = pd.crosstab(completo['tipo_loja'], completo['faixa_dom'], normalize='index').round(4) * 100
print(tab_faixa_dom.round(1).to_string())

# ============================================================
# ITEM 4: grupo raro
# ============================================================
print()
print('=' * 70)
print('ITEM 4: GRUPO RARO (tipo_loja b) - 17 lojas (1,5%)')
print('=' * 70)

grupo_raro = completo[completo['tipo_loja'] == 'b'].copy()
print(f'Tamanho do grupo: {len(grupo_raro)} lojas ({len(grupo_raro)/len(completo)*100:.2f}% da rede)')
print()
print('Estatisticas DESCRITIVAS completas das metricas DENTRO do grupo b:')
for c in cols_metrica + ['dias_abertos','fat_total']:
    desc = grupo_raro[c].describe(percentiles=[0.10, 0.25, 0.5, 0.75, 0.9])
    print(f'  {c}:')
    print(desc.to_string())
    print()

# Mostrar sortimento dentro do grupo b
print('Distribuicao de sortimento DENTRO do tipo_loja b:')
print(grupo_raro['sortimento'].value_counts(dropna=False).sort_index().to_string())
print()

# Comparar a distribuicao do grupo b contra a rede em VM_abertos (sobreposicao de quartis)
print('Comparar posicao das lojas do tipo b no ranking global de VM_abertos:')
completo['rank_vm_global'] = completo['vm_abertos'].rank(ascending=False, method='first').astype(int)
perc_b = (completo['rank_vm_global'] <= len(completo)*0.50).sum()
lojas_b_top = (completo['tipo_loja'] == 'b') & (completo['rank_vm_global'] <= len(completo)*0.50)
lojas_b_mid = (completo['tipo_loja'] == 'b') & (completo['rank_vm_global'] > len(completo)*0.50) & (completo['rank_vm_global'] <= len(completo)*0.80)
lojas_b_bot = (completo['tipo_loja'] == 'b') & (completo['rank_vm_global'] > len(completo)*0.80)
print(f'  Tipo b no top 50% do ranking global: {lojas_b_top.sum()} de 17 ({lojas_b_top.sum()/17*100:.1f}%)')
print(f'  Tipo b no meio 30% (50-80% rank)    : {lojas_b_mid.sum()} de 17 ({lojas_b_mid.sum()/17*100:.1f}%)')
print(f'  Tipo b no ultimo 20% (piores)       : {lojas_b_bot.sum()} de 17 ({lojas_b_bot.sum()/17*100:.1f}%)')
