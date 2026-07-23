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

# Determinar periodo completo coberto na base
DATA_MIN = df_treino['data'].min()
DATA_MAX = df_treino['data'].max()
CALENDARIO_COMPLETO = pd.date_range(start=DATA_MIN, end=DATA_MAX, freq='D')
TOTAL_DIAS_ESPERADO = len(CALENDARIO_COMPLETO)
print(f'Periodo completo: {DATA_MIN.strftime("%d/%m/%Y")} a {DATA_MAX.strftime("%d/%m/%Y")}  ({TOTAL_DIAS_ESPERADO} dias corridos)')

# ============================================================
# ITEM 1: Contagem de dias de registro por loja
# ============================================================
print()
print('=' * 70)
print('ITEM 1: DISTRIBUICAO DA CONTAGEM DE DIAS POR LOJA')
print('=' * 70)

contagem = df_treino.groupby('loja')['data'].count().reset_index(name='dias_registrados')
contagem['dias_faltantes'] = TOTAL_DIAS_ESPERADO - contagem['dias_registrados']
contagem['completa'] = contagem['dias_faltantes'] == 0

print(f'Total de lojas: {len(contagem)}')
print(f'Lojas com serie COMPLETA ({TOTAL_DIAS_ESPERADO} dias): {contagem["completa"].sum()}')
print(f'Lojas com serie INCOMPLETA (faltam dias): {(~contagem["completa"]).sum()}')
print()

print('Distribuicao da contagem de dias registrados:')
print(contagem['dias_registrados'].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_string())
print()

print('Contagem de lojas por faixa de dias faltantes:')
faixas_falt = pd.cut(
    contagem['dias_faltantes'],
    bins=[-1, 0, 1, 5, 10, 20, 50, 100, 200, 500, TOTAL_DIAS_ESPERADO],
    labels=['0 (completo)', '1 dia', '2-5 dias', '6-10 dias', '11-20 dias', '21-50 dias', '51-100 dias', '101-200 dias', '201-500 dias', '> 500 dias']
)
print(faixas_falt.value_counts().sort_index().to_string())

print()
incompletas = contagem[~contagem['completa']].copy()
print(f'Top-10 lojas com MAIS dias faltantes:')
print(incompletas.sort_values('dias_faltantes', ascending=False).head(10)[['loja','dias_registrados','dias_faltantes']].to_string(index=False))

# Salvar para usar nos itens seguintes
contagem.to_csv('data/02_interim/contagem_dias_por_loja.csv', index=False)

# ============================================================
# ITEM 2: Uma loja do grupo incompleto, datas faltantes exatas
# ============================================================
print()
print('=' * 70)
print('ITEM 2: LOJA INCOMPLETA EXEMPLAR - DATAS FALTANTES')
print('=' * 70)

# Escolher a loja do topo da lista (mais dias faltantes, nao numero 1, caso nao exista pegar outra)
if len(incompletas) > 0:
    loja_ex = incompletas.sort_values('dias_faltantes', ascending=False).iloc[0]['loja']
    dias_loja = set(df_treino.loc[df_treino['loja'] == loja_ex, 'data'])
    faltam = sorted(set(CALENDARIO_COMPLETO) - dias_loja)
    print(f'Loja exemplo: #{int(loja_ex)}')
    print(f'Dias registrados : {len(dias_loja)}')
    print(f'Dias faltantes   : {len(faltam)}')
    if len(faltam) > 0:
        # Calcular blocos contiguos faltantes
        blocos = []
        bloco_inicio = faltam[0]
        bloco_fim = faltam[0]
        for d in faltam[1:]:
            if (d - bloco_fim).days == 1:
                bloco_fim = d
            else:
                blocos.append((bloco_inicio, bloco_fim, (bloco_fim - bloco_inicio).days + 1))
                bloco_inicio = d
                bloco_fim = d
        blocos.append((bloco_inicio, bloco_fim, (bloco_fim - bloco_inicio).days + 1))
        print(f'Numero de blocos contiguos faltantes: {len(blocos)}')
        print()
        print(f'{"Bloco":<6} {"Data inicio":<14} {"Data fim":<14} {"Duracao (dias)"}')
        print('-' * 50)
        for i, (ini, fim, n) in enumerate(blocos, 1):
            print(f'{i:<6} {ini.strftime("%d/%m/%Y"):<14} {fim.strftime("%d/%m/%Y"):<14} {n}')

        # Se for muitas datas, mostrar primeiras 20 e ultimas 20
        if len(faltam) > 40:
            print()
            print(f'Primeiras 20 datas faltantes ({faltam[0].strftime("%d/%m/%Y")} a {faltam[19].strftime("%d/%m/%Y")}):')
            print([d.strftime('%Y-%m-%d') for d in faltam[:20]])
            print()
            print(f'Ultimas 20 datas faltantes ({faltam[-20].strftime("%d/%m/%Y")} a {faltam[-1].strftime("%d/%m/%Y")}):')
            print([d.strftime('%Y-%m-%d') for d in faltam[-20:]])
        else:
            print()
            print('Todas as datas faltantes:')
            print([d.strftime('%Y-%m-%d') for d in faltam])

# ============================================================
# ITEM 3: Como os dias faltantes aparecem? nulo, zero ou ausencia?
# ============================================================
print()
print('=' * 70)
print('ITEM 3: COMO OS DIAS FALTANTES APARECEM NA BASE?')
print('=' * 70)

# Para a loja exemplo, verificamos se alguma linha tem faturamento NaN / 0 naqueles dias
loja_ex_df = df_treino[df_treino['loja'] == loja_ex].set_index('data').reindex(CALENDARIO_COMPLETO)
loja_ex_df.index.name = 'data'
loja_ex_df = loja_ex_df.reset_index()

# Agora, para as datas originalmente faltantes no df_treino, como aparecem apos reindex?
faltas_set = set(faltam)
loja_ex_df['_faltava'] = loja_ex_df['data'].isin(faltas_set)

print(f'Para a loja {int(loja_ex)}, apos reindexar com calendario completo:')
print()
# Contar NaN, zero, etc. nas colunas cruciais
cols = ['loja', 'faturamento', 'clientes', 'aberta']
resumo = pd.DataFrame(columns=['NaN','=0','>0 (ou nao-nulo)','Outros'], index=cols)
for c in cols:
    nan_count = loja_ex_df[c].isna().sum()
    zero_count = (loja_ex_df[c] == 0).sum()
    if c == 'loja':
        positive_count = ((~loja_ex_df[c].isna()) & (loja_ex_df[c] != 0)).sum()
    else:
        positive_count = (loja_ex_df[c] > 0).sum()
    outros = len(loja_ex_df) - nan_count - zero_count - positive_count
    resumo.loc[c] = [nan_count, zero_count, positive_count, outros]
print(resumo.to_string())
print()
print(f'CONCLUSAO ITEM 3 (para a loja {int(loja_ex)}):')
print(f'  - Linhas com loja = NaN (ausencia total): {loja_ex_df["loja"].isna().sum()}  '
      f'de {len(loja_ex_df)} total = {loja_ex_df["loja"].isna().sum() / len(loja_ex_df) * 100:.2f}%')
print(f'  - Dentre as {loja_ex_df["_faltava"].sum()} datas que faltavam originalmente:')
print(f'      * Todas elas tem loja NaN? ...................... '
      f'{loja_ex_df.loc[loja_ex_df["_faltava"], "loja"].isna().all()}')

# ============================================================
# ITEM 4: Buraco e igual ou diferente entre lojas incompletas?
# ============================================================
print()
print('=' * 70)
print('ITEM 4: PADRAO DE FALTAS NAS LOJAS INCOMPLETAS - MESMO BURACO?')
print('=' * 70)

# Coletar todas as datas faltantes por loja incompleta
lojas_incompletas_lst = list(incompletas['loja'].values.astype(int))
print(f'Lojas incompletas analisadas: {len(lojas_incompletas_lst)}')
print()

faltas_por_loja = {}
for lj in lojas_incompletas_lst:
    d_loja = set(df_treino.loc[df_treino['loja'] == lj, 'data'])
    faltas_por_loja[lj] = set(CALENDARIO_COMPLETO) - d_loja

# Interseccao de todas as faltas - datas que faltam em TODAS as lojas incompletas
if len(faltas_por_loja) > 1:
    intersecao_total = set.intersection(*faltas_por_loja.values())
    uniao_total = set.union(*faltas_por_loja.values())
else:
    intersecao_total = faltas_por_loja[lojas_incompletas_lst[0]]
    uniao_total = faltas_por_loja[lojas_incompletas_lst[0]]

print(f'Total de datas distintas que faltam na UNIAO das lojas incompletas: {len(uniao_total)}')
print(f'Total de datas que faltam na INTERSECAO (faltam em TODAS as {len(lojas_incompletas_lst)} lojas): {len(intersecao_total)}')
print()

# Mostra intersecao se nao vazia
if len(intersecao_total) > 0:
    lst_inter = sorted(intersecao_total)
    # Agrupa em blocos
    blocos_int = []
    bi, bf = lst_inter[0], lst_inter[0]
    for d in lst_inter[1:]:
        if (d - bf).days == 1:
            bf = d
        else:
            blocos_int.append((bi, bf, (bf - bi).days + 1))
            bi, bf = d, d
    blocos_int.append((bi, bf, (bf - bi).days + 1))
    print(f'Blocos de datas que faltam em TODAS as lojas incompletas:')
    print(f'{"Bloco":<6} {"Inicio":<14} {"Fim":<14} {"Dias"}')
    print('-' * 45)
    for i, (ini, fim, n) in enumerate(blocos_int, 1):
        print(f'{i:<6} {ini.strftime("%d/%m/%Y"):<14} {fim.strftime("%d/%m/%Y"):<14} {n}')
    print()

# Contar quantas lojas tem EXATAMENTE a mesma lista de faltas que a loja exemplo
mesmo_padrao = sum(1 for lj in lojas_incompletas_lst
                   if faltas_por_loja[lj] == faltas_por_loja[loja_ex])
print(f'Lojas com EXATAMENTE o mesmo padrao de faltas da loja #{int(loja_ex)}: {mesmo_padrao} de {len(lojas_incompletas_lst)}')
print()

# Contar quantos padroes distintos existem
padroes = set()
for lj, f in faltas_por_loja.items():
    # representacao canonica do conjunto de dias faltantes como tupla ordenada
    padroes.add(tuple(sorted(f)))
print(f'Padroes DE faltas DISTINTOS entre as {len(lojas_incompletas_lst)} lojas incompletas: {len(padroes)}')

# Para as primeiras 10 lojas, mostrar resumo
if len(lojas_incompletas_lst) > 1:
    print()
    print('Comparativo das primeiras 10 lojas incompletas:')
    top10 = sorted(lojas_incompletas_lst, key=lambda x: -len(faltas_por_loja[x]))[:10]
    comp_df = pd.DataFrame([
        {
            'loja': lj,
            'dias_faltantes': len(faltas_por_loja[lj]),
            'primeira_falta': min(faltas_por_loja[lj]).strftime('%d/%m/%Y') if faltas_por_loja[lj] else '-',
            'ultima_falta': max(faltas_por_loja[lj]).strftime('%d/%m/%Y') if faltas_por_loja[lj] else '-',
            'igual_loja_exemplo?': faltas_por_loja[lj] == faltas_por_loja[loja_ex],
        }
        for lj in top10
    ])
    print(comp_df.to_string(index=False))

# ============================================================
# ITEM 5: Impacto - ranking por soma de faturamento vs media em dia aberto
# ============================================================
print()
print('=' * 70)
print('ITEM 5: IMPACTO NOS RANKINGS DE FATURAMENTO')
print('=' * 70)

# Montar tabela por loja: soma total, media todos os dias, media so dias abertos
rank = df_treino.groupby('loja').agg(
    soma_fat=('faturamento', 'sum'),
    dias_registrados=('data','count'),
    dias_abertos=('aberta', 'sum'),
).reset_index()
rank['media_todos_dias'] = rank['soma_fat'] / TOTAL_DIAS_ESPERADO  # media usando calendario completo (reproduz erro de dias faltantes na soma e na media todos os dias)
rank['media_somente_abertos'] = np.where(rank['dias_abertos'] > 0, rank['soma_fat'] / rank['dias_abertos'], np.nan)

# Integrar com a classificacao completa/incompleta
rank = rank.merge(contagem[['loja','dias_faltantes','completa']], on='loja', how='left')

# Ranking por soma de faturamento (ranking menor = melhor)
rank['rank_soma'] = rank['soma_fat'].rank(ascending=False, method='first').astype(int)
# Ranking por media de faturamento em dia aberto
rank['rank_media_abertos'] = rank['media_somente_abertos'].rank(ascending=False, method='first').astype(int)

# Quartis - pior quartil (ultimo quartil, valores menores)
N_TOTAL = len(rank)
limiar_quartil4 = N_TOTAL * 0.75  # rank > limiar_quartil4 = pior 25%
rank['pior_quartil_soma'] = rank['rank_soma'] > limiar_quartil4
rank['pior_quartil_media_abertos'] = rank['rank_media_abertos'] > limiar_quartil4

esperado_por_acaso = 0.25  # 25%

print(f'Total de lojas: {N_TOTAL}')
print(f'Lojas incompletas       : {(~rank["completa"]).sum()}')
print(f'Lojas completas         : {(rank["completa"]).sum()}')
print(f'Limiar do pior quartil (rank > {limiar_quartil4:.0f}) corresponde ao quartil 4 = piores 25%')
print()

# Contagem para lojas incompletas
incomp = rank[~rank['completa']]
comp = rank[rank['completa']]

pior_soma_inc = incomp['pior_quartil_soma'].sum()
pior_med_inc = incomp['pior_quartil_media_abertos'].sum()
pior_soma_comp = comp['pior_quartil_soma'].sum()
pior_med_comp = comp['pior_quartil_media_abertos'].sum()

resumo_rank = pd.DataFrame([
    {
        'Grupo': 'Lojas INCOMPLETAS',
        'n lojas': len(incomp),
        'no pior quartil (soma)': f'{pior_soma_inc} ({pior_soma_inc/len(incomp)*100:.1f}%)',
        'esperado por acaso': f'{int(round(esperado_por_acaso*len(incomp)))} ({esperado_por_acaso*100:.0f}%)',
        'no pior quartil (media_abertos)': f'{pior_med_inc} ({pior_med_inc/len(incomp)*100:.1f}%)',
    },
    {
        'Grupo': 'Lojas COMPLETAS',
        'n lojas': len(comp),
        'no pior quartil (soma)': f'{pior_soma_comp} ({pior_soma_comp/len(comp)*100:.1f}%)',
        'esperado por acaso': f'{int(round(esperado_por_acaso*len(comp)))} ({esperado_por_acaso*100:.0f}%)',
        'no pior quartil (media_abertos)': f'{pior_med_comp} ({pior_med_comp/len(comp)*100:.1f}%)',
    },
])
print(resumo_rank.to_string(index=False))
print()

# Proporcao excesso (Razao observado / esperado)
razao_soma_inc = (pior_soma_inc / len(incomp)) / esperado_por_acaso
razao_med_inc  = (pior_med_inc  / len(incomp)) / esperado_por_acaso
print(f'Razao (observado / esperado por acaso) para lojas INCOMPLETAS:')
print(f'  Ranking por SOMA DE FATURAMENTO       : {razao_soma_inc:.2f}x')
print(f'  Ranking por MEDIA SO EM DIAS ABERTOS  : {razao_med_inc:.2f}x')
print()
print('(Razao = 1.00x -> mesmo que sorteio. >1.00x -> penalizado. <1.00x -> favorecido.)')

# Detalhe: rank_soma e rank_media_abertos para a loja exemplo
detalhe_ex = rank[rank['loja'] == loja_ex]
if len(detalhe_ex) > 0:
    print()
    print(f'Detalhe da loja #{int(loja_ex)}:')
    print(f'  Soma faturamento R$: {detalhe_ex["soma_fat"].iloc[0]:,.0f}')
    print(f'  Dias faltantes     : {int(detalhe_ex["dias_faltantes"].iloc[0])}')
    print(f'  Rank soma (1=melhor): {detalhe_ex["rank_soma"].iloc[0]:>4} ({detalhe_ex["rank_soma"].iloc[0]/N_TOTAL*100:.1f}% pior)')
    print(f'  Rank media abertos : {detalhe_ex["rank_media_abertos"].iloc[0]:>4} ({detalhe_ex["rank_media_abertos"].iloc[0]/N_TOTAL*100:.1f}% pior)')
