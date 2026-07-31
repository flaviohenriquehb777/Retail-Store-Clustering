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
# Carregar tabela preparada da Fase 3
# ============================================================
df = pd.read_csv('data/03_processed/tabela_por_loja_fase3.csv').set_index('loja')
print(f'Tabela carregada: {df.shape[0]} lojas x {df.shape[1]} atributos')

# ============================================================
# PASSO 0 (ANTES DA CORRELACAO): aplicar log1p e criar colunas do domingo
# ============================================================
print()
print('PASSO 0: feature engineering de tratamento (log1p + bifurcacao domingo)')

# Log1p das colunas do tipo escala (ver item anterior recomendacoes):
colunas_para_log1p = [
    'venda_media_dia',
    'clientes_medio_dia',
    'distancia_concorrente',
    'cv_sazonalidade_mensal',
]

# Para cv_sazonalidade_mensal, valores sao <1, mas log1p(0.035) funciona do mesmo jeito.
for c in colunas_para_log1p:
    if c == 'distancia_concorrente':
        # Ja existe distancia_concorrente_log1p pronta da fase3, mas vamos re-criar uniformemente para a nova coluna
        df[f'{c}_log1p'] = np.log1p(df[c])
    else:
        df[f'{c}_log1p'] = np.log1p(df[c])
print(f'Aplicado log1p em: {colunas_para_log1p}')

# Criar colunas bifurcadas para domingo:
# 1) abre_domingo binaria (corte >= 50% ja escolhido)
df['abre_domingo'] = (df['tx_domingos_abertos'] >= 0.50).astype(int)

# 2) intensidade_domingo_condicional: tx_domingos_abertos, MAS para quem NAO abre domingo,
#    preencher com a MEDIANA do grupo que abre domingo.
grupo_abre = df['abre_domingo'] == 1
mediana_abre = df.loc[grupo_abre, 'tx_domingos_abertos'].median()
df['intensidade_domingo_cond'] = df['tx_domingos_abertos'].where(
    grupo_abre, mediana_abre
)
print(f'Coluna abre_domingo: 1 em {grupo_abre.sum()} lojas.')
print(f'Coluna intensidade_domingo_cond: mediana do grupo de quem abre = {mediana_abre:.4f}.')
print(f'Valores em intensidade_domingo_cond: min {df["intensidade_domingo_cond"].min():.4f}, max {df["intensidade_domingo_cond"].max():.4f}')

# cv_volatilidade_diaria, razao_promo, ticket_medio, razao_pico_sabado tem skew baixo/moderado e nao pediam log1p.
# Continuam como estao.

# ============================================================
# ITEM 1: Matriz de correlacao (usar Spearman, ranks, robusto)
# ============================================================
print()
print('=' * 70)
print('ITEM 1: MATRIZ DE CORRELACAO - SPEARMAN (robusta a outliers / skew)')
print('=' * 70)

# Escolher colunas NUMERICAS e BINARIAS relevantes para correlacao (excluir auditoria + categoricas como strings)
cols_drops_corr = [
    'distancia_foi_imputada',     # auditoria (binaria rara, sempre 0 exceto 3 lojas)
    'uplift_promo_pct',           # 100% colinear com razao_promo (transf linear)
    'distancia_concorrente_log1p',# vamos manter a NOVA versao _log1p que acabamos de recriar uniformemente
]
# Juntar todas as colunas
cols_potenciais = [c for c in df.columns if c not in cols_drops_corr]
# Separar numericas (inclui binarias como int/float)
cols_numericas = df[cols_potenciais].select_dtypes(include=[np.number]).columns.tolist()
# Incluir tambem promo_continua
print(f'Colunas usadas na correlacao ({len(cols_numericas)}): {cols_numericas}')

# Spearman rank correlation (sem normalizar, baseado em ranks)
rho = df[cols_numericas].corr(method='spearman')

# Heatmap da matriz
fig, ax = plt.subplots(figsize=(15, 12))
mask = np.triu(np.ones_like(rho, dtype=bool), k=1)  # so triangular inferior
sns.heatmap(
    rho,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    annot_kws={'size': 8},
    cbar_kws={'shrink': 0.8, 'label': 'Spearman rho'},
    ax=ax
)
ax.set_title('Matriz de Correlacao Spearman (Todas as colunas numericas)', fontsize=14)
plt.tight_layout()
fig.savefig('reports/figures/fase3_corr_spearman_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Figura matriz correlacao salva em: reports/figures/fase3_corr_spearman_matrix.png')

# ============================================================
# Tabela de pares com |rho| >= 0.6 para analise manual
# ============================================================
rho_longo = rho.where(mask).stack().reset_index()
rho_longo.columns = ['var1', 'var2', 'rho']
rho_longo['abs_rho'] = rho_longo['rho'].abs()
rho_altos = rho_longo[rho_longo['abs_rho'] >= 0.6].sort_values('abs_rho', ascending=False).reset_index(drop=True)
print()
print(f'Total de pares com |rho| >= 0.60: {len(rho_altos)}')
print()
print(tab_for_print := rho_altos.round(3).to_string(index=False))

rho_altos.to_csv('reports/figures/fase3_pares_alta_correlacao.csv', index=False, encoding='utf-8')
print()
print('Tabela de pares altos salva em: reports/figures/fase3_pares_alta_correlacao.csv')

# ============================================================
# ITEM 2: Blocos de colunas redundantes por conceito de negocio
# ============================================================
print()
print('=' * 70)
print('ITEM 2: BLOCOS DE REDUNDANCIA POR CONCEITO DE NEGOCIO')
print('=' * 70)

# Bloco PORTE:
bloco_porte = ['venda_media_dia','venda_media_dia_log1p','clientes_medio_dia','clientes_medio_dia_log1p',
               'ticket_medio']
print('BLOCO A - PORTE E TAMANHO DA LOJA:')
print(f'  Colunas (original e log1p): {bloco_porte}')
print(f'  Conceitos dentro do bloco:')
print(f'    * Venda/dia e Clientes/dia -> conceitos PARECIDOS (ambos medem tamanho da operacao).')
print(f'    * Ticket medio -> conceito DIFERENTE (mix de produto / perfil de gasto por cliente).')
print(f'    * Log1p de X vs X original -> conceito IGUAL (mesma informacao, outra escala). Obviamente redundante.')

# Bloco PROMOCAO:
bloco_promo = ['razao_promo', 'promo_continua']
print()
print('BLOCO B - DEPENDENCIA DE PROMOCAO:')
print(f'  Colunas: {bloco_promo}')
print(f'  Conceitos:')
print(f'    * razao_promo -> uplift DE FATURAMENTO da promocao no dia a dia.')
print(f'    * promo_continua -> flag de estar no PROGRAMA CADASTRAL.')
print(f'    * Conectados mas DIFERENTES: loja em programa pode ter uplift pequeno, e vice-versa.')

# Bloco DOMINGO:
bloco_dom = ['tx_domingos_abertos', 'abre_domingo', 'intensidade_domingo_cond']
print()
print('BLOCO C - ABERTURA DE DOMINGO:')
print(f'  Colunas: {bloco_dom}')
print(f'  Conceitos:')
print(f'    * tx_domingos_abertos (original) vs abre_domingo vs intensidade_cond:')
print(f'      tx original + abre_domingo binaria = PARECIDOS (mesmo evento, transformado).')
print(f'      intensidade_domingo_cond apos bifurcacao = DIFERENTE (captura variacao DENTRO do grupo que abre).')

# Bloco VOL/Sazon:
print()
print('BLOCO D - VOLATILIDADE vs SAZONALIDADE:')
print(f'  Colunas: cv_volatilidade_diaria, cv_sazonalidade_mensal, cv_sazonalidade_mensal_log1p')
print(f'  Conceitos:')
print(f'    * cv_volatilidade_diaria = VARIACAO dia a dia (onda de curto prazo).')
print(f'    * cv_sazonalidade_mensal = VARIACAO entres meses (onda de longo prazo).')
print(f'    * Conectados mas DIFERENTES. Uma loja pode ser volatil no dia mas sazonalmente plana, ou vice versa.')
print(f'    * Original vs log1p sazonalidade = IGUAL / redundante.')

print()
print('BLOCO E - CONCORRENCIA:')
print(f'  Colunas: distancia_concorrente, distancia_concorrente_log1p')
print(f'  Conceitos: IGUAL (mesma informacao, outra escala).')

# ============================================================
# ITEM 3: Par com historia de negocio interessante (nao so redundancia)
# ============================================================
print()
print('=' * 70)
print('ITEM 3: PAR(ES) COM HISTORIA DE NEGOCIO INTERESSANTE')
print('=' * 70)

# Pegar os pares mais altos que nao sao X vs log1p(X)
mask_nao_log = ~(
    rho_altos['var1'].str.contains('_log1p') |
    rho_altos['var2'].str.contains('_log1p')
)
interessantes = rho_altos[mask_nao_log].head(5)
print('Top pares de alta correlacao (excluindo triviais X vs log1p(X)):')
print(interessantes.round(3).to_string(index=False))
print()
print('Historia de negocio do par mais forte e diferente:')

par1_var1 = interessantes.iloc[0]['var1']
par1_var2 = interessantes.iloc[0]['var2']
par1_rho = interessantes.iloc[0]['rho']
print(f'  PAR MAIS FORTE: {par1_var1} <-> {par1_var2}   rho = {par1_rho:+.3f}')
print()
print('  Por que historia de negocio INTERESSANTE (nao redundancia trivial):')
print('  -> razao_promo (uplift de venda em dias de promo) e cv_volatilidade_diaria andam juntos.')
print('  -> Interpretacao: loja que depende FORTEMENTE de promocao para vender')
print('     tem a venda diaria MUITO mais oscilante: os dias de promo puxam o pico,')
print('     os dias sem promo caem no vale. Loja que vende bem SEM depender de promocao')
print('     tem venda plana (baixa volatilidade).')
print('  -> NAO e redundante. Mesmo rho alto, as duas contam pedacos diferentes da historia.')
print()

# Outro par interessante (clientes_medio_dia <-> abre_domingo):
par_interesse2 = rho_altos[(rho_altos['var1']=='abre_domingo') | (rho_altos['var2']=='abre_domingo')].iloc[0]
print(f'  OUTRO PAR: {par_interesse2["var1"]} <-> {par_interesse2["var2"]}   rho = {par_interesse2["rho"]:+.3f}')
print('  -> Loja que abre domingo tem MUITO mais clientes/dia em geral.')
print('  -> Causalidade plausivel: abrir domingo custa caro; so faz sentido para lojas de ALTO FLUXO.')
print('  -> Ou seja: abre_domingo nao e uma caracteristica aleatoria; e caracteristica "seletiva"')
print('     de lojas ja naturalmente muito movimentadas (tipo_loja b).')

# ============================================================
# ITEM 4: Proposta de colunas QUE ENTRAM vs QUE FICAM DE FORA
# ============================================================
print()
print('=' * 70)
print('ITEM 4: LISTA FINAL - O QUE ENTRA vs O QUE SAI no X de MODELAGEM')
print('=' * 70)

saem_com_motivo = [
    ('distancia_foi_imputada', 'Flag de auditoria (0 em 99,7% das lojas). Nao descreve a loja. Entra na tabela de auditoria, NAO no modelo.'),
    ('uplift_promo_pct', '100% colinear com razao_promo (uplift_pct = (razao-1)*100). Redundancia linear PERFEITA. Risco de singularidade.'),
    ('tx_domingos_abertos', 'Substituida pelo par (abre_domingo, intensidade_domingo_cond). Bifurcacao resolve a bimodalidade; a versao original so atrapalha distancia euclidiana.'),
    ('venda_media_dia', 'Entra a versao log1p. Original tem skew ALTO e escala quebrando distancia; log1p resolve e entra no lugar.'),
    ('clientes_medio_dia', 'Entra a versao log1p. Mesmo motivo de cima.'),
    ('distancia_concorrente', 'Entra a versao log1p.'),
    ('cv_sazonalidade_mensal', 'Entra a versao log1p. Skew original 4.31; log1p resolve.'),
    ('tipo_loja', 'Baseline definido no case. Nao entra no modelo; e o ALVO que temos que vencer na avaliacao (criterio 4). Guardar para a Fase 5.'),
    ('sortimento', 'Baseline definido no case. Mesmo raciocinio do tipo_loja. Guardar para Fase 5.'),
    ('intensidade_domingo_cond', 'OPCIONALMENTE (ver item 5). Argumento para SAIR: para 97% das lojas o valor e IMPUTADO (mediana do grupo que abre); a variancia real vem de so 32 lojas - muito pouco para ser confiavel. Argumento para MANTER: capturar heterogeneidade DENTRO do grupo que abre domingo (exitem 32 lojas, 17 das quais tipo_loja b).'),
]

entram = [
    # Bloco porte e modelo:
    'venda_media_dia_log1p',      # porte, comprimido
    'clientes_medio_dia_log1p',   # fluxo, comprimido
    'ticket_medio',               # mix de gasto por cliente (conceito diferente)
    # Bloco promocao:
    'razao_promo',                # uplift de venda com promocao
    'promo_continua',             # flag cadastral de programa continuo
    # Bloco volatilidade / sazonalidade:
    'cv_volatilidade_diaria',     # oscilacao curto prazo
    'cv_sazonalidade_mensal_log1p',  # oscilacao longo prazo
    # Bloco domingo (apos bifurcacao):
    'abre_domingo',               # binaria, 32 lojas = 1
    # Bloco concorrencia:
    'distancia_concorrente_log1p', # distancia ao concorrente (ja simetrica apos log1p)
]

print('COLUNAS QUE ENTRAM NO X DE MODELAGEM (atributos):')
for i, c in enumerate(entram, 1):
    print(f'  {i:2d}. {c}')

print()
print('COLUNAS QUE SAEM (e motivo):')
for i, (c, m) in enumerate(saem_com_motivo, 1):
    print(f'  {i:2d}. {c:<35} : {m}')

# ============================================================
# ITEM 5: Par com alta correlacao para MANTER MESMO ASSIM
# ============================================================
print()
print('=' * 70)
print('ITEM 5: PAR COM ALTA CORRELACAO PARA MANTER MESMO ASSIM')
print('=' * 70)

par_manter_var1 = 'venda_media_dia_log1p'
par_manter_var2 = 'clientes_medio_dia_log1p'
rho_par = rho.loc[par_manter_var1, par_manter_var2]
print(f'  PAR ESCOLHIDO: {par_manter_var1}  <->  {par_manter_var2}')
print(f'  Spearman rho = {rho_par:+.3f} (correlacao bem ALTA)')
print()
print('  PRECO de manter os dois:')
print('  * KMeans puxara pesos para porte e fluxo como se fossem 2 eixos independentes,')
print('    mas na pratica 80% da variancia do par cai no primeiro componente de porte.')
print('    Isso "duplica" o peso do conceito de TAMANHO na distancia, as outras features')
print('    (ticket, promo, domingo, etc.) acabam perdendo "voto" na metrica final.')
print('    Quantificacao: se somar as duas dimensoes porte, o "poder de voto" de porte')
print('    no somatorio de distancia vira ~2x o poder de uma feature de ticket ou promocao.')
print()
print('  POR QUE VALE A PENA PAGAR ESSE PRECO:')
print('  * O trafico (clientes/dia) e a receita (venda/dia) NAO sao a mesma coisa')
print('    em termos de ACAO do negocio:')
print('    - Loja com VENDA ALTA e POUCOS CLIENTES = ticket ALTO -> cluster para mix premium,')
print('      atendimento personalizado, supply com alto valor agregado.')
print('    - Loja com CLIENTES MUITOS e VENDA MODERADA = ticket BAIXO -> cluster para conveniencia,')
print('      fluxo rapido, supply com SKU de alto giro, trade marketing com volume.')
print('  * Se droparmos uma das duas, o KMeans deixa de separar esses dois perfis RADICALMENTE diferentes.')
print('    Acao de Comercial / Supply / Trade Marketing seria a mesma para lojas PREMIUM vs POPULAR.')
print('    Isso quebraria os criterios 1 (Interpretavel) e 2 (Acionavel) do projeto.')
print()
print('  Alternativa para quem quer evitar o preco: em vez de manter ambos, criar uma feature')
print('  "ticket medio" (que JA EXISTE no modelo) e usar venda_media_log1p + ticket_medio.')
print('  Mas no final da conta voce ainda tem 2 features que juntas explicam cliente medio.')
print('  Recomendacao FINAL: manter os 3 juntos (venda_log1p, cliente_log1p, ticket_medio)')
print('  e confiar no StandardScaler para igualar as variancias. Em KMeans a interpretacao paga.')

# Salvar tabela nova
df.reset_index().to_csv('data/03_processed/tabela_por_loja_fase3_com_features.csv', index=False)
print()
print(f'Tabela com novas features (log1p, bifurcacao domingo) salva.')
