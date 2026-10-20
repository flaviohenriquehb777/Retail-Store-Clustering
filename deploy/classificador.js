/* ===================================================================
   classificador.js — pipeline idêntico ao predict.py, mas em JS puro.
   NÃO usa sklearn. Implementa o mesmo pipeline numpy (Rota B).
   Resultado = ARI 1.0 vs o KMeans original de referência (1.115 lojas).
   NÃO HÁ RETREINO. Tudo roda no navegador, 100% offline após a página
   carregar (params_modelo.json já embutido neste arquivo para funcionar
   via file:// se necessário).
=================================================================== */

// --- Parâmetros do modelo idênticos ao deploy/params_modelo.json ---
const MODELO = {
  versao: "k5_v2_prof6",
  mediana_dist_concorrente: 2.325,   // metros
  scaler: {
    // Ordem FIXA (igual scaler.feature_names_in_):
    ordem: [
      "clientes_medio_dia_log1p",
      "ticket_medio",
      "razao_promo",
      "cv_volatilidade_diaria",
      "cv_sazonalidade_mensal_log1p",
      "distancia_concorrente_log1p",
    ],
    mean:  [6.5502132356, 9.6437556419, 1.4142931045, 0.2709220248, 0.0906262433, 7.6457360245],
    scale: [0.3725652295, 1.985967194,  0.1809643564, 0.0593830181, 0.03308053,   1.5494502093],
  },
  // Ordem clusters = KMeans 0..4 (índice do array = kmeans_cluster_id)
  clusters: [
    { id_kmeans: 0, ordem_negocio: "C00", nome: "Ticket Premium / Mix Caro",
      n_lojas_originais: 287, pct_rede: 25.7,
      centroide_escalado: [-0.6991852119, 1.2037893525, -0.4070338927, -0.3295425036,  0.0462835519,  0.5836444714],
      perfil_medio_raw: { venda_media_dia:6653.22, clientes_medio_dia:552.59, ticket_medio:12.03, razao_promo:1.3406, cv_volatilidade_diaria:0.2514, cv_sazonalidade_mensal:0.0968, distancia_concorrente_m:8221.1 },
      cor_tag: "C00" },
    { id_kmeans: 1, ordem_negocio: "C01", nome: "Alto Fluxo / Ponto de Passagem",
      n_lojas_originais: 193, pct_rede: 17.3,
      centroide_escalado: [ 1.2829218652,-1.1046381675,-0.41292547,  -0.3665345617,  0.0702351276, -1.3164331725],
      perfil_medio_raw: { venda_media_dia:8816.23, clientes_medio_dia:1211.54, ticket_medio:7.45, razao_promo:1.3396, cv_volatilidade_diaria:0.2492, cv_sazonalidade_mensal:0.0978, distancia_concorrente_m:433.3 },
      cor_tag: "C01" },
    { id_kmeans: 2, ordem_negocio: "C02", nome: "Promo-Dependent",
      n_lojas_originais: 229, pct_rede: 20.5,
      centroide_escalado: [-0.3618648098,-0.0621546771, 1.4051950784, 1.2931949221, -0.1410594022,  0.129802341 ],
      perfil_medio_raw: { venda_media_dia:5967.91, clientes_medio_dia:630.04, ticket_medio:9.52, razao_promo:1.6686, cv_volatilidade_diaria:0.3477, cv_sazonalidade_mensal:0.0899, distancia_concorrente_m:6904.3 },
      cor_tag: "C02" },
    { id_kmeans: 3, ordem_negocio: "C03", nome: "Sazonal Raro / Temporada",
      n_lojas_originais: 49,  pct_rede: 4.4,
      centroide_escalado: [-0.2642011764,-0.3660823788,-0.3016855583, 1.2355488647,  3.2185358901,  0.1940335667],
      perfil_medio_raw: { venda_media_dia:5737.92, clientes_medio_dia:669.45, ticket_medio:8.92, razao_promo:1.3597, cv_volatilidade_diaria:0.3443, cv_sazonalidade_mensal:0.2200, distancia_concorrente_m:9061.0 },
      cor_tag: "C03" },
    { id_kmeans: 4, ordem_negocio: "C04", nome: "Standard / Classe Média",
      n_lojas_originais: 357, pct_rede: 32.0,
      centroide_escalado: [ 0.1369051398,-0.2804507569,-0.309506268, -0.5360326683, -0.4264549427,  0.1325861572],
      perfil_medio_raw: { venda_media_dia:6928.01, clientes_medio_dia:761.45, ticket_medio:9.09, razao_promo:1.3583, cv_volatilidade_diaria:0.2391, cv_sazonalidade_mensal:0.0797, distancia_concorrente_m:4339.1 },
      cor_tag: "C04" },
  ],
};

// Math.log1p existe em ES6+ (IE11 é o único que não tem; ninguém usa mais).
const log1p = Math.log1p || ((x) => Math.log(1 + x));

// ============================================================
// FUNÇÕES DO PIPELINE
// ============================================================
/**
 * Recebe uma loja raw (obj {clientes_medio_dia, ticket_medio, ...})
 * e retorna classificação completa.
 */
function classificarLoja(lojaRaw) {
  const err = validarLojaRaw(lojaRaw);
  if (err) return { erro: err };

  // 1) Validações + imputação distância
  // Trata string vazia / null / undefined / whitespace como distância ausente (caso comum em CSV vazio)
  let distBruta = lojaRaw.distancia_concorrente;
  if (typeof distBruta === "string") distBruta = distBruta.trim() === "" ? NaN : Number(distBruta);
  else distBruta = Number(distBruta);
  // Importante: null >= 0 retorna true em JS e isFinite(null) = true → não pega null.
  // Por isso usamos Number() primeiro, mas já setamos para NaN quando string vazia.
  const distancia_foi_imputada = Number.isNaN(distBruta) || !(distBruta >= 0) || !isFinite(distBruta);
  let dist = distancia_foi_imputada ? MODELO.mediana_dist_concorrente : distBruta;

  const raw = {
    clientes_medio_dia:      Math.max(0, Number(lojaRaw.clientes_medio_dia)),
    ticket_medio:            Math.max(1e-6, Number(lojaRaw.ticket_medio)),
    razao_promo:             Math.max(0, Number(lojaRaw.razao_promo)),
    cv_volatilidade_diaria:  Math.max(0, Number(lojaRaw.cv_volatilidade_diaria)),
    cv_sazonalidade_mensal:  Math.max(0, Number(lojaRaw.cv_sazonalidade_mensal)),
    distancia_concorrente:   Math.max(0, dist),
  };

  // 2) Transformação (log1p)
  const X = [
    log1p(raw.clientes_medio_dia),
    raw.ticket_medio,
    raw.razao_promo,
    raw.cv_volatilidade_diaria,
    log1p(raw.cv_sazonalidade_mensal),
    log1p(raw.distancia_concorrente),
  ];

  // 3) Padronização (z-score com scaler carregado)
  const mean = MODELO.scaler.mean;
  const scl  = MODELO.scaler.scale;
  const Xs = new Array(6);
  for (let j = 0; j < 6; j++) Xs[j] = (X[j] - mean[j]) / scl[j];

  // 4) Distância Euclidiana a cada centróide
  const dists = new Array(5);
  let minIdx = 0, minVal = Infinity;
  for (let i = 0; i < 5; i++) {
    const c = MODELO.clusters[i].centroide_escalado;
    let d2 = 0;
    for (let j = 0; j < 6; j++) {
      const diff = Xs[j] - c[j];
      d2 += diff * diff;
    }
    const d = Math.sqrt(d2);
    dists[i] = d;
    if (d < minVal) { minVal = d; minIdx = i; }
  }
  // Dist média a todos os centroides (para normalizar limiar outlier)
  let soma = 0;
  for (let i = 0; i < 5; i++) soma += dists[i];
  const mediaDists = soma / 5;
  const LIMIAR_B_ABSOLUTO = 4.0;
  const flagOutlier = minVal > 3.0 * mediaDists || minVal > LIMIAR_B_ABSOLUTO;

  const cluster = MODELO.clusters[minIdx];

  // Retorna tudo
  return {
    id_loja: lojaRaw.id_loja ?? lojaRaw.loja ?? lojaRaw.id ?? null,
    nome_loja: lojaRaw.nome_loja ?? lojaRaw.nome ?? null,
    raw,
    distancia_foi_imputada,
    X_features_transformadas: X,
    X_escalado: Xs,
    kmeans_cluster_id: cluster.id_kmeans,
    ordem_negocio: cluster.ordem_negocio,
    nome_perfil: cluster.nome,
    cor_tag: cluster.cor_tag,
    distancia_ao_centroide: minVal,
    media_distancia_a_todos_centroides: mediaDists,
    flag_outlier_fora_do_espaco: flagOutlier,
    dist_por_ordem_negocio: {
      "C00": dists[0], "C01": dists[1], "C02": dists[2], "C03": dists[3], "C04": dists[4],
    },
  };
}

function validarLojaRaw(l) {
  const obrigatorias = ["clientes_medio_dia","ticket_medio","razao_promo","cv_volatilidade_diaria","cv_sazonalidade_mensal"];
  for (const c of obrigatorias) {
    const v = Number(l[c]);
    if (!(v >= 0 && isFinite(v))) {
      return `Coluna obrigatória ausente ou inválida: ${c} (recebi: ${JSON.stringify(l[c])}).`;
    }
  }
  // distancia_concorrente pode ser null/NaN: imputamos
  return null;
}

// ============================================================
// FUNÇÕES CSV (parse minúsculo, aceita separadores ; e ,)
// ============================================================
function parseCSV(text) {
  // Detecta separador: conta , e ; na primeira linha não vazia
  const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim().length);
  if (!lines.length) return [];
  const header = lines[0];
  const sep = (header.match(/;/g) || []).length >= (header.match(/,/g) || []).length ? ";" : ",";
  const cab = splitCsvLine(header, sep).map((s) => normalizaCol(s));
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCsvLine(lines[i], sep);
    if (cols.length === 0) continue;
    const obj = {};
    for (let j = 0; j < cab.length; j++) obj[cab[j]] = (cols[j] ?? "").trim();
    rows.push(obj);
  }
  return rows;
}
function normalizaCol(s) {
  return s.trim()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\s+/g, "_");
}
function splitCsvLine(line, sep) {
  const out = [];
  let cur = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuote) {
      if (c === '"') {
        if (line[i+1] === '"') { cur += '"'; i++; }
        else inQuote = false;
      } else cur += c;
    } else {
      if (c === '"') inQuote = true;
      else if (c === sep) { out.push(cur); cur = ""; }
      else cur += c;
    }
  }
  out.push(cur);
  return out;
}
function toCSV(rows) {
  if (!rows.length) return "";
  const cab = Object.keys(rows[0]);
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    if (/[",;\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  return [cab.join(";"), ...rows.map((r) => cab.map((c) => esc(r[c])).join(";"))].join("\n");
}

// ============================================================
// UI helpers
// ============================================================
function fmtNum(n, d=2) { if (n === null || n === undefined || isNaN(n)) return "—"; return Number(n).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }); }
function fmtPct(n, d=1) { return fmtNum(n*100, d) + "%"; }
function $id(id) { return document.getElementById(id); }
