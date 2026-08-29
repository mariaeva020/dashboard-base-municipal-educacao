# ============================================================
# DASHBOARD DESCRITIVO DAS BASES MUNICIPAIS
# Base longitudinal municipal brasileira aplicada
# à análise educacional
# ============================================================

from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard Descritivo das Bases Municipais",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 8px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.90rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }

    div[data-testid="stTabs"] button {
        font-size: 0.90rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CAMINHOS
# ============================================================

PASTA_APP = Path(
    __file__
).resolve().parent

PASTA_DADOS = (
    PASTA_APP
    /
    "dados"
)

CAMINHO_CATALOGO = (
    PASTA_DADOS
    /
    "catalogo_dados.csv"
)

CAMINHO_CATALOGO_VARIAVEIS = (
    PASTA_DADOS
    /
    "catalogo_variaveis.csv"
)


# ============================================================
# 4. VARIÁVEIS DE IDENTIFICAÇÃO
# ============================================================

COLUNAS_IDENTIFICACAO = {
    "ano",
    "cod_uf",
    "sigla_uf",
    "cod_municipio",
    "nome_municipio",
    "rede"
}

COLUNAS_ID_NUMERICAS = {
    "ano",
    "cod_uf",
    "cod_municipio"
}


# ============================================================
# PADRÃO VISUAL E FORMATAÇÃO DO DASHBOARD
# ============================================================

# ------------------------------------------------------------
# PALETA DE CORES
# Cores discretas, legíveis e visualmente coerentes.
# ------------------------------------------------------------

COR_PRIMARIA = "#356859"       # verde petróleo
COR_SECUNDARIA = "#D5A253"     # ocre
COR_TERCIARIA = "#806491"      # violeta suave
COR_QUATERNARIA = "#C66B52"    # terracota
COR_QUINARIA = "#66788A"       # azul acinzentado
COR_SEXTA = "#8A8E5A"          # oliva
COR_NEUTRA = "#73777B"

PALETA_CORES = [
    COR_PRIMARIA,
    COR_SECUNDARIA,
    COR_TERCIARIA,
    COR_QUATERNARIA,
    COR_QUINARIA,
    COR_SEXTA
]

px.defaults.color_discrete_sequence = PALETA_CORES


# ------------------------------------------------------------
# ESCALAS PARA HEATMAPS
# ------------------------------------------------------------

ESCALA_SEQUENCIAL = [
    [0.00, "#F7F3EA"],
    [0.25, "#E8D8BC"],
    [0.50, "#D5A253"],
    [0.75, "#C47A55"],
    [1.00, "#8B4D45"]
]

ESCALA_DIVERGENTE = [
    [0.00, "#8C5365"],
    [0.25, "#C79AA6"],
    [0.50, "#F7F3EA"],
    [0.75, "#88AFA4"],
    [1.00, "#356859"]
]


# ------------------------------------------------------------
# CORES ESTRUTURAIS
# ------------------------------------------------------------

COR_EIXO = "#6B7280"
COR_TEXTO = "#2F3437"
COR_BORDA = "#D7DCE1"
COR_FUNDO = "#FFFFFF"


# ============================================================
# NOMES AMIGÁVEIS DAS VARIÁVEIS
# ============================================================

SIGLAS = {
    "ideb": "IDEB",
    "saeb": "SAEB",
    "uf": "UF",
    "pib": "PIB",
    "bpc": "BPC",
    "tdi": "TDI",
    "had": "HAD",
    "atu": "ATU",
    "ied": "IED",
    "afd": "AFD",
    "ird": "IRD",
    "icg": "ICG",
    "cv": "CV",
    "iqr": "IQR",
    "ibge": "IBGE",
    "cgu": "CGU"
}


SUBSTITUICOES_TERMOS = {
    "qtd": "Quantidade",
    "pct": "%",
    "cod": "Código",
    "media": "Média",
    "mediana": "Mediana",
    "minimo": "Mínimo",
    "maximo": "Máximo",
    "desvio": "Desvio",
    "padrao": "padrão",
    "aprovacao": "aprovação",
    "matematica": "matemática",
    "portugues": "português",
    "populacao": "população",
    "educacao": "educação",
    "producao": "produção",
    "importacao": "importação",
    "exportacao": "exportação",
    "municipio": "município",
    "municipios": "municípios",
    "variavel": "variável",
    "variaveis": "variáveis",
    "correlacao": "correlação",
    "assimetria": "Assimetria",
    "curtose": "Curtose"
}


def nome_amigavel(
    nome
):
    """
    Converte nomes técnicos para nomes adequados à exibição.

    Exemplo:
    nota_matematica_saeb
    -> Nota matemática SAEB
    """

    nome = str(
        nome
    ).strip()

    # Casos especiais completos
    especiais = {
        "cod_municipio":
            "Código do município",

        "nome_municipio":
            "Município",

        "sigla_uf":
            "UF",

        "cod_uf":
            "Código da UF",

        "taxa_aprovacao_etapa":
            "Taxa de aprovação da etapa",

        "indicador_rendimento":
            "Indicador de rendimento",

        "nota_media_padronizada":
            "Nota média padronizada",

        "nota_matematica_saeb":
            "Nota de matemática no SAEB",

        "nota_portugues_saeb":
            "Nota de português no SAEB",

        "qtd_registros":
            "Quantidade de registros",

        "qtd_municipios":
            "Quantidade de municípios",

        "pct_ausentes":
            "Ausentes (%)",

        "pct_zeros":
            "Zeros (%)",

        "retencao_pct":
            "Retenção (%)"
    }

    if nome.lower() in especiais:

        return especiais[
            nome.lower()
        ]

    partes = nome.replace(
        "_",
        " "
    ).split()

    resultado = []

    for parte in partes:

        parte_lower = (
            parte.lower()
        )

        if parte_lower in SIGLAS:

            resultado.append(
                SIGLAS[
                    parte_lower
                ]
            )

        elif parte_lower in SUBSTITUICOES_TERMOS:

            resultado.append(
                SUBSTITUICOES_TERMOS[
                    parte_lower
                ]
            )

        else:

            resultado.append(
                parte_lower
            )

    texto = " ".join(
        resultado
    )

    # Inicial maiúscula apenas quando não começa por sigla.
    primeira_palavra = texto.split()[0]

    if primeira_palavra not in (
        SIGLAS.values()
    ):

        texto = (
            texto[0].upper()
            +
            texto[1:]
        )

    return texto


# ============================================================
# FORMATAÇÃO NUMÉRICA BRASILEIRA
# ============================================================

def formatar_inteiro_br(
    valor
):

    if pd.isna(
        valor
    ):

        return "—"

    return (
        f"{int(round(valor)):,}"
        .replace(
            ",",
            "."
        )
    )


def formatar_decimal_br(
    valor,
    casas=2
):

    if pd.isna(
        valor
    ):

        return "—"

    texto = (
        f"{float(valor):,.{casas}f}"
    )

    texto = (
        texto
        .replace(
            ",",
            "X"
        )
        .replace(
            ".",
            ","
        )
        .replace(
            "X",
            "."
        )
    )

    return texto


def serie_e_inteira(
    serie
):

    valores = (
        pd.to_numeric(
            serie,
            errors="coerce"
        )
        .dropna()
    )

    if valores.empty:

        return False

    return bool(
        np.allclose(
            valores,
            np.round(
                valores
            )
        )
    )


def casas_decimais_variavel(
    df,
    variavel
):

    if variavel not in df.columns:

        return 2

    serie = df[
        variavel
    ]

    if serie_e_inteira(
        serie
    ):

        return 0

    return 2


# ============================================================
# FORMATAÇÃO PADRONIZADA DAS TABELAS
# ============================================================

def preparar_tabela(
    df
):

    tabela = df.copy()
    # Identificadores não recebem formatação numérica quantitativa.
    for coluna_id in [
        "ano",
        "cod_uf",
        "cod_municipio"
    ]:
        if coluna_id in tabela.columns:

            tabela[coluna_id] = (
                pd.to_numeric(
                    tabela[coluna_id],
                    errors="coerce"
                )
                .astype("Int64")
                .astype("string")
            )

    # ========================================================
    # NOMES DE VARIÁVEIS DENTRO DAS TABELAS
    # ========================================================

    colunas_com_nomes_variaveis = [
        "variavel",
        "Variável",
        "Variável 1",
        "Variável 2"
    ]

    for coluna_variavel in colunas_com_nomes_variaveis:

        if coluna_variavel in tabela.columns:

            tabela[coluna_variavel] = (
                tabela[coluna_variavel]
                .apply(
                    lambda valor:
                        nome_amigavel(valor)
                        if pd.notna(valor)
                        else valor
                )
            )
    tabela = tabela.rename(
        columns={
            coluna:
                nome_amigavel(
                    coluna
                )
            for coluna in tabela.columns
        }
    )

    formatadores = {}

    for coluna in tabela.columns:

        if not pd.api.types.is_numeric_dtype(
            tabela[
                coluna
            ]
        ):

            continue

        nome_lower = (
            coluna.lower()
        )

        if (
            "%"
            in coluna
            or
            "percentual"
            in nome_lower
        ):

            formatadores[
                coluna
            ] = lambda x: (
                formatar_decimal_br(
                    x,
                    2
                )
            )

        elif serie_e_inteira(
            tabela[
                coluna
            ]
        ):

            formatadores[
                coluna
            ] = formatar_inteiro_br

        else:

            formatadores[
                coluna
            ] = lambda x: (
                formatar_decimal_br(
                    x,
                    2
                )
            )

    estilo = (
        tabela
        .style
        .format(
            formatadores,
            na_rep="—"
        )
        .set_properties(
            **{
                "text-align":
                    "center"
            }
        )
        .set_table_styles(
            [
                {
                    "selector":
                        "th",

                    "props": [
                        (
                            "text-align",
                            "center"
                        ),
                        (
                            "font-weight",
                            "600"
                        ),
                        (
                            "background-color",
                            "#F4F5F6"
                        )
                    ]
                },

                {
                    "selector":
                        "td",

                    "props": [
                        (
                            "text-align",
                            "center"
                        )
                    ]
                }
            ]
        )
    )

    return estilo


def exibir_tabela(
    df,
    altura=None
):

    argumentos = {
        "use_container_width":
            True,

        "hide_index":
            True
    }

    if altura is not None:

        argumentos[
            "height"
        ] = altura

    st.dataframe(
        preparar_tabela(
            df
        ),
        **argumentos
    )


# ============================================================
# ESTILO PADRÃO DOS GRÁFICOS
# ============================================================

def aplicar_estilo_grafico(
    fig,
    titulo,
    titulo_x=None,
    titulo_y=None,
    altura=500,
    legenda=True
):

    fig.update_layout(
        title={
            "text":
                titulo,

            "x":
                0.01,

            "xanchor":
                "left",

            "font": {
                "size":
                    19,

                "color":
                    COR_TEXTO
            }
        },

        template=
            "plotly_white",

        paper_bgcolor=
            COR_FUNDO,

        plot_bgcolor=
            COR_FUNDO,

        font={
            "family":
                "Arial",

            "size":
                13,

            "color":
                COR_TEXTO
        },

        height=
            altura,

        separators=
            ",.",

        margin=dict(
            l=60,
            r=30,
            t=75,
            b=60
        ),

        showlegend=
            legenda,

        legend=dict(
            orientation=
                "h",

            yanchor=
                "bottom",

            y=
                1.02,

            xanchor=
                "right",

            x=
                1
        )
    )

    fig.update_xaxes(
        title_text=
            titulo_x,

        showgrid=
            False,

        zeroline=
            False,

        showline=
            True,

        linecolor=
            COR_EIXO,

        linewidth=
            1,

        ticks=
            "outside",

        tickcolor=
            COR_EIXO,

        automargin=
            True
    )

    fig.update_yaxes(
        title_text=
            titulo_y,

        showgrid=
            False,

        zeroline=
            False,

        showline=
            True,

        linecolor=
            COR_EIXO,

        linewidth=
            1,

        ticks=
            "outside",

        tickcolor=
            COR_EIXO,

        automargin=
            True
    )

    return fig


# ============================================================
# RÓTULOS DOS GRÁFICOS
# ============================================================

def adicionar_rotulos_barras(
    fig,
    orientacao="vertical",
    casas=0,
    percentual=False
):

    formato = (
        f",.{casas}f"
    )

    sufixo = (
        "%"
        if percentual
        else ""
    )

    if orientacao == "horizontal":

        template = (
            f"%{{x:{formato}}}"
            f"{sufixo}"
        )

    else:

        template = (
            f"%{{y:{formato}}}"
            f"{sufixo}"
        )

    fig.update_traces(
        texttemplate=
            template,

        textposition=
            "outside",

        cliponaxis=
            False,

        textfont_size=
            11
    )

    return fig


def adicionar_rotulos_linha(
    fig,
    casas=0,
    percentual=False
):

    formato = (
        f",.{casas}f"
    )

    sufixo = (
        "%"
        if percentual
        else ""
    )

    fig.update_traces(
        mode=
            "lines+markers+text",

        texttemplate=(
            f"%{{y:{formato}}}"
            f"{sufixo}"
        ),

        textposition=
            "top center"
    )

    return fig


# ============================================================
# 5. VALIDAÇÃO DOS ARQUIVOS
# ============================================================

if not PASTA_DADOS.exists():

    st.error(
        "A pasta 'dados' não foi encontrada."
    )

    st.stop()


if not CAMINHO_CATALOGO.exists():

    st.error(
        "O arquivo catalogo_dados.csv não foi encontrado."
    )

    st.stop()


if not CAMINHO_CATALOGO_VARIAVEIS.exists():

    st.error(
        "O arquivo catalogo_variaveis.csv não foi encontrado."
    )

    st.stop()


# ============================================================
# 6. CARREGAMENTO
# ============================================================

@st.cache_data
def carregar_catalogo():

    return pd.read_csv(
        CAMINHO_CATALOGO
    )


@st.cache_data
def carregar_catalogo_variaveis():

    return pd.read_csv(
        CAMINHO_CATALOGO_VARIAVEIS
    )


@st.cache_data(
    show_spinner=False
)
def carregar_base(
    arquivo
):

    caminho = (
        PASTA_DADOS
        /
        arquivo
    )

    df = pd.read_parquet(
        caminho
    )

    if "ano" in df.columns:

        df["ano"] = pd.to_numeric(
            df["ano"],
            errors="coerce"
        ).astype(
            "Int64"
        )

    if "cod_municipio" in df.columns:

        df["cod_municipio"] = pd.to_numeric(
            df["cod_municipio"],
            errors="coerce"
        ).astype(
            "Int64"
        )

    if "sigla_uf" in df.columns:

        df["sigla_uf"] = (
            df["sigla_uf"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    return df


catalogo = carregar_catalogo()

catalogo_variaveis = (
    carregar_catalogo_variaveis()
)


# ============================================================
# 7. FUNÇÕES AUXILIARES
# ============================================================

def formatar_inteiro(
    valor
):

    if pd.isna(
        valor
    ):
        return "-"

    return (
        f"{int(valor):,}"
        .replace(
            ",",
            "."
        )
    )


def formatar_percentual(
    valor
):

    if pd.isna(
        valor
    ):
        return "-"

    return (
        f"{valor:.2f}%"
        .replace(
            ".",
            ","
        )
    )


def colunas_numericas(
    df
):

    return [
        coluna
        for coluna in df.columns
        if (
            pd.api.types.is_numeric_dtype(
                df[coluna]
            )
            and
            coluna not in COLUNAS_ID_NUMERICAS
        )
    ]


def aplicar_filtros(
    df,
    ano=None,
    uf=None
):

    resultado = df.copy()

    if (
        ano is not None
        and
        "ano" in resultado.columns
    ):

        resultado = resultado[
            resultado["ano"]
            ==
            ano
        ]

    if (
        uf is not None
        and
        "sigla_uf" in resultado.columns
    ):

        resultado = resultado[
            resultado["sigla_uf"]
            ==
            uf
        ]

    return resultado


def identificar_ideb(
    df
):

    candidatas = [
        coluna
        for coluna in df.columns
        if (
            coluna == "ideb"
            or
            coluna.startswith(
                "ideb_"
            )
        )
    ]

    if not candidatas:
        return None

    return candidatas[0]


def identificar_componentes_ideb(
    df
):

    prefixos = [
        "taxa_aprovacao_",
        "indicador_rendimento_",
        "nota_media_padronizada_",
        "nota_matematica_saeb_",
        "nota_portugues_saeb_"
    ]

    return [
        coluna
        for coluna in df.columns
        if any(
            coluna.startswith(
                prefixo
            )
            for prefixo in prefixos
        )
    ]


# ============================================================
# 8. PERFIL DAS VARIÁVEIS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def calcular_perfil(
    df
):

    registros = []

    total = len(
        df
    )

    for coluna in df.columns:

        serie = df[
            coluna
        ]

        registro = {
            "Variável":
                coluna,

            "Tipo":
                str(
                    serie.dtype
                ),

            "Válidos":
                int(
                    serie.notna()
                    .sum()
                ),

            "Ausentes":
                int(
                    serie.isna()
                    .sum()
                ),

            "Ausentes (%)":
                (
                    serie.isna()
                    .mean()
                    *
                    100
                ),

            "Valores únicos":
                int(
                    serie.nunique(
                        dropna=True
                    )
                )
        }

        if pd.api.types.is_numeric_dtype(
            serie
        ):

            numerica = pd.to_numeric(
                serie,
                errors="coerce"
            )

            registro[
                "Zeros"
            ] = int(
                (
                    numerica
                    ==
                    0
                ).sum()
            )

            registro[
                "Zeros (%)"
            ] = (
                (
                    numerica
                    ==
                    0
                ).sum()
                /
                total
                *
                100
                if total > 0
                else np.nan
            )

            registro[
                "Negativos"
            ] = int(
                (
                    numerica
                    <
                    0
                ).sum()
            )

        else:

            registro[
                "Zeros"
            ] = np.nan

            registro[
                "Zeros (%)"
            ] = np.nan

            registro[
                "Negativos"
            ] = np.nan

        registros.append(
            registro
        )

    return pd.DataFrame(
        registros
    )


# ============================================================
# 9. ESTATÍSTICAS NUMÉRICAS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def calcular_estatisticas(
    df
):

    registros = []

    for coluna in colunas_numericas(
        df
    ):

        serie = (
            pd.to_numeric(
                df[
                    coluna
                ],
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .astype(float)
        )

        if serie.empty:
            continue

        q1 = serie.quantile(
            0.25
        )

        q3 = serie.quantile(
            0.75
        )

        iqr = q3 - q1

        limite_inferior = (
            q1
            -
            1.5
            *
            iqr
        )

        limite_superior = (
            q3
            +
            1.5
            *
            iqr
        )

        outliers = (
            (
                serie
                <
                limite_inferior
            )
            |
            (
                serie
                >
                limite_superior
            )
        )

        media = serie.mean()

        dp = serie.std(
            ddof=1
        )

        cv = (
            dp
            /
            abs(
                media
            )
            *
            100
            if abs(
                media
            )
            >
            1e-12
            else np.nan
        )

        registros.append({
            "Variável":
                coluna,

            "N":
                len(
                    serie
                ),

            "Média":
                media,

            "Desvio-padrão":
                dp,

            "CV (%)":
                cv,

            "Mínimo":
                serie.min(),

            "P1":
                serie.quantile(
                    0.01
                ),

            "P5":
                serie.quantile(
                    0.05
                ),

            "Q1":
                q1,

            "Mediana":
                serie.median(),

            "Q3":
                q3,

            "P95":
                serie.quantile(
                    0.95
                ),

            "P99":
                serie.quantile(
                    0.99
                ),

            "Máximo":
                serie.max(),

            "IQR":
                iqr,

            "Assimetria":
                (
                    serie.skew()
                    if len(
                        serie
                    )
                    >=
                    3
                    else np.nan
                ),

            "Curtose":
                (
                    serie.kurt()
                    if len(
                        serie
                    )
                    >=
                    4
                    else np.nan
                ),

            "Outliers IQR":
                int(
                    outliers.sum()
                ),

            "Outliers IQR (%)":
                (
                    outliers.mean()
                    *
                    100
                )
        })

    return pd.DataFrame(
        registros
    )


# ============================================================
# 10. OUTLIERS POR ANO
# ============================================================

def identificar_outliers(
    df,
    variavel
):

    if variavel not in df.columns:
        return pd.DataFrame()

    resultados = []

    if "ano" in df.columns:

        grupos = df.groupby(
            "ano",
            dropna=False
        )

    else:

        grupos = [
            (
                "Total",
                df
            )
        ]

    for ano, grupo in grupos:

        serie = pd.to_numeric(
            grupo[
                variavel
            ],
            errors="coerce"
        )

        validos = serie.dropna()

        if validos.empty:
            continue

        q1 = validos.quantile(
            0.25
        )

        q3 = validos.quantile(
            0.75
        )

        iqr = q3 - q1

        li = (
            q1
            -
            1.5
            *
            iqr
        )

        ls = (
            q3
            +
            1.5
            *
            iqr
        )

        mascara = (
            (
                serie
                <
                li
            )
            |
            (
                serie
                >
                ls
            )
        )

        if not mascara.any():
            continue

        colunas = [
            coluna
            for coluna in [
                "ano",
                "sigla_uf",
                "cod_municipio",
                "nome_municipio",
                variavel
            ]
            if coluna in grupo.columns
        ]

        temp = grupo.loc[
            mascara,
            colunas
        ].copy()

        temp[
            "limite_inferior_iqr"
        ] = li

        temp[
            "limite_superior_iqr"
        ] = ls

        resultados.append(
            temp
        )

    if not resultados:

        return pd.DataFrame()

    return pd.concat(
        resultados,
        ignore_index=True
    )


# ============================================================
# 11. DIAGNÓSTICO DE PRÉ-PROCESSAMENTO
# ============================================================

def diagnostico_preprocessamento(
    df
):

    perfil = calcular_perfil(
        df
    )

    estat = calcular_estatisticas(
        df
    )

    resultado = perfil.merge(
        estat[
            [
                coluna
                for coluna in [
                    "Variável",
                    "CV (%)",
                    "Assimetria",
                    "Curtose",
                    "Outliers IQR (%)"
                ]
                if coluna
                in estat.columns
            ]
        ],
        on="Variável",
        how="left"
    )

    registros = []

    for _, linha in resultado.iterrows():

        variavel = linha[
            "Variável"
        ]

        if variavel in COLUNAS_IDENTIFICACAO:
            continue

        alertas = []

        prioridade = "Baixa"

        pct_ausentes = linha[
            "Ausentes (%)"
        ]

        pct_zeros = linha.get(
            "Zeros (%)",
            np.nan
        )

        assimetria = linha.get(
            "Assimetria",
            np.nan
        )

        pct_outliers = linha.get(
            "Outliers IQR (%)",
            np.nan
        )

        if pct_ausentes >= 50:

            alertas.append(
                "ausência muito elevada"
            )

            prioridade = "Alta"

        elif pct_ausentes >= 20:

            alertas.append(
                "ausência elevada"
            )

            prioridade = "Média"

        elif pct_ausentes >= 5:

            alertas.append(
                "ausência moderada"
            )

        if linha[
            "Valores únicos"
        ] <= 1:

            alertas.append(
                "variável constante"
            )

            prioridade = "Alta"

        if (
            pd.notna(
                assimetria
            )
            and
            abs(
                assimetria
            )
            >=
            2
        ):

            alertas.append(
                "assimetria elevada"
            )

            if prioridade == "Baixa":
                prioridade = "Média"

        if (
            pd.notna(
                pct_zeros
            )
            and
            pct_zeros
            >=
            50
        ):

            alertas.append(
                "alta concentração de zeros"
            )

        if (
            pd.notna(
                pct_outliers
            )
            and
            pct_outliers
            >=
            5
        ):

            alertas.append(
                "frequência elevada de valores extremos"
            )

        if not alertas:

            alertas.append(
                "nenhum alerta automático identificado"
            )

        registros.append({
            "Variável":
                variavel,

            "Prioridade":
                prioridade,

            "Ausentes (%)":
                pct_ausentes,

            "Zeros (%)":
                pct_zeros,

            "Assimetria":
                assimetria,

            "Outliers IQR (%)":
                pct_outliers,

            "Diagnóstico":
                "; ".join(
                    alertas
                )
        })

    resultado = pd.DataFrame(
        registros
    )

    if resultado.empty:
        return resultado

    ordem = {
        "Alta": 1,
        "Média": 2,
        "Baixa": 3
    }

    resultado[
        "_ordem"
    ] = (
        resultado[
            "Prioridade"
        ]
        .map(
            ordem
        )
    )

    return (
        resultado
        .sort_values(
            [
                "_ordem",
                "Ausentes (%)"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop(
            columns="_ordem"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 12. CONVERTER DATAFRAME PARA PARQUET
# ============================================================

def dataframe_para_parquet(
    df
):

    buffer = BytesIO()

    df.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    return buffer.getvalue()


# ============================================================
# 13. FILTROS PRINCIPAIS
# ============================================================

st.sidebar.title(
    "Filtros"
)


tipos = (
    catalogo[
        "tipo_base"
    ]
    .dropna()
    .unique()
    .tolist()
)


tipo_base = st.sidebar.selectbox(
    "Tipo de base",
    options=tipos,
    index=(
        tipos.index(
            "Painéis integrados completos"
        )
        if
        "Painéis integrados completos"
        in tipos
        else
        0
    )
)


catalogo_tipo = catalogo[
    catalogo[
        "tipo_base"
    ]
    ==
    tipo_base
]


bases = (
    catalogo_tipo[
        "base"
    ]
    .dropna()
    .unique()
    .tolist()
)


nome_base = st.sidebar.selectbox(
    "Base",
    options=bases
)


registro = (
    catalogo_tipo[
        catalogo_tipo[
            "base"
        ]
        ==
        nome_base
    ]
    .iloc[0]
)


df_original = carregar_base(
    registro[
        "arquivo"
    ]
)


# ============================================================
# ANO
# ============================================================

if "ano" in df_original.columns:

    anos = sorted(
        df_original[
            "ano"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    ano = st.sidebar.selectbox(
        "Ano",
        options=[
            "Todos"
        ]
        +
        anos
    )

    ano_filtro = (
        None
        if ano == "Todos"
        else ano
    )

else:

    ano_filtro = None


# ============================================================
# UF
# ============================================================

if "sigla_uf" in df_original.columns:

    ufs = sorted(
        df_original[
            "sigla_uf"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    uf = st.sidebar.selectbox(
        "UF",
        options=[
            "Todas"
        ]
        +
        ufs
    )

    uf_filtro = (
        None
        if uf == "Todas"
        else uf
    )

else:

    uf_filtro = None


df = aplicar_filtros(
    df_original,
    ano_filtro,
    uf_filtro
)


numericas = colunas_numericas(
    df
)


if numericas:

    variavel = st.sidebar.selectbox(
        "Variável numérica",
        options=numericas,
        format_func=nome_amigavel
    )

else:

    variavel = None


st.sidebar.divider()


metodo_correlacao = (
    st.sidebar.selectbox(
        "Método de correlação",
        options=[
            "pearson",
            "spearman"
        ],
        format_func=lambda x: x.capitalize()
    )
)


limiar_correlacao = (
    st.sidebar.slider(
        "Limiar de correlação",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01
    )
)


# ============================================================
# 14. CABEÇALHO
# ============================================================

st.title(
    "Dashboard Descritivo das Bases Municipais"
)

st.caption(
    "Caracterização estatística, qualidade, cobertura, "
    "distribuição, evolução temporal, análise territorial "
    "e diagnóstico dos dados."
)


st.info(
    "Os resultados possuem caráter descritivo e diagnóstico. "
    "Valores ausentes, valores extremos, assimetrias e "
    "correlações não implicam transformação, imputação "
    "ou exclusão automática de observações ou variáveis."
)


# ============================================================
# 15. CARDS
# ============================================================

total_registros = len(
    df
)

total_variaveis = df.shape[
    1
]

municipios = (
    df[
        "cod_municipio"
    ]
    .nunique(
        dropna=True
    )
    if
    "cod_municipio"
    in df.columns
    else np.nan
)

anos_qtd = (
    df[
        "ano"
    ]
    .nunique(
        dropna=True
    )
    if
    "ano"
    in df.columns
    else np.nan
)

duplicidades = (
    df.duplicated(
        subset=[
            "ano",
            "cod_municipio"
        ]
    ).sum()
    if all(
        coluna in df.columns
        for coluna in [
            "ano",
            "cod_municipio"
        ]
    )
    else np.nan
)

total_celulas = (
    df.shape[
        0
    ]
    *
    df.shape[
        1
    ]
)

pct_ausencia = (
    df.isna()
    .sum()
    .sum()
    /
    total_celulas
    *
    100
    if total_celulas > 0
    else np.nan
)


c1, c2, c3, c4, c5, c6 = (
    st.columns(
        6
    )
)


c1.metric(
    "Registros",
    formatar_inteiro(
        total_registros
    )
)

c2.metric(
    "Variáveis",
    formatar_inteiro(
        total_variaveis
    )
)

c3.metric(
    "Municípios",
    formatar_inteiro(
        municipios
    )
)

c4.metric(
    "Anos",
    formatar_inteiro(
        anos_qtd
    )
)

c5.metric(
    "Duplicidades",
    formatar_inteiro(
        duplicidades
    )
)

c6.metric(
    "Ausência global",
    formatar_percentual(
        pct_ausencia
    )
)


st.write("")


# ============================================================
# 16. ABAS
# ============================================================

abas = st.tabs(
    [
        "Visão geral",
        "Qualidade",
        "Estatísticas",
        "Distribuição",
        "Temporal",
        "Territorial",
        "Outliers",
        "Correlações",
        "IDEB",
        "Completa × Analítica",
        "Proveniência",
        "Pré-processamento",
        "Dados e download"
    ]
)


# ============================================================
# ABA 1 - VISÃO GERAL
# ============================================================

with abas[0]:

    st.subheader(
        "Estrutura da base selecionada"
    )

    info = pd.DataFrame(
        [
            {
                "Tipo de base":
                    tipo_base,

                "Base":
                    nome_base,

                "Arquivo":
                    registro[
                        "arquivo"
                    ],

                "Registros":
                    total_registros,

                "Variáveis":
                    total_variaveis,

                "Municípios":
                    municipios,

                "Anos":
                    anos_qtd,

                "Duplicidades":
                    duplicidades,

                "Ausência global (%)":
                    pct_ausencia
            }
        ]
    )

    exibir_tabela(
        info
     )

    if "ano" in df_original.columns:

        st.subheader(
            "Cobertura temporal"
        )

        cobertura = (
            df_original
            .groupby(
                "ano"
            )
            .agg(
                registros=(
                    "ano",
                    "size"
                ),
                municipios=(
                    "cod_municipio",
                    "nunique"
                )
                if
                "cod_municipio"
                in df_original.columns
                else
                (
                    "ano",
                    "size"
                )
            )
            .reset_index()
        )

        col_a, col_b = st.columns(
            [
                2,
                1
            ]
        )

        with col_a:

            fig = px.line(
                cobertura,
                x="ano",
                y="municipios",
                markers=True,
                color_discrete_sequence=[
                    COR_PRIMARIA
                ]
            )

            fig = aplicar_estilo_grafico(
                fig,
                titulo="Cobertura territorial por ano",
                titulo_x="Ano",
                titulo_y="Quantidade de municípios",
                altura=470,
                legenda=False
            )

            fig = adicionar_rotulos_linha(
                fig,
                casas=0
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col_b:

           exibir_tabela(
                cobertura
            )

    st.subheader(
        "Amostra dos registros"
    )
    exibir_tabela(
        df.head(
            50
        )
    )


# ============================================================
# ABA 2 - QUALIDADE
# ============================================================

with abas[1]:

    perfil = calcular_perfil(
        df
    )

    st.subheader(
        "Perfil das variáveis"
    )

    exibir_tabela(
        perfil
    )

    st.subheader(
        "Variáveis com maior proporção de ausências"
    )

    ausencias = (
        perfil[
            [
                "Variável",
                "Ausentes (%)"
            ]
        ]
        .sort_values(
            "Ausentes (%)",
            ascending=False
        )
        .head(
            30
        )
    )

    ausencias_plot = (
        ausencias.copy()
    )

    ausencias_plot[
        "Variável"
    ] = (
        ausencias_plot[
            "Variável"
        ]
        .apply(
            nome_amigavel
        )
    )

    fig = px.bar(
        ausencias_plot.sort_values(
            "Ausentes (%)"
        ),
        x="Ausentes (%)",
        y="Variável",
        orientation="h",
        color_discrete_sequence=[
            COR_QUATERNARIA
        ]
    )

    fig = aplicar_estilo_grafico(
        fig,
        titulo="Variáveis com maior proporção de valores ausentes",
        titulo_x="Valores ausentes (%)",
        titulo_y="Variável",
        altura=720,
        legenda=False
    )

    fig = adicionar_rotulos_barras(
        fig,
        orientacao="horizontal",
        casas=2,
        percentual=True
    )

    fig.update_layout(
        template="plotly_white",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "ano" in df_original.columns:

        st.subheader(
            "Ausência por variável e ano"
        )

        registros_ausencia = []

        for ano_item, grupo in (
            df_original.groupby(
                "ano"
            )
        ):

            for coluna in df_original.columns:

                registros_ausencia.append({
                    "ano":
                        ano_item,

                    "variavel":
                        coluna,

                    "pct_ausentes":
                        (
                            grupo[
                                coluna
                            ]
                            .isna()
                            .mean()
                            *
                            100
                        )
                })

        tabela_ausencia = pd.DataFrame(
            registros_ausencia
        )

        top_vars = (
            tabela_ausencia
            .groupby(
                "variavel"
            )[
                "pct_ausentes"
            ]
            .max()
            .nlargest(
                25
            )
            .index
        )

        matriz = (
            tabela_ausencia[
                tabela_ausencia[
                    "variavel"
                ]
                .isin(
                    top_vars
                )
            ]
            .pivot(
                index="variavel",
                columns="ano",
                values="pct_ausentes"
            )
        )

        matriz_exibicao = matriz.copy()

        matriz_exibicao.index = [
            nome_amigavel(indice)
            for indice in matriz_exibicao.index
        ]

        fig = px.imshow(
            matriz_exibicao,
            aspect="auto",
            labels={
                "x": "Ano",
                "y": "Variável",
                "color": "Ausentes (%)"
            },
            color_continuous_scale=ESCALA_SEQUENCIAL,
            text_auto=".1f"
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo="Distribuição dos valores ausentes por variável e ano",
            titulo_x="Ano",
            titulo_y="Variável",
            altura=720,
            legenda=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# ============================================================
# ABA 3 - ESTATÍSTICAS
# ============================================================

with abas[2]:

    estatisticas = calcular_estatisticas(
        df
    )

    st.subheader(
        "Estatísticas descritivas das variáveis numéricas"
    )

    exibir_tabela(
        estatisticas
    )

    if (
        variavel is not None
        and
        "ano" in df_original.columns
    ):

        st.subheader(
            f"Estatísticas de {nome_amigavel(variavel)} por ano"
        )

        resumo_ano = (
            df_original
            .groupby(
                "ano"
            )[
                variavel
            ]
            .agg(
                N="count",
                Média="mean",
                Mediana="median",
                Desvio_Padrão="std",
                Mínimo="min",
                Máximo="max"
            )
            .reset_index()
        )

        exibir_tabela(
            resumo_ano

        )


# ============================================================
# ABA 4 - DISTRIBUIÇÃO
# ============================================================

with abas[3]:

    if variavel is None:

        st.warning(
            "A base não possui variável numérica disponível."
        )

    else:

        st.subheader(
            f"Distribuição de {nome_amigavel(variavel)}"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            nome_variavel = nome_amigavel(
                variavel
            )

            fig = px.histogram(
                df,
                x=variavel,
                nbins=50,
                marginal="rug",
                color_discrete_sequence=[
                    COR_PRIMARIA
                ]
            )

            fig = aplicar_estilo_grafico(
                fig,
                titulo=(
                    f"Distribuição de {nome_variavel}"
                ),
                titulo_x=
                    nome_variavel,
                titulo_y=
                    "Frequência",
                altura=500,
                legenda=False
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            if (
                "ano" in df.columns
                and
                df[
                    "ano"
                ].nunique()
                >
                1
            ):

                temp = df[
                    [
                        "ano",
                        variavel
                    ]
                ].dropna()

                temp[
                    "ano"
                ] = temp[
                    "ano"
                ].astype(str)

                fig = px.box(
                    temp,
                    x="ano",
                    y=variavel,
                    points=False
                )

            else:

                fig = px.box(
                    df,
                    y=variavel,
                    points=False
                )

            fig.update_layout(
                template="plotly_white"
            )

            fig.update_traces(
                marker_color=
                    COR_TERCIARIA,

                line_color=
                    COR_TERCIARIA,

                fillcolor=
                    "rgba(128,100,145,0.25)"
            )

            fig = aplicar_estilo_grafico(
                fig,
                titulo=(
                    f"Distribuição de {nome_amigavel(variavel)}"
                ),
                titulo_x=(
                    "Ano"
                    if
                    "ano" in df.columns
                    and
                    df["ano"].nunique() > 1
                    else None
                ),
                titulo_y=
                    nome_amigavel(
                        variavel
                    ),
                altura=500,
                legenda=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# ABA 5 - TEMPORAL
# ============================================================

with abas[4]:

    if (
        variavel is None
        or
        "ano" not in df_original.columns
    ):

        st.warning(
            "Análise temporal indisponível."
        )

    else:

        df_temporal = aplicar_filtros(
            df_original,
            ano=None,
            uf=uf_filtro
        )

        temporal = (
            df_temporal
            .groupby(
                "ano"
            )[
                variavel
            ]
            .agg(
                N="count",
                Média="mean",
                Mediana="median",
                Desvio_Padrão="std",
                Mínimo="min",
                Máximo="max"
            )
            .reset_index()
        )

        st.subheader(
            f"Evolução temporal — {nome_amigavel(variavel)}"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=temporal[
                    "ano"
                ],
                y=temporal[
                    "Média"
                ],
                mode="lines+markers",
                name="Média"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=temporal[
                    "ano"
                ],
                y=temporal[
                    "Mediana"
                ],
                mode="lines+markers",
                name="Mediana"
            )
        )

        fig.update_traces(
            line=dict(
                width=2.5
            ),
            marker=dict(
                size=8
            )
        )

        # Cores distintas, porém coerentes.
        fig.data[0].line.color = (
            COR_PRIMARIA
        )

        fig.data[0].marker.color = (
            COR_PRIMARIA
        )

        if len(
            fig.data
        ) > 1:

            fig.data[1].line.color = (
                COR_SECUNDARIA
            )

            fig.data[1].marker.color = (
                COR_SECUNDARIA
            )


        fig = aplicar_estilo_grafico(
            fig,
            titulo=(
                f"Evolução temporal de "
                f"{nome_amigavel(variavel)}"
            ),
            titulo_x=
                "Ano",
            titulo_y=
                nome_amigavel(
                    variavel
                ),
            altura=520,
            legenda=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        exibir_tabela(
            temporal
        )


# ============================================================
# ABA 6 - TERRITORIAL
# ============================================================

with abas[5]:

    if (
        variavel is None
        or
        "sigla_uf" not in df_original.columns
    ):

        st.warning(
            "Análise territorial indisponível."
        )

    else:

        df_territorial = aplicar_filtros(
            df_original,
            ano=ano_filtro,
            uf=None
        )

        territorial = (
            df_territorial
            .groupby(
                "sigla_uf"
            )[
                variavel
            ]
            .agg(
                N="count",
                Média="mean",
                Mediana="median",
                Desvio_Padrão="std",
                Mínimo="min",
                Máximo="max"
            )
            .reset_index()
            .sort_values(
                "Mediana",
                ascending=False
            )
        )

        st.subheader(
            f"Distribuição territorial — {nome_amigavel(variavel)}"
        )

        fig = px.bar(
            territorial.sort_values(
                "Mediana"
            ),
            x="Mediana",
            y="sigla_uf",
            orientation="h",
            color_discrete_sequence=[
                COR_PRIMARIA
            ]
        )

        casas = casas_decimais_variavel(
            df_territorial,
            variavel
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo=(
                f"Mediana de "
                f"{nome_amigavel(variavel)} "
                f"por UF"
            ),
            titulo_x=
                nome_amigavel(
                    variavel
                ),
            titulo_y=
                "UF",
            altura=720,
            legenda=False
        )

        fig = adicionar_rotulos_barras(
            fig,
            orientacao="horizontal",
            casas=casas
        )

        fig.update_layout(
            template="plotly_white",
            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        exibir_tabela(
            territorial
        )


# ============================================================
# ABA 7 - OUTLIERS
# ============================================================

with abas[6]:

    if variavel is None:

        st.warning(
            "Selecione uma variável numérica."
        )

    else:

        outliers = identificar_outliers(
            df,
            variavel
        )

        st.subheader(
            f"Valores extremos pelo critério do IQR — {nome_amigavel(variavel)}"
        )

        col1, col2 = st.columns(
            2
        )

        col1.metric(
            "Registros sinalizados",
            formatar_inteiro(
                len(
                    outliers
                )
            )
        )

        col2.metric(
            "% dos registros",
            formatar_percentual(
                (
                    len(
                        outliers
                    )
                    /
                    len(
                        df
                    )
                    *
                    100
                )
                if len(
                    df
                ) > 0
                else np.nan
            )
        )

        if outliers.empty:

            st.success(
                "Nenhum registro foi classificado como "
                "valor extremo pelo critério selecionado."
            )

        else:

            if "ano" in outliers.columns:

                contagem_outliers = (
                    outliers
                    .groupby(
                        "ano"
                    )
                    .size()
                    .reset_index(
                        name="Quantidade"
                    )
                )

                fig = px.bar(
                    contagem_outliers,
                    x="ano",
                    y="Quantidade",
                    color_discrete_sequence=[
                        COR_QUATERNARIA
                    ]
                )

                fig = aplicar_estilo_grafico(
                    fig,
                    titulo=(
                        f"Valores extremos de "
                        f"{nome_amigavel(variavel)} "
                        f"por ano"
                    ),
                    titulo_x=
                        "Ano",
                    titulo_y=
                        "Quantidade de registros",
                    altura=480,
                    legenda=False
                )

                fig = adicionar_rotulos_barras(
                    fig,
                    orientacao="vertical",
                    casas=0
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            exibir_tabela(
                outliers
            )

        st.caption(
            "A classificação pelo IQR constitui um "
            "diagnóstico descritivo e não implica "
            "exclusão automática das observações."
        )


# ============================================================
# ABA 8 - CORRELAÇÕES
# ============================================================

with abas[7]:

    if (
        variavel is None
        or
        len(
            numericas
        )
        <
        2
    ):

        st.warning(
            "Não há variáveis suficientes para esta análise."
        )

    else:

        st.subheader(
            f"Correlações com {nome_amigavel(variavel)}"
        )

        matriz = (
            df[
                numericas
            ]
            .corr(
                method=metodo_correlacao,
                min_periods=30
            )
        )

        corr_variavel = (
            matriz[
                variavel
            ]
            .drop(
                variavel
            )
            .dropna()
            .reset_index()
        )

        corr_variavel.columns = [
            "Variável",
            "Correlação"
        ]

        corr_variavel[
            "Correlação absoluta"
        ] = corr_variavel[
            "Correlação"
        ].abs()

        corr_variavel = (
            corr_variavel
            .sort_values(
                "Correlação absoluta",
                ascending=False
            )
        )

        top_corr = corr_variavel.head(
            20
        )

        top_corr_plot = (
            top_corr.copy()
        )

        top_corr_plot[
            "Variável"
        ] = (
            top_corr_plot[
                "Variável"
            ]
            .apply(
                nome_amigavel
            )
        )

        fig = px.bar(
            top_corr_plot.sort_values(
                "Correlação"
            ),
            x="Correlação",
            y="Variável",
            orientation="h",
            color_discrete_sequence=[
                COR_TERCIARIA
            ]
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo=(
                f"Variáveis mais correlacionadas com "
                f"{nome_amigavel(variavel)}"
            ),
            titulo_x=
                "Correlação",
            titulo_y=
                "Variável",
            altura=620,
            legenda=False
        )

        fig = adicionar_rotulos_barras(
            fig,
            orientacao="horizontal",
            casas=2
        )

        fig.update_layout(
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        exibir_tabela(
            corr_variavel
        )

        st.subheader(
            f"Pares com |correlação| ≥ {limiar_correlacao:.2f}"
        )

        pares = []

        for i in range(
            len(
                numericas
            )
        ):

            for j in range(
                i + 1,
                len(
                    numericas
                )
            ):

                valor = matriz.iloc[
                    i,
                    j
                ]

                if (
                    pd.notna(
                        valor
                    )
                    and
                    abs(
                        valor
                    )
                    >=
                    limiar_correlacao
                ):

                    pares.append({
                        "Variável 1":
                            numericas[i],

                        "Variável 2":
                            numericas[j],

                        "Correlação":
                            valor,

                        "|Correlação|":
                            abs(
                                valor
                            )
                    })

        pares = pd.DataFrame(
            pares
        )

        if pares.empty:

            st.info(
                "Nenhum par atingiu o limiar selecionado."
            )

        else:

            pares = pares.sort_values(
                "|Correlação|",
                ascending=False
            )

            exibir_tabela(
                pares
            )

        variaveis_heatmap = [
            variavel
        ] + (
            top_corr[
                "Variável"
            ]
            .head(
                10
            )
            .tolist()
        )

        matriz_top = (
            df[
                variaveis_heatmap
            ]
            .corr(
                method=metodo_correlacao,
                min_periods=30
            )
        )

        matriz_top_exibicao = matriz_top.copy()

        nomes_corr = [
            nome_amigavel(coluna)
            for coluna in matriz_top_exibicao.columns
        ]

        matriz_top_exibicao.columns = nomes_corr
        matriz_top_exibicao.index = nomes_corr

        fig = px.imshow(
            matriz_top_exibicao,
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=ESCALA_DIVERGENTE
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo="Matriz de correlação das variáveis selecionadas",
            titulo_x="",
            titulo_y="",
            altura=700,
            legenda=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================
# ABA 9 - IDEB
# ============================================================

with abas[8]:

    coluna_ideb = identificar_ideb(
        df_original
    )

    if coluna_ideb is None:

        st.warning(
            "A base selecionada não possui variável IDEB."
        )

    else:

        st.subheader(
            "Disponibilidade do IDEB"
        )

        if "ano" in df_original.columns:

            disponibilidade = (
                df_original
                .groupby(
                    "ano"
                )[
                    coluna_ideb
                ]
                .agg(
                    Total="size",
                    Disponível="count"
                )
                .reset_index()
            )

            disponibilidade[
                "Ausente"
            ] = (
                disponibilidade[
                    "Total"
                ]
                -
                disponibilidade[
                    "Disponível"
                ]
            )

            disponibilidade[
                "Disponível (%)"
            ] = (
                disponibilidade[
                    "Disponível"
                ]
                /
                disponibilidade[
                    "Total"
                ]
                *
                100
            )

            col1, col2 = st.columns(
                2
            )

            with col1:

                fig = px.line(
                    disponibilidade,
                    x="ano",
                    y="Disponível (%)",
                    markers=True,
                    color_discrete_sequence=[
                        COR_PRIMARIA
                    ]
                )

                fig = aplicar_estilo_grafico(
                    fig,
                    titulo="Disponibilidade do IDEB por ano",
                    titulo_x="Ano",
                    titulo_y="IDEB disponível (%)",
                    altura=480,
                    legenda=False
                )

                fig = adicionar_rotulos_linha(
                    fig,
                    casas=2,
                    percentual=True
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                exibir_tabela(
                    disponibilidade
                )

        componentes = (
            identificar_componentes_ideb(
                df_original
            )
        )

        st.subheader(
            "Distribuição do IDEB"
        )

        fig = px.histogram(
            df_original,
            x=coluna_ideb,
            nbins=40,
            color_discrete_sequence=[
                COR_PRIMARIA
            ]
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo="Distribuição do IDEB",
            titulo_x="IDEB",
            titulo_y="Frequência",
            altura=500,
            legenda=False
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if componentes:

            st.subheader(
                "Relação entre IDEB e seus componentes"
            )

            colunas_ideb = [
                coluna_ideb
            ] + componentes

            matriz_ideb = (
                df_original[
                    colunas_ideb
                ]
                .corr(
                    min_periods=30
                )
            )
            matriz_ideb_exibicao = (
                matriz_ideb.copy()
            )

            nomes_ideb = [
                nome_amigavel(
                    coluna
                )
                for coluna
                in matriz_ideb_exibicao.columns
            ]

            matriz_ideb_exibicao.columns = (
                nomes_ideb
            )

            matriz_ideb_exibicao.index = (
                nomes_ideb
            )

            fig = px.imshow(
                matriz_ideb_exibicao,
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=
                    ESCALA_DIVERGENTE
            )

            fig = aplicar_estilo_grafico(
                fig,
                titulo=(
                    "Correlação entre o IDEB e seus componentes"
                ),
                titulo_x="",
                titulo_y="",
                altura=620,
                legenda=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Os componentes diretos do IDEB não estão "
                "presentes nesta versão da base."
            )


# ============================================================
# ABA 10 - COMPLETA X ANALÍTICA
# ============================================================

with abas[9]:

    etapas_validas = [
        "Anos Iniciais",
        "Anos Finais",
        "Ensino Médio"
    ]

    if nome_base not in etapas_validas:

        st.info(
            "Esta comparação está disponível para "
            "Anos Iniciais, Anos Finais e Ensino Médio."
        )

    else:

        registro_completa = catalogo[
            (
                catalogo[
                    "tipo_base"
                ]
                ==
                "Painéis integrados completos"
            )
            &
            (
                catalogo[
                    "base"
                ]
                ==
                nome_base
            )
        ]

        registro_analitica = catalogo[
            (
                catalogo[
                    "tipo_base"
                ]
                ==
                "Bases analíticas"
            )
            &
            (
                catalogo[
                    "base"
                ]
                ==
                nome_base
            )
        ]

        if (
            registro_completa.empty
            or
            registro_analitica.empty
        ):

            st.warning(
                "Não foi possível localizar as duas versões da base."
            )

        else:

            completa = carregar_base(
                registro_completa.iloc[
                    0
                ][
                    "arquivo"
                ]
            )

            analitica = carregar_base(
                registro_analitica.iloc[
                    0
                ][
                    "arquivo"
                ]
            )

            if (
                "ano" in completa.columns
                and
                "ano" in analitica.columns
            ):

                qtd_completa = (
                    completa
                    .groupby(
                        "ano"
                    )
                    .size()
                    .reset_index(
                        name="Painel completo"
                    )
                )

                qtd_analitica = (
                    analitica
                    .groupby(
                        "ano"
                    )
                    .size()
                    .reset_index(
                        name="Base analítica"
                    )
                )

                retencao = qtd_completa.merge(
                    qtd_analitica,
                    on="ano",
                    how="left"
                )

                retencao[
                    "Base analítica"
                ] = (
                    retencao[
                        "Base analítica"
                    ]
                    .fillna(
                        0
                    )
                )

                retencao[
                    "Retenção (%)"
                ] = (
                    retencao[
                        "Base analítica"
                    ]
                    /
                    retencao[
                        "Painel completo"
                    ]
                    *
                    100
                )

                st.subheader(
                    "Retenção da amostra analítica"
                )

                fig = px.bar(
                    retencao,
                    x="ano",
                    y="Retenção (%)",
                    color_discrete_sequence=[
                        COR_PRIMARIA
                    ]
                )

                fig = aplicar_estilo_grafico(
                    fig,
                    titulo=(
                        "Registros preservados na base analítica por ano"
                    ),
                    titulo_x=
                        "Ano",
                    titulo_y=
                        "Retenção (%)",
                    altura=480,
                    legenda=False
                )

                fig = adicionar_rotulos_barras(
                    fig,
                    orientacao="vertical",
                    casas=2,
                    percentual=True
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                exibir_tabela(
                    retencao
                )


# ============================================================
# ABA 11 - PROVENIÊNCIA
# ============================================================

with abas[10]:

    colunas_origem = [
        coluna
        for coluna in df.columns
        if coluna.startswith(
            "origem_"
        )
    ]

    if not colunas_origem:

        st.info(
            "A base selecionada não possui coluna explícita "
            "de proveniência."
        )

    else:

        for coluna in colunas_origem:

            st.subheader(
                nome_amigavel(coluna)
            )

            if "ano" in df.columns:

                proveniencia = (
                    df
                    .groupby(
                        [
                            "ano",
                            coluna
                        ],
                        dropna=False
                    )
                    .size()
                    .reset_index(
                        name="Registros"
                    )
                )

                # ------------------------------------------------
                # Versão apenas para exibição
                # Mantém os valores originais na base.
                # ------------------------------------------------

                proveniencia_exibicao = (
                    proveniencia.copy()
                )

                proveniencia_exibicao[
                    coluna
                ] = (
                    proveniencia_exibicao[
                        coluna
                    ]
                    .apply(
                        lambda valor:
                            nome_amigavel(valor)
                            if pd.notna(valor)
                            else valor
                    )
                )

                # ------------------------------------------------
                # Gráfico
                # ------------------------------------------------

                fig = px.bar(
                    proveniencia_exibicao,
                    x="ano",
                    y="Registros",
                    color=coluna,
                    barmode="stack",
                    color_discrete_sequence=
                        PALETA_CORES,
                    labels={
                        "ano":
                            "Ano",

                        "Registros":
                            "Quantidade de registros",

                        coluna:
                            nome_amigavel(coluna)
                    }
                )

                fig = aplicar_estilo_grafico(
                    fig,
                    titulo="Proveniência dos registros por ano",
                    titulo_x="Ano",
                    titulo_y="Quantidade de registros",
                    altura=500,
                    legenda=True
                )

                fig.update_traces(
                    texttemplate="%{y:,.0f}",
                    textposition="inside"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ------------------------------------------------
                # Tabela
                # ------------------------------------------------

                exibir_tabela(
                    proveniencia_exibicao
                )

# ============================================================
# ABA 12 - PRÉ-PROCESSAMENTO
# ============================================================

with abas[11]:

    st.subheader(
        "Diagnóstico de possíveis necessidades de pré-processamento"
    )

    st.warning(
        "Os alertas desta seção são diagnósticos. "
        "Nenhuma transformação, imputação, normalização, "
        "winsorização ou exclusão é executada automaticamente."
    )

    diagnostico = (
        diagnostico_preprocessamento(
            df
        )
    )

    if diagnostico.empty:

        st.info(
            "Não foi possível produzir o diagnóstico."
        )

    else:

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Prioridade alta",
            formatar_inteiro(
                (
                    diagnostico["Prioridade"]
                    ==
                    "Alta"
                ).sum()
            )
        )

        c2.metric(
            "Prioridade média",
            formatar_inteiro(
                (
                    diagnostico["Prioridade"]
                    ==
                    "Média"
                ).sum()
            )
        )

        c3.metric(
            "Prioridade baixa",
            formatar_inteiro(
                (
                    diagnostico["Prioridade"]
                    ==
                    "Baixa"
                ).sum()
            )
        )
        contagem = (
            diagnostico[
                "Prioridade"
            ]
            .value_counts()
            .rename_axis(
                "Prioridade"
            )
            .reset_index(
                name="Quantidade"
            )
        )

        ordem_prioridade = [
            "Alta",
            "Média",
            "Baixa"
        ]

        cores_prioridade = {
            "Alta":
                COR_QUATERNARIA,

            "Média":
                COR_SECUNDARIA,

            "Baixa":
                COR_PRIMARIA
        }

        fig = px.bar(
            contagem,
            x="Prioridade",
            y="Quantidade",
            color="Prioridade",
            category_orders={
                "Prioridade":
                    ordem_prioridade
            },
            color_discrete_map=
                cores_prioridade
        )

        fig = aplicar_estilo_grafico(
            fig,
            titulo=(
                "Variáveis segundo a prioridade diagnóstica"
            ),
            titulo_x=
                "Prioridade",
            titulo_y=
                "Quantidade de variáveis",
            altura=480,
            legenda=False
        )

        fig = adicionar_rotulos_barras(
            fig,
            orientacao="vertical",
            casas=0
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        exibir_tabela(
            diagnostico
        )


# ============================================================
# ABA 13 - DADOS E DOWNLOAD
# ============================================================

with abas[12]:

    st.subheader(
        "Dados filtrados"
    )

    st.write(
        f"**{formatar_inteiro(len(df))} registros "
        f"× {formatar_inteiro(df.shape[1])} variáveis**"
    )

    # ========================================================
    # PRÉVIA FORMATADA DOS DADOS
    # Mantém os nomes técnicos na base original e nos downloads.
    # ========================================================

    LIMITE_VISUALIZACAO = 500

    df_previa = df.head(
        LIMITE_VISUALIZACAO
    )

    exibir_tabela(
        df_previa,
        altura=500
    )

    if len(df) > LIMITE_VISUALIZACAO:

        st.caption(
            f"São exibidos os primeiros "
            f"{formatar_inteiro(LIMITE_VISUALIZACAO)} registros. "
            f"Os arquivos para download contêm os "
            f"{formatar_inteiro(len(df))} registros filtrados."
        )

    # ========================================================
    # DOWNLOADS
    # Os arquivos mantêm os nomes técnicos originais.
    # ========================================================

    col1, col2 = st.columns(
        2
    )

    with col1:

        csv = (
            df.to_csv(
                index=False
            )
            .encode(
                "utf-8-sig"
            )
        )

        st.download_button(
            label="Baixar dados filtrados em CSV",
            data=csv,
            file_name="dados_filtrados.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        parquet = dataframe_para_parquet(
            df
        )

        st.download_button(
            label="Baixar dados filtrados em Parquet",
            data=parquet,
            file_name="dados_filtrados.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )

    # ========================================================
    # CATÁLOGO DA BASE
    # ========================================================

    st.subheader(
        "Catálogo da base"
    )

    variaveis_catalogadas = (
        catalogo_variaveis[
            (
                catalogo_variaveis[
                    "tipo_base"
                ]
                ==
                tipo_base
            )
            &
            (
                catalogo_variaveis[
                    "base"
                ]
                ==
                nome_base
            )
        ]
    )

    exibir_tabela(
        variaveis_catalogadas
    )
# ============================================================
# 17. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Dashboard produzido a partir dos produtos do pipeline "
    "de tratamento, harmonização, integração, validação e "
    "análise de qualidade da base longitudinal municipal."
) 
