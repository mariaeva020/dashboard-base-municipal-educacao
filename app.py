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
        options=numericas
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
        ]
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

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
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
                title=(
                    "Quantidade de municípios por ano"
                )
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col_b:

            st.dataframe(
                cobertura,
                use_container_width=True,
                hide_index=True
            )

    st.subheader(
        "Amostra dos registros"
    )

    st.dataframe(
        df.head(
            50
        ),
        use_container_width=True,
        hide_index=True
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

    st.dataframe(
        perfil,
        use_container_width=True,
        hide_index=True
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

    fig = px.bar(
        ausencias.sort_values(
            "Ausentes (%)"
        ),
        x="Ausentes (%)",
        y="Variável",
        orientation="h"
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

        fig = px.imshow(
            matriz,
            aspect="auto",
            labels=dict(
                color="Ausentes (%)"
            ),
            title=(
                "Mapa de ausência por variável e ano"
            )
        )

        fig.update_layout(
            height=700
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

    st.dataframe(
        estatisticas,
        use_container_width=True,
        hide_index=True
    )

    if (
        variavel is not None
        and
        "ano" in df_original.columns
    ):

        st.subheader(
            f"Estatísticas de {variavel} por ano"
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

        st.dataframe(
            resumo_ano,
            use_container_width=True,
            hide_index=True
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
            f"Distribuição de {variavel}"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            fig = px.histogram(
                df,
                x=variavel,
                nbins=50,
                marginal="rug"
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
            f"Evolução temporal — {variavel}"
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

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Ano",
            yaxis_title=variavel
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            temporal,
            use_container_width=True,
            hide_index=True
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
            f"Distribuição territorial — {variavel}"
        )

        fig = px.bar(
            territorial.sort_values(
                "Mediana"
            ),
            x="Mediana",
            y="sigla_uf",
            orientation="h",
            labels={
                "sigla_uf":
                    "UF"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            territorial,
            use_container_width=True,
            hide_index=True
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
            f"Valores extremos pelo critério do IQR — {variavel}"
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
                    y="Quantidade"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.dataframe(
                outliers,
                use_container_width=True,
                hide_index=True
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
            f"Correlações com {variavel}"
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

        fig = px.bar(
            top_corr.sort_values(
                "Correlação"
            ),
            x="Correlação",
            y="Variável",
            orientation="h"
        )

        fig.update_layout(
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            corr_variavel,
            use_container_width=True,
            hide_index=True
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

            st.dataframe(
                pares,
                use_container_width=True,
                hide_index=True
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

        fig = px.imshow(
            matriz_top,
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            title=(
                "Matriz de correlação das variáveis selecionadas"
            )
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
                    markers=True
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                st.dataframe(
                    disponibilidade,
                    use_container_width=True,
                    hide_index=True
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
            nbins=40
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

            fig = px.imshow(
                matriz_ideb,
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto"
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
                    y="Retenção (%)"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    retencao,
                    use_container_width=True,
                    hide_index=True
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
                coluna
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

                fig = px.bar(
                    proveniencia,
                    x="ano",
                    y="Registros",
                    color=coluna,
                    barmode="stack"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    proveniencia,
                    use_container_width=True,
                    hide_index=True
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
            (
                diagnostico[
                    "Prioridade"
                ]
                ==
                "Alta"
            ).sum()
        )

        c2.metric(
            "Prioridade média",
            (
                diagnostico[
                    "Prioridade"
                ]
                ==
                "Média"
            ).sum()
        )

        c3.metric(
            "Prioridade baixa",
            (
                diagnostico[
                    "Prioridade"
                ]
                ==
                "Baixa"
            ).sum()
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

        fig = px.bar(
            contagem,
            x="Prioridade",
            y="Quantidade"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            diagnostico,
            use_container_width=True,
            hide_index=True
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

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=500
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        csv = df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        )

        st.download_button(
            label="Baixar dados filtrados em CSV",
            data=csv,
            file_name=(
                "dados_filtrados.csv"
            ),
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
            file_name=(
                "dados_filtrados.parquet"
            ),
            mime="application/octet-stream",
            use_container_width=True
        )

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

    st.dataframe(
        variaveis_catalogadas,
        use_container_width=True,
        hide_index=True
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
