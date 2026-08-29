# ============================================================
# DASHBOARD DESCRITIVO DAS BASES MUNICIPAIS
# ETAPA 1 - VALIDAÇÃO DA APLICAÇÃO
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title=(
        "Dashboard Descritivo das Bases Municipais"
    ),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CAMINHOS
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

CAMINHO_VARIAVEIS = (
    PASTA_DADOS
    /
    "catalogo_variaveis.csv"
)


# ============================================================
# VALIDAR ARQUIVOS ESSENCIAIS
# ============================================================

if not PASTA_DADOS.exists():

    st.error(
        "A pasta 'dados' não foi encontrada."
    )

    st.stop()


if not CAMINHO_CATALOGO.exists():

    st.error(
        "O arquivo catalogo_dados.csv "
        "não foi encontrado."
    )

    st.stop()


if not CAMINHO_VARIAVEIS.exists():

    st.error(
        "O arquivo catalogo_variaveis.csv "
        "não foi encontrado."
    )

    st.stop()


# ============================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================

@st.cache_data
def carregar_catalogo():

    return pd.read_csv(
        CAMINHO_CATALOGO
    )


@st.cache_data
def carregar_catalogo_variaveis():

    return pd.read_csv(
        CAMINHO_VARIAVEIS
    )


@st.cache_data(
    show_spinner=False
)
def carregar_base(
    nome_arquivo
):

    caminho = (
        PASTA_DADOS
        /
        nome_arquivo
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    return pd.read_parquet(
        caminho,
        engine="pyarrow"
    )


# ============================================================
# CARREGAR CATÁLOGOS
# ============================================================

catalogo = carregar_catalogo()

catalogo_variaveis = (
    carregar_catalogo_variaveis()
)


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "Dashboard Descritivo das Bases Municipais"
)

st.caption(
    "Base longitudinal municipal brasileira "
    "aplicada à análise educacional"
)

st.info(
    "Esta versão inicial verifica a integridade "
    "da conexão entre a aplicação e os produtos "
    "gerados pelo pipeline de preparação dos dados."
)


# ============================================================
# FILTROS LATERAIS
# ============================================================

st.sidebar.header(
    "Seleção da base"
)


tipos_disponiveis = (
    catalogo[
        "tipo_base"
    ]
    .dropna()
    .unique()
    .tolist()
)


tipo_base = st.sidebar.selectbox(
    "Tipo de base",
    options=tipos_disponiveis
)


catalogo_tipo = (
    catalogo[
        catalogo[
            "tipo_base"
        ]
        ==
        tipo_base
    ]
)


bases_disponiveis = (
    catalogo_tipo[
        "base"
    ]
    .dropna()
    .unique()
    .tolist()
)


nome_base = st.sidebar.selectbox(
    "Base",
    options=bases_disponiveis
)


registro_base = (
    catalogo_tipo[
        catalogo_tipo[
            "base"
        ]
        ==
        nome_base
    ]
    .iloc[
        0
    ]
)


arquivo_base = (
    registro_base[
        "arquivo"
    ]
)


# ============================================================
# CARREGAR BASE SELECIONADA
# ============================================================

try:

    df = carregar_base(
        arquivo_base
    )

except Exception as erro:

    st.error(
        "Não foi possível carregar a base selecionada."
    )

    st.exception(
        erro
    )

    st.stop()


# ============================================================
# CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "Registros",
        f"{len(df):,}".replace(
            ",",
            "."
        )
    )


with col2:

    st.metric(
        "Variáveis",
        df.shape[
            1
        ]
    )


with col3:

    if "cod_municipio" in df.columns:

        qtd_municipios = (
            df[
                "cod_municipio"
            ]
            .nunique(
                dropna=True
            )
        )

        st.metric(
            "Unidades territoriais",
            f"{qtd_municipios:,}".replace(
                ",",
                "."
            )
        )

    else:

        st.metric(
            "Unidades territoriais",
            "-"
        )


with col4:

    if "ano" in df.columns:

        qtd_anos = (
            df[
                "ano"
            ]
            .nunique(
                dropna=True
            )
        )

        st.metric(
            "Anos",
            qtd_anos
        )

    else:

        st.metric(
            "Anos",
            "-"
        )


# ============================================================
# INFORMAÇÕES DA BASE
# ============================================================

st.subheader(
    "Base selecionada"
)


info_base = pd.DataFrame(
    [
        {
            "Tipo":
                tipo_base,

            "Base":
                nome_base,

            "Arquivo":
                arquivo_base,

            "Linhas esperadas":
                int(
                    registro_base[
                        "qtd_linhas"
                    ]
                ),

            "Linhas carregadas":
                len(
                    df
                ),

            "Colunas esperadas":
                int(
                    registro_base[
                        "qtd_colunas"
                    ]
                ),

            "Colunas carregadas":
                df.shape[
                    1
                ],

            "Tamanho do arquivo (MB)":
                round(
                    float(
                        registro_base[
                            "tamanho_mb"
                        ]
                    ),
                    2
                )
        }
    ]
)


st.dataframe(
    info_base,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VALIDAR DIMENSÕES
# ============================================================

linhas_ok = (
    len(
        df
    )
    ==
    int(
        registro_base[
            "qtd_linhas"
        ]
    )
)


colunas_ok = (
    df.shape[
        1
    ]
    ==
    int(
        registro_base[
            "qtd_colunas"
        ]
    )
)


if (
    linhas_ok
    and
    colunas_ok
):

    st.success(
        "A base foi carregada e as dimensões "
        "correspondem ao catálogo gerado pelo pipeline."
    )

else:

    st.error(
        "As dimensões da base carregada não "
        "correspondem ao catálogo."
    )


# ============================================================
# CATÁLOGO DE VARIÁVEIS
# ============================================================

st.subheader(
    "Variáveis disponíveis"
)


variaveis_base = (
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
    variaveis_base,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AMOSTRA DA BASE
# ============================================================

st.subheader(
    "Amostra dos dados"
)


st.dataframe(
    df.head(
        50
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RODAPÉ METODOLÓGICO
# ============================================================

st.divider()

st.caption(
    "A aplicação utiliza exclusivamente produtos "
    "gerados pelo pipeline documentado de tratamento, "
    "harmonização, integração e validação da base."
)