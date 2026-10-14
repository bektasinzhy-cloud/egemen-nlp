import ast
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from nltk.stem import PorterStemmer, SnowballStemmer
from wordcloud import WordCloud

# ------------------------------------------
# 1. САЙТ ПАРАМЕТРЛЕРІ
# ------------------------------------------

st.set_page_config(
    page_title="EgemenNLP | Smart Word Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# 2. ДИЗАЙН
# ------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #fff8fc 0%,
        #ffe9f3 50%,
        #f9efff 100%
    );
    color: #4d3454;
}

.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #fff0f7;
    border-right: 1px solid #f7c8df;
}

h1, h2, h3 {
    color: #a63f78 !important;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(
        110deg,
        #f7a6c9,
        #e8a3dc,
        #c8b0fa
    );
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px #e9a4c255;
}

.hero h1 {
    color: white !important;
    margin: 0;
}

.hero p {
    color: white;
}

.result-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 22px;
    border: 1px solid #f7cde2;
    box-shadow: 0 5px 22px #eaa7c430;
    min-height: 125px;
}

.result-label {
    color: #b66d94;
    font-size: 14px;
}

.result-value {
    color: #a63f78;
    font-size: 25px;
    font-weight: 750;
    overflow-wrap: anywhere;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg, #f38dbb, #cf8ce8
    );
    color: white;
    border: none;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 17px;
    border: 1px solid #f7cde2;
}

[data-testid="stMetricValue"] {
    color: #bd5d96;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #fffafd;
    border: 1px solid #efb9d3;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# 3. ФАЙЛДАРДЫ ОҚУ
# ------------------------------------------

BASE = Path(__file__).resolve().parent

@st.cache_data
def load_csv(filename):
    path = BASE / filename

    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")

    return pd.DataFrame()

news = load_csv("egemen_news_final.csv")
stem_data = load_csv("egemen_stemming.csv")
lemma_data = load_csv("egemen_lemmatization_corrected.csv")
top_words = load_csv("egemen_top50_words_final.csv")

def read_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

# ------------------------------------------
# 4. СТЕММИНГ ЖӘНЕ ЛЕММАТИЗАЦИЯ
# ------------------------------------------

porter = PorterStemmer()
snowball = SnowballStemmer("russian")

lemma_dictionary = {
    "аймақтар": "аймақ",
    "қарағанды": "қарағанды",
    "әбілда": "әбілда",
    "облысында": "облыс",
    "коронавируспен": "коронавирус",
    "жұқтырғандардың": "жұқтыр",
    "келеді": "кел",
    "құтырынды": "құтыр",
    "кітаптарды": "кітап",
    "қалаларда": "қала",
    "оқушылардың": "оқушы",
    "үйлерімізге": "үй"
}

def analyze_word(word):
    word = word.lower().strip()

    return {
        "Бастапқы сөз": word,
        "Porter": porter.stem(word),
        "Snowball": snowball.stem(word),
        "Лемматизация": lemma_dictionary.get(word, word)
    }

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"[^a-zа-яәіңғүұқөһё0-9\s-]", " ", text
    )
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return re.findall(
        r"[a-zа-яәіңғүұқөһё]+(?:-[a-zа-яәіңғүұқөһё]+)*",
        text.lower()
    )

stopwords = set("""
және мен бен пен да де та те
үшін туралы арқылы бойынша
бұл осы сол ол олар біз сіз
бір бірақ немесе әрі
сияқты дейін кейін ғана тек
""".split())

def remove_stopwords(tokens):
    return [
        w for w in tokens
        if w not in stopwords and len(w) > 1
    ]

def result_card(label, value, icon):
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">{icon} {label}</div>
        <div class="result-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# 5. БҮЙІР МӘЗІР
# ------------------------------------------

st.sidebar.title("🧠 EgemenNLP")
st.sidebar.caption("Kazakh NLP Laboratory")

page = st.sidebar.radio(
    "Бөлімдер",
    [
        "🏠 Dashboard",
        "✨ Smart Word Analyzer",
        "🧪 Live Text Lab",
        "⚔️ Morphology Battle",
        "📊 Corpus Explorer",
        "☁️ Word Cloud",
        "📰 Жаңалықтар"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "БӨЖ №1 | NLP жобасы\n\n"
    "Python + Jupyter + Streamlit"
)

# ------------------------------------------
# 6. БАСТЫ БЕТ
# ------------------------------------------

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🧠 EgemenNLP</h1>
        <p>Қазақ тіліндегі жаңалықтарды интеллектуалды өңдеу</p>
        <p>Smart Word Analyzer • Text Lab • Corpus Explorer</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    a.metric("🔗 Табылған сілтемелер", "312 389")
    b.metric("📄 Жиналған мақалалар", "1 000")
    c.metric("📚 Соңғы корпус", len(news))
    d.metric("🔤 Сүзілген токендер", "169 141")

    st.subheader("📊 Стемминг пен лемматизация")

    stats = pd.DataFrame({
        "Әдіс": [
            "Бастапқы", "Porter", "Snowball", "Лемма"
        ],
        "Бірегей сөздер": [
            40010, 39999, 36571, 40004
        ]
    })

    fig = px.bar(
        stats,
        x="Әдіс",
        y="Бірегей сөздер",
        color="Әдіс",
        text="Бірегей сөздер",
        color_discrete_sequence=[
            "#618cf5", "#53cba8", "#ff986c", "#a779ef"
        ]
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Статистика бұрынғы Jupyter нәтижелерінен алынған. "
        "Porter және Snowball қазақ тіліне арнайы жасалмаған."
    )

# ------------------------------------------
# 7. SMART WORD ANALYZER
# ------------------------------------------

elif page == "✨ Smart Word Analyzer":

    st.markdown("""
    <div class="hero">
        <h1>✨ Smart Word Analyzer</h1>
        <p>Бір сөз енгізіңіз — нәтижелер автоматты түрде шығады</p>
    </div>
    """, unsafe_allow_html=True)

    word = st.text_input(
        "Қазақша сөз енгізіңіз",
        value="облысында",
        placeholder="Мысалы: аймақтар"
    )

    examples = [
        "аймақтар", "облысында", "қарағанды",
        "коронавируспен", "құтырынды"
    ]

    selected = st.selectbox(
        "Немесе дайын мысалды таңдаңыз",
        ["Өзім енгіземін"] + examples
    )

    if selected != "Өзім енгіземін":
        word = selected

    word = word.strip()

    if word and re.fullmatch(
        r"[a-zа-яәіңғүұқөһё-]+", word.lower()
    ):

        result = analyze_word(word)

        a, b, c, d = st.columns(4)

        with a:
            result_card(
                "Бастапқы сөз",
                result["Бастапқы сөз"], "📖"
            )

        with b:
            result_card(
                "Porter стеммингі",
                result["Porter"], "⚙️"
            )

        with c:
            result_card(
                "Snowball стеммингі",
                result["Snowball"], "❄️"
            )

        with d:
            result_card(
                "Лемматизация",
                result["Лемматизация"], "💡"
            )

        st.divider()
        st.subheader("🔄 Word Journey")

        st.markdown(
            f"**{result['Бастапқы сөз']}**"
            f" → **{result['Snowball']}**"
            f" → **{result['Лемматизация']}**"
        )

        st.caption(
            "Бұл — әдістердің нәтижелерін визуалды салыстыру. "
            "Snowball нәтижесі лемматизацияның кірісі емес."
        )

        if word.lower() not in lemma_dictionary:
            st.warning(
                "Бұл сөз шағын лемма сөздігінде жоқ. "
                "Лемматизация нәтижесі бастапқы сөзбен бірдей."
            )

        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True,
            hide_index=True
        )

    elif word:
        st.warning("Тек бір сөз енгізіңіз, бос орын қолданбаңыз.")

# ------------------------------------------
# 8. LIVE TEXT LAB
# ------------------------------------------

elif page == "🧪 Live Text Lab":

    st.title("🧪 Live Text Lab")
    st.write("Мәтін енгізіп, барлық өңдеу кезеңін бақылаңыз.")

    text = st.text_area(
        "Мәтін",
        value=(
            "Қарағанды облысында коронавируспен "
            "күрес күшейген сайын жаңалықтар көбейді."
        ),
        height=170
    )

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)

    tabs = st.tabs([
        "🧹 Тазалау",
        "🔤 Токенизация",
        "🚫 Стоп-сөздер",
        "⚙️ Стемминг",
        "💡 Лемматизация"
    ])

    with tabs[0]:
        st.write(cleaned)

    with tabs[1]:
        st.write(tokens)
        st.metric("Токен саны", len(tokens))

    with tabs[2]:
        st.write(filtered)
        st.metric("Қалған токендер", len(filtered))

    with tabs[3]:
        st.dataframe(
            pd.DataFrame({
                "Бастапқы": filtered,
                "Porter": [
                    porter.stem(w) for w in filtered
                ],
                "Snowball": [
                    snowball.stem(w) for w in filtered
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

    with tabs[4]:
        st.dataframe(
            pd.DataFrame({
                "Бастапқы": filtered,
                "Лемма": [
                    lemma_dictionary.get(w, w)
                    for w in filtered
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

# ------------------------------------------
# 9. MORPHOLOGY BATTLE
# ------------------------------------------

elif page == "⚔️ Morphology Battle":

    st.title("⚔️ Morphology Battle")
    st.write("Үш әдістің нәтижелерін салыстырыңыз.")

    if not stem_data.empty and not lemma_data.empty:

        rows = []

        for i in range(min(len(stem_data), len(lemma_data))):
            original = read_list(
                stem_data.iloc[i]["tokens_no_stopwords"]
            )
            p = read_list(stem_data.iloc[i]["porter_stems"])
            s = read_list(stem_data.iloc[i]["snowball_stems"])
            l = read_list(lemma_data.iloc[i]["lemmas"])

            for values in zip(original, p, s, l):
                rows.append(values)

        compare = pd.DataFrame(
            rows,
            columns=[
                "Бастапқы", "Porter",
                "Snowball", "Лемма"
            ]
        )

        st.metric("Салыстырылған токендер", len(compare))

        st.dataframe(
            compare.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Салыстыруды CSV ретінде сақтау",
            data=compare.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="morphology_comparison.csv",
            mime="text/csv"
        )

    else:
        st.warning("Стемминг немесе лемматизация CSV файлы табылмады.")

# ------------------------------------------
# 10. CORPUS EXPLORER
# ------------------------------------------

elif page == "📊 Corpus Explorer":

    st.title("📊 Corpus Explorer")

    st.metric("Мақалалар саны", len(news))

    if not news.empty:

        search = st.text_input(
            "🔎 Мақала іздеу",
            placeholder="Мысалы: Алматы"
        )

        filtered_news = news.copy()

        if search:
            filtered_news = filtered_news[
                filtered_news["title"].fillna("").str.contains(
                    search,
                    case=False,
                    regex=False
                )
            ]

        st.dataframe(
            filtered_news[["title", "url"]].head(100),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("📈 Сөз жиілігі")

    if not top_words.empty:
        n = st.slider("Сөз саны", 5, 50, 20)

        chart_data = top_words.head(n)

        fig = px.bar(
            chart_data,
            x="Сөз",
            y="Жиілік",
            color="Жиілік",
            color_continuous_scale="Purples"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            chart_data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Жиілік CSV файлы табылмады.")

# ------------------------------------------
# 11. WORD CLOUD
# ------------------------------------------

elif page == "☁️ Word Cloud":

    st.title("☁️ Қазақша сөздер бұлты")

    if not top_words.empty:

        frequencies = dict(
            zip(
                top_words["Сөз"].astype(str),
                top_words["Жиілік"]
            )
        )

        # Қазақ әріптерін қолдайтын қаріп
        font_candidates = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf")
        ]

        font_path = next(
            (str(p) for p in font_candidates if p.exists()),
            None
        )

        cloud = WordCloud(
            width=1200,
            height=550,
            background_color="white",
            colormap="cool",
            font_path=font_path,
            collocations=False
        ).generate_from_frequencies(frequencies)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Сөздер жиілігінің файлы табылмады.")

# ------------------------------------------
# 12. ЖАҢАЛЫҚТАР
# ------------------------------------------

elif page == "📰 Жаңалықтар":

    st.title("📰 Egemen.kz жаңалықтар корпусы")

    if not news.empty:

        st.dataframe(
            news[["title", "url"]].head(100),
            use_container_width=True,
            hide_index=True
        )

        titles = news["title"].fillna("").tolist()

        chosen = st.selectbox(
            "Мақаланы таңдаңыз",
            range(len(titles)),
            format_func=lambda i: titles[i]
        )

        article = news.iloc[chosen]

        st.subheader(str(article["title"]))

        if "article_text" in news.columns:
            st.write(str(article["article_text"]))
        else:
            st.write(str(article["text"]))

        st.link_button(
            "🔗 Түпнұсқа мақаланы ашу",
            str(article["url"])
        )

    else:
        st.warning("Жаңалықтар корпусы табылмады.")
