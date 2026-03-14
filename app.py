import streamlit as st
import spacy
from spacy import displacy
import streamlit.components.v1 as components
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Page config
st.set_page_config(page_title="Dependency Parser", page_icon="🧠", layout="wide")

# Custom CSS for smooth animation
st.markdown("""
<style>
.fade-in {
    animation: fadeIn 1.5s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="fade-in title">🧠 NLP Dependency Parser</div>', unsafe_allow_html=True)
st.markdown('<div class="fade-in subtitle">Enter a sentence to analyze its linguistic structure</div>', unsafe_allow_html=True)

st.write("")

# User input
sentence = st.text_input("Enter a sentence:")
st.caption("Example: The quick brown fox jumps over the lazy dog")

if sentence:

    doc = nlp(sentence)

    col1, col2 = st.columns(2)

    # Token + POS Table
    with col1:
        st.subheader("📋 Token Analysis")

        data = []
        for token in doc:
            data.append({
                "Word": token.text,
                "POS": token.pos_,
                "Dependency": token.dep_
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    # Named Entities
    with col2:
        st.subheader("🏷 Named Entities")

        if doc.ents:
            for ent in doc.ents:
                st.write(f"**{ent.text}** → {ent.label_}")
        else:
            st.write("No named entities detected.")

    st.write("")

    # Dependency Graph
    st.subheader("🔗 Dependency Visualization")

    options = {
    "distance": 120,
    "compact": False,
    "color": "#ffffff",
    "bg": "#1e1e1e",
    "font": "Arial",
    "arrow_spacing": 20,
    "arrow_width": 3
    }

    html = displacy.render(doc, style="dep", options=options)
    components.html(html, height=450, scrolling=True)

    # Sentence statistics
    st.subheader("📊 Sentence Statistics")

    col3, col4, col5 = st.columns(3)

    col3.metric("Total Words", len(doc))
    col4.metric("Unique POS Tags", len(set([token.pos_ for token in doc])))
    col5.metric("Named Entities", len(doc.ents))