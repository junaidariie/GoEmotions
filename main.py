import streamlit as st
import plotly.express as px
from emotions_backend import find_emotions

#=================== Page Config ===================
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🙂",
    layout="centered",
    initial_sidebar_state="collapsed"
)

#=================== Custom Styling =================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #00e1ff;
    margin-bottom: -10px;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 30px;
}
.big-emotion {
    display: inline-block;
    padding: 10px 20px;
    margin-top: 15px;
    border-radius: 25px;
    background: linear-gradient(90deg,#00e1ff55,#00e1ff22);
    color: #00e1ff;
    font-weight: 700;
    border: 2px solid #00e1ff88;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

#=================== Header ============================

st.markdown('<div class="title">Emotion Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand the emotional tone behind your text</div>', unsafe_allow_html=True)

#=================== Model Capability Section ============================

with st.expander("Model Capabilities"):
    st.write("This model can detect **28 different emotions**, including:")
    st.write("""
- Joy, Love, Gratitude, Excitement  
- Anger, Disgust, Annoyance, Disapproval  
- Sadness, Grief, Remorse  
- Surprise, Realization  
- Fear, Nervousness  
- Optimism, Pride, Approval  
- Curiosity, Desire, Caring  
- Confusion, Embarrassment, Disappointment  
- Relief  
- Neutral
""")

#=================== Sample Text Buttons ============================

sample_texts = [
    "I love this movie so much!",
    "I hate you, leave me alone.",
    "I'm excited but also nervous for tomorrow.",
    "Thank you so much, I really appreciate your help.",
    "I'm so disappointed in you.",
    "Nothing matters anymore, everything feels empty.",
    "Wow great job, another failure...",
    "The meeting starts at 3 PM."
]

st.write("### Try a sample:")

cols = st.columns(4)

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

for i, txt in enumerate(sample_texts):
    if cols[i % 4].button(txt[:22] + "..."):
        st.session_state["text_input"] = txt

#=================== Text Input ============================

text = st.text_area(
    "Enter text to analyze:",
    value=st.session_state["text_input"],
    placeholder="Type something like: I love this movie!",
    height=130
)

#=================== Prediction Button ============================

if st.button("Analyze Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing emotions..."):
            result = find_emotions(text)

        top_emotion, top_prob = result["top"]

        st.subheader("Top Emotion")
        st.markdown(
            f"<span class='big-emotion'>{top_emotion.upper()} ({top_prob*100:.1f}%)</span>",
            unsafe_allow_html=True
        )

        st.subheader("Emotion Profile")

        emotions = [top_emotion] + [e for e,_ in result["others"]]
        probs = [top_prob] + [p for _,p in result["others"]]

        #=================== Plotly Bar Chart ====================
        fig = px.bar(
            x=emotions,
            y=probs,
            range_y=[0,1],
            color=emotions,
            color_discrete_sequence=px.colors.sequential.Teal,
            title="Emotion Probability Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        #=================== Progress Bars ====================
        st.subheader("Details")

        for emotion, prob in zip(emotions, probs):
            st.write(f"**{emotion.capitalize()}** ({prob*100:.1f}%)")
            st.progress(prob)
