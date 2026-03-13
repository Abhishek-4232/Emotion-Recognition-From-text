
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Page configuration
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>

.main {
background-color: #4A90E2;
}

h1 {
color: #4A90E2;
text-align: center;
}

.stTextArea textarea {
border-radius: 10px;
border: 2px solid #4A90E2;
}

.stButton>button {
background-color: #4A90E2;
color: white;
border-radius: 10px;
height: 45px;
width: 200px;
font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# Load ML model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))


# 3D Emoji image paths
emotions_emoji_dict = {
    "anger": "emoji/anger.png",
    "disgust": "emoji/disgust.png",
    "fear": "emoji/fear.png",
    "happy": "emoji/happy.png",
    "joy": "emoji/joy.png",
    "neutral": "emoji/neutral.png",
    "sad": "emoji/sad.png",
    "sadness": "emoji/sad.png",
    "shame": "emoji/shame.png",
    "surprise": "emoji/surprise.png"
}


# Prediction function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


# Probability function
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


# Main app
def main():

    st.title("🤖 AI Text Emotion Detection")

    st.subheader("Detect emotions in text using Machine Learning")

    # User input form
    with st.form(key='emotion_form'):

        raw_text = st.text_area("Type your sentence here")

        submit_text = st.form_submit_button(label='SUBMIT')


    if submit_text:

        with st.spinner("Analyzing emotion..."):

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        # LEFT COLUMN
        with col1:

            st.success("Original Text")

            st.write(raw_text)

            st.success("Prediction Result")

            emoji_icon = emotions_emoji_dict.get(prediction)

            st.markdown(f"### 🎯 Emotion: **{prediction.upper()}**")

            if emoji_icon:
                st.image(emoji_icon, width=150)

            st.write(f"Confidence Score: {np.max(probability):.2f}")

            # Balloons for happy emotions
            if prediction in ["joy", "happy"]:
                st.balloons()

        # RIGHT COLUMN
        with col2:

            st.success("Emotion Probability")

            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

            proba_df_clean = proba_df.T.reset_index()

            proba_df_clean.columns = ["emotions", "probability"]

            chart = alt.Chart(proba_df_clean).mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            ).encode(
                x='emotions',
                y='probability',
                color=alt.Color('emotions', scale=alt.Scale(scheme='set2'))
            )

            st.altair_chart(chart, use_container_width=True)


if __name__ == '__main__':
    main()