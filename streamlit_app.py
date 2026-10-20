import streamlit as st
import requests

st.set_page_config("Marriage Stability Predictor")
st.title("Marriage Stability Predictor")
st.markdown("Answer the questions below to assess whether your relationship is stable")
st.divider()

st.subheader("Questions")

st.markdown("5-point frequency Likert scale: 0 for 'Never', 1 for 'Rarely', 2 for 'Sometimes', 3 for 'Often' and 4 for 'Always'")

col1, col2 = st.columns(2)

with col1:
    travel = st.slider("I enjoy traveling with my wife", 0, 4, 2)
    harmony = st.slider("I think my spouse and I are in harmony", 0, 4, 2)
    dreams = st.slider("Our dreams are similar and harmonious", 0, 4, 2)
    happy = st.slider("We share the same views about being happy in our life", 0, 4, 2)
    marriage = st.slider("My spouse and I have similar ideas about how marriage should be", 0, 4, 2)

with col2:
    anxieties = st.slider("I know my spouse's basic anxieties", 0, 4, 2)
    aggro = st.slider("I feel aggressive when I argue with my spouse", 0, 4, 2)
    hate = st.slider("I hate my spouse's way of opening a subject", 0, 4, 2)
    idk = st.slider("We're just starting a discussion before I know what's going on", 0, 4, 2)
    good = st.slider("Sometimes I think it's good for me to leave home for a while", 0, 4, 2)

payload = {
    "enjoy_travel": travel,
    "harmony": harmony,
    "dreams": dreams,
    "happy": happy,
    "marriage": marriage,
    "anxieties": anxieties,
    "Aggro_argue": aggro,
    "hate_subjects": hate,
    "idk_what_s_going_on": idk,
    "good_to_leave_home": good
}

st.divider()

if st.button("Predict marriage stability"):
    response = requests.post("https://marriage-prediction-api.onrender.com/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction")
        message = result.get("message")

        st.subheader("Prediction result")

        if prediction == 0:
            st.success("Your marriage has a good chance of lasting!")
        else:
            st.warning("Your relationship has some vulnerabilities")

        st.write(message)

    else:
        st.error("Error while calling the API")
