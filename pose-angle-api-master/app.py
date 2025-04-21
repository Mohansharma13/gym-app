import streamlit as st
import tempfile
import google.generativeai as genai
from analyzer import PoseAnalyzer
import os

# === Gemini Configuration ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PUT_YOUR_KEY_HERE")
if not GEMINI_API_KEY or GEMINI_API_KEY == "PUT_YOUR_KEY_HERE":
    st.error("❌ Missing Gemini API Key.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# === UI ===
st.title("🏋️ Gym Pose Angle Analyzer")
st.markdown("Upload your workout video and get AI-based feedback on your **form**.")

uploaded_file = st.file_uploader("Upload your exercise video", type=["mp4", "mov", "avi"])
exercise_name = st.text_input("What exercise are you performing? (e.g., bicep curls, squats)")

if uploaded_file and st.button("🔍 Analyze"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("⏳ Processing video..."):
        analyzer = PoseAnalyzer()
        angles = analyzer.analyze_video(tmp_path)

    st.success("✅ Analysis Complete!")
    st.json(angles)

    # Gemini prompt
    prompt = (
        f"I'm doing {exercise_name}. "
        f"My elbow angles range from {angles['elbow_min_angle']}° to {angles['elbow_max_angle']}°, "
        f"and my knee angles range from {angles['knee_min_angle']}° to {angles['knee_max_angle']}°. "
        f"Please evaluate my form and suggest any improvements."
    )

    with st.spinner("💬 Getting AI feedback..."):
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

    st.subheader("🧠 Gemini Feedback:")
    st.write(response.text if hasattr(response, "text") else "No feedback generated.")
