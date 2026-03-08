import streamlit as st
import pandas as pd
from PIL import Image

from modules.alphabet_predict import predict_alphabet
from modules.translator import translate_text
from modules.word_predict import predict_word
from modules.formatter import format_text
from modules.alphabet_animation import show_letter
from modules.word_animation import show_word_video

st.set_page_config(
    page_title="Gestura",
    layout="centered",
    page_icon="✋"
)
st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
}

h1 {
    text-align:center;
}

.stButton>button {
    background: linear-gradient(90deg,#3B82F6,#2563EB);
    color:white;
    border-radius:10px;
    height:40px;
    width:200px;
}

.stTextInput>div>div>input {
    background-color:#1F2937;
    color:white;
}

.stSelectbox>div>div {
    background-color:#1F2937;
}

</style>
""", unsafe_allow_html=True)
# ---------------- USER DATA ----------------

def load_users():
    return pd.read_csv("users.csv")

def save_user(name,email,password,role):

    df = load_users()

    df.loc[len(df)] = [name,email,password,role]

    df.to_csv("users.csv",index=False)

def check_login(email, password):

    df = load_users()

    email = email.strip()
    password = password.strip()

    for i in range(len(df)):
        if str(df.loc[i,"email"]).strip() == email and str(df.loc[i,"password"]).strip() == password:
            return df.loc[i,"role"]

    return None
# ---------------- SESSION ----------------

if "login" not in st.session_state:
    st.session_state.login=False

# ---------------- LOGIN PAGE ----------------

if not st.session_state.login:

    st.title("Gestura")

    page = st.radio("Select", ["Login", "Signup"], horizontal=True, label_visibility="collapsed")

    if page=="Signup":

        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password")

        role = st.selectbox("Role",["Signer","Non-Signer"])

        if st.button("Create Account"):
            save_user(name, email, password, role)

            st.success("Account created")

            # Automatically login
            st.session_state.login = True
            st.session_state.role = role

            st.rerun()

    if page=="Login":

        email = st.text_input("Email").strip()
        password = st.text_input("Password", type="password").strip()

        if st.button("Login"):

            role = check_login(email, password)

            if role:

                st.session_state.login = True
                st.session_state.role = role

                st.rerun()

            else:
                st.error("Invalid login")

# ---------------- DASHBOARD ----------------

else:

    st.sidebar.title("Gestura")

    page = st.sidebar.selectbox(
        "Select Mode",
        ["Alphabet Level","Word Level","Logout"]
    )

    if page == "Logout":

        st.session_state.login = False
        st.rerun()

    if page == "Alphabet Level":

        st.title("Alphabet Detection")

    if page == "Word Level":

        st.title("Word Detection")
    if "letters" not in st.session_state:
        st.session_state.letters = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = 0
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    # ---------------- ALPHABET LEVEL ----------------

    if page == "Alphabet Level":
        if "alpha_sentence" not in st.session_state:
            st.session_state.alpha_sentence = ""

        if "last_letter" not in st.session_state:
            st.session_state.last_letter = ""
        if st.session_state.role == "Signer":

            mode = st.radio("Input", ["Upload Image", "Camera"])

            if mode == "Upload Image":

                files = st.file_uploader(
                    "Upload Images",
                    accept_multiple_files=True,
                    key=f"uploader_{st.session_state.uploader_key}"
                )

                if files:

                    cols = st.columns(len(files))
                    images = []

                    for i, file in enumerate(files):
                        img = Image.open(file)
                        cols[i].image(img, width=120)

                        images.append(img)

                    if st.button("Predict Word"):

                        new_files = files[st.session_state.processed_files:]

                        new_letters = []

                        for file in new_files:
                            img = Image.open(file)
                            letter = predict_alphabet(img)
                            new_letters.append(letter)

                        st.session_state.letters.extend(new_letters)

                        st.session_state.processed_files = len(files)

                if st.button("Add Space"):
                    st.session_state.letters.append(" ")

                if st.button("Clear Sentence"):
                    st.session_state.letters = []
                    st.session_state.processed_files = 0

                    # change uploader key to reset uploader
                    st.session_state.uploader_key += 1

                    st.rerun()

                sentence = "".join(st.session_state.letters)

                st.success(f"Sentence: {sentence}")

                # ---------------- FORMATTER ----------------

                format_mode = st.selectbox(
                    "Sentence Style",
                    ["Standard", "Simplified", "Formal"]
                )

                formatted_sentence = format_text(sentence, format_mode)

                st.write("Formatted Sentence:")
                st.info(formatted_sentence)

                # ---------------- TRANSLATION ----------------

                language = st.selectbox(
                    "Translate To",
                    ["English", "Malayalam", "Hindi"]
                )

                translated_sentence = translate_text(formatted_sentence, language)

                st.write("Translated Sentence:")
                st.success(translated_sentence)

            # ---------------- CAMERA MODE ----------------

            if mode == "Camera":

                if "camera_on_alpha" not in st.session_state:
                    st.session_state.camera_on_alpha = False

                col1, col2, col3, col4= st.columns(4)

                if col1.button("📷 Start Camera"):
                    st.session_state.camera_on_alpha = True

                if col2.button("⛔ Stop Camera"):
                    st.session_state.camera_on_alpha = False

                if col3.button("Clear Word"):
                    st.session_state.alpha_sentence = ""
                    st.session_state.last_letter = ""
                if col4.button("Add Space"):
                    st.session_state.alpha_sentence += " "

                frame_placeholder = st.empty()
                letter_placeholder = st.empty()
                word_placeholder = st.empty()

                if st.session_state.camera_on_alpha:

                    import cv2
                    from modules.alphabet_camera import predict_alphabet_frame

                    cap = cv2.VideoCapture(0)

                    while st.session_state.camera_on_alpha:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame = cv2.flip(frame, 1)

                        frame, letter = predict_alphabet_frame(frame)

                        frame_placeholder.image(frame, channels="BGR")

                        letter_placeholder.success(f"Detected Letter: {letter}")

                        # ---------- BUILD WORD ----------
                        if letter not in ["Waiting", "?"]:

                            if letter != st.session_state.last_letter:
                                st.session_state.alpha_sentence += letter
                                st.session_state.last_letter = letter

                        word_placeholder.info(
                            f"Predicted Word: {st.session_state.alpha_sentence}"
                        )

                    cap.release()
        else:

            text = st.text_input("Enter text")

            if st.button("Show Alphabet Signs"):

                for letter in text:
                    show_letter(letter)

 # ---------------- WORD LEVEL ----------------

if page == "Word Level":

    if st.session_state.role == "Signer":

        # create session variable
        if "detected_sentence" not in st.session_state:
            st.session_state.detected_sentence = ""

        if st.button("Start Camera Detection"):
            st.warning("Press 'Q' on your keyboard to quit the camera.")
            sentence = predict_word()

            # store result
            st.session_state.detected_sentence = sentence

        # show result if available
        if st.session_state.detected_sentence != "":

            st.success(
                f"Detected Sentence: {st.session_state.detected_sentence}"
            )

            language = st.selectbox(
                "Translate To",
                ["English", "Malayalam", "Hindi"]
            )

            translated_text = translate_text(
                st.session_state.detected_sentence,
                language
            )

            st.write("Translated Text:")
            st.info(translated_text)

    else:
        text = st.text_input("Enter word")

        if st.button("Show Word Sign"):
            show_word_video(text)

