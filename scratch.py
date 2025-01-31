import streamlit as st
import asyncio
import io
import pygame
from edge_tts import Communicate
from deep_translator import GoogleTranslator


# פונקציות עזר
async def text_to_speech_to_memory(text, voice):
    communicate = Communicate(text, voice=voice)
    output = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            output.write(chunk["data"])
    output.seek(0)
    return output


def translate_text(text, dest_language):
    try:
        if dest_language == "he":
            return text
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        return text


# הגדרת הקולות
voices = {
    "עברית": "he-IL-AvriNeural",
    "אנגלית": "en-US-JennyNeural",
    "ספרדית": "es-ES-AlvaroNeural",
    "צרפתית": "fr-FR-DeniseNeural",
    "גרמנית": "de-DE-ConradNeural",
    "יפנית": "ja-JP-NanamiNeural"
}

language_codes = {
    "עברית": "he",
    "אנגלית": "en",
    "ספרדית": "es",
    "צרפתית": "fr",
    "גרמנית": "de",
    "יפנית": "ja"
}

# יצירת ממשק המשתמש ב-Streamlit
st.title("תרגום טקסט לקול")

# בחירת שפה
selected_language = st.selectbox("בחר שפה לתרגום", list(voices.keys()))

# טקסט להמרה
text_input = st.text_area("הכנס טקסט:")

# כפתור המרה והשמעה
if st.button("המר והשמע"):
    if not text_input:
        st.error("נא להכניס טקסט.")
    else:
        # המרת טקסט ושמיעתו
        try:
            voice = voices[selected_language]
            language_code = language_codes[selected_language]
            translated_text = translate_text(text_input, language_code)

            # הצגת הודעת סטטוס
            st.info("מתרגם טקסט...")
            st.info("מעבד טקסט...")

            # המרה לקול ושמירה לקובץ זמני
            audio_data = asyncio.run(text_to_speech_to_memory(translated_text, voice))

            # השמעה עם pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_data, "mp3")
            pygame.mixer.music.play()

            # השמעה אוטומטית מבלי יצירת קובץ פיזי
            while pygame.mixer.music.get_busy():
                continue

            st.success("ההמרה הושלמה!")
        except Exception as e:
            st.error("שגיאה בעיבוד הטקסט.")