import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
import asyncio
import io
import base64
import time
from edge_tts import Communicate


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

LANGUAGE_MAP = {
    "עברית": "he",
    "אנגלית": "en",
    "ערבית": "ar",
    "צרפתית": "fr",
    "יפנית": "ja",
    "גרמנית": "de"
}

SIGNS_TRANSLATIONS = {
    "יד פתוחה": {
        "he": "יד פתוחה",
        "en": "open palm",
        "ar": "كف مفتوح",
        "fr": "paume ouverte",
        "ja": "開いた手のひら",
        "de": "offene Handfläche"
    }
}

VOICE_MAP = {
    "he": "he-IL-AvriNeural",
    "en": "en-US-JennyNeural",
    "ar": "ar-SA-ZariyahNeural",
    "fr": "fr-FR-DeniseNeural",
    "ja": "ja-JP-NanamiNeural",
    "de": "de-DE-KatjaNeural"
}


class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(SIGNS_TRANSLATIONS))
        )

    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)


model = SignLanguageModel()
model.eval()


class StateManager:
    def __init__(self):
        self.current_language = "he"
        self.last_prediction = None
        self.consecutive_frames = 0
        self.last_audio_time = 0
        self.audio_delay = 5
        self.detection_buffer = []
        self.buffer_size = 5

    def update_detection_buffer(self, is_open):
        self.detection_buffer.append(is_open)
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)

    def is_hand_open_stable(self):
        if len(self.detection_buffer) < self.buffer_size:
            return False
        return sum(self.detection_buffer) >= self.buffer_size * 0.8


state = StateManager()


async def text_to_speech(text, language):
    voice = VOICE_MAP.get(language, "en-US-JennyNeural")
    communicate = Communicate(text, voice=voice)
    output = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            output.write(chunk["data"])
    output.seek(0)
    return output


def get_finger_angles(hand_landmarks):

    finger_joints = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]

    def calculate_angle(p1, p2, p3):
        vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    angles = []
    for finger in finger_joints:
        for i in range(1, len(finger) - 2):
            angle = calculate_angle(
                hand_landmarks.landmark[finger[i]],
                hand_landmarks.landmark[finger[i + 1]],
                hand_landmarks.landmark[finger[i + 2]]
            )
            angles.append(angle)

    return angles


def get_finger_distances(hand_landmarks):
    fingertips = [4, 8, 12, 16, 20]
    distances = []

    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            p1 = hand_landmarks.landmark[fingertips[i]]
            p2 = hand_landmarks.landmark[fingertips[j]]
            distance = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
            distances.append(distance)

    return distances


def is_open_hand(hand_landmarks):
    angles = get_finger_angles(hand_landmarks)
    distances = get_finger_distances(hand_landmarks)

    finger_straightness = all(165 <= angle <= 195 for angle in angles)

    min_distance = min(distances)
    fingers_spread = min_distance > 0.07

    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    palm_orientation = abs(wrist.z - middle_mcp.z) < 0.1

    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    thumb_distance = np.sqrt((thumb_tip.x - thumb_base.x) ** 2 +
                             (thumb_tip.y - thumb_base.y) ** 2 +
                             (thumb_tip.z - thumb_base.z) ** 2)
    thumb_open = thumb_distance > 0.08

    # Update detection buffer
    is_open = all([finger_straightness, fingers_spread, palm_orientation, thumb_open])
    state.update_detection_buffer(is_open)

    return state.is_hand_open_stable()


def predict_sign(frame, selected_language):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    detected_signs = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_open_hand(hand_landmarks):
                sign_translation = SIGNS_TRANSLATIONS["יד פתוחה"][selected_language]
                detected_signs.append(sign_translation)
                continue

            points = []
            for landmark in hand_landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])

            points = torch.FloatTensor(points).reshape(1, -1)

            with torch.no_grad():
                prediction = model(points)
                predicted_class = torch.argmax(prediction[0]).item()
                confidence = prediction[0][predicted_class].item()

                if confidence > 0.9:  # Increased confidence threshold
                    sign_key = list(SIGNS_TRANSLATIONS.keys())[predicted_class]
                    sign_translation = SIGNS_TRANSLATIONS[sign_key][selected_language]
                    detected_signs.append(sign_translation)

    if detected_signs:
        return detected_signs[0], frame
    return None, frame


def language_change(new_language_name):
    state.current_language = LANGUAGE_MAP[new_language_name]
    state.last_prediction = None
    state.consecutive_frames = 0
    state.detection_buffer.clear()
    return new_language_name


def webcam_feed():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sign, processed_frame = predict_sign(frame, state.current_language)

        if sign:
            if sign == state.last_prediction:
                state.consecutive_frames += 1
                if state.consecutive_frames >= 10:
                    current_time = time.time()
                    if current_time - state.last_audio_time >= state.audio_delay:
                        audio_html = asyncio.run(play_audio_async(sign, state.current_language))
                        state.last_audio_time = current_time
                    yield processed_frame, sign, audio_html
            else:
                state.consecutive_frames = 1
                state.last_prediction = sign
                audio_html = asyncio.run(play_audio_async(sign, state.current_language))
                state.last_audio_time = time.time()
                yield processed_frame, sign, audio_html
        else:
            yield processed_frame, "", ""

    cap.release()


async def play_audio_async(sign, language):
    audio_data = await text_to_speech(sign, language)
    audio_base64 = base64.b64encode(audio_data.getvalue()).decode("utf-8")
    return f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'



with gr.Blocks() as iface:
    gr.Markdown("#תרגום שפת הסימנים לקול")


    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=list(LANGUAGE_MAP.keys()),
            value="",
            label="בחר שפת פלט",
            info="בחר את השפה בה תושמע התוצאה"
        )

    with gr.Row():
        camera_feed = gr.Image(label="תצוגת מצלמה")
        detected_sign = gr.Text(label="סימן שזוהה")
        audio_output = gr.HTML(label="פלט קולי")

    language_dropdown.change(
        fn=language_change,
        inputs=[language_dropdown],
        outputs=[language_dropdown]
    )

    iface.load(
        fn=webcam_feed,
        outputs=[camera_feed, detected_sign, audio_output]
    )

if __name__ == "__main__":
    iface.launch(share=False)