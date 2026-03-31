import gradio as gr
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- [설정] 시각화 도구 준비 ---
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv2.LINE_AA)
    return annotated_image

# --- [준비] MediaPipe 모델 로드 ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- [핵심] 분석 함수 ---
def analyze_hand(input_image):
    if input_image is None:
        return None

    # 거울 모드 해결 (좌우 반전)
    flipped_image = cv2.flip(input_image, 1)
    
    # MediaPipe 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_image)
    
    # 손 탐지
    result = detector.detect(mp_image)
    
    # 결과 시각화
    return draw_landmarks_on_image(flipped_image, result)

# --- [UI] Gradio 실행 ---
demo = gr.Interface(
    fn=analyze_hand,
    inputs=gr.Image(sources=["webcam"]),
    outputs="image",
    title="🖐️ 실시간 AI 손인식 분석기"
)

if __name__ == "__main__":
    demo.launch()
