#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("hand_image.jpeg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
# 1. BGR로 변환 (OpenCV 출력용)
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
output_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# 2. 사이즈 줄이기 (예: 가로 640, 세로 480 또는 비율로 조절)
# 원하는 크기를 직접 지정하거나, 0.5배처럼 비율로 줄일 수 있습니다.
width = 640
height = 480
resized_image = cv2.resize(output_image, (width, height), interpolation=cv2.INTER_AREA)

# 3. 출력
cv2_imshow(resized_image)

"""##그라디오를 활용하여 손모양 분석하기"""

def analyze_hand(input_image):
    if input_image is None:
        return None

    # [수정] 이미지를 좌우로 반전시킵니다 (거울 모드 해결)
    # 1은 좌우 반전, 0은 상하 반전입니다.
    flipped_image = cv2.flip(input_image, 1)

    # 1. 반전된 사진을 AI가 이해할 수 있는 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_image)

    # 2. 분석 수행
    result = detector.detect(mp_image)

    # 3. 반전된 사진 위에 그림 그려서 결과 내보내기
    return draw_landmarks_on_image(flipped_image, result)

# UI 실행 (inputs 부분을 최신 규칙에 맞게 수정)
demo = gr.Interface(
    fn=analyze_hand,
    inputs=gr.Image(sources=["webcam"]), # "webcam" 대신 이 형식을 사용해야 합니다.
    outputs="image"
)

demo.launch(share=True)
