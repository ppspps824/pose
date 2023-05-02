import base64
import glob
import random
import time
import urllib

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from st_clickable_images import clickable_images

KEYPOINT_THRESHOLD = 0.2


@st.cache_resource
def dl_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model


# @st.cache_data
def audio_load():
    bgm_path = (
        "https://drive.google.com/uc?id=1PKOHg8HTRvYhirULB87FUO8Jap7QGu9W"  # 入力する音声ファイル
    )
    with urllib.request.urlopen(url=bgm_path) as file1:
        contents1 = file1.read()

    audio_str1 = f"data:audio/ogg;base64,{base64.b64encode(contents1).decode()}"
    st.session_state.bgm_html = f"""
                    <audio loop id="audio1" autoplay=True>
                    <source src="{audio_str1}" type="audio/ogg" autoplay=True>
                    </audio>
                """
    audio_path1 = (
        "https://drive.google.com/uc?id=1ui9nWGK52lfvl2az7iD-jUVOFbj4F4o1"  # 入力する音声ファイル
    )
    with urllib.request.urlopen(url=audio_path1) as file2:
        contents2 = file2.read()

    audio_str2 = f"data:audio/ogg;base64,{base64.b64encode(contents2).decode()}"
    st.session_state.audio_html1 = f"""
                    <audio id="audio1" autoplay=True>
                    <source src="{audio_str2}" type="audio/ogg" autoplay=True>
                    </audio>
                """


def bgm_play():
    time.sleep(0.5)
    st.markdown(st.session_state.bgm_html, unsafe_allow_html=True)


def correct_sound():
    time.sleep(0.5)
    st.markdown(st.session_state.audio_html1, unsafe_allow_html=True)


def change_image():
    st.session_state.image_path = random.choice(st.session_state.image_path_list)


def game():
    # Tensorflow Hubを利用してモデルダウンロード
    with st.spinner("読み込み中・・・"):
        model = dl_model()
        movenet = model.signatures["serving_default"]

    col2, col3 = st.columns(2)
    with col2:
        image_place2 = st.empty()
    with col3:
        image_place3 = st.empty()
    text_place = st.empty()

    # カメラ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        target_image = cv2.imread(st.session_state.image_path)
        keypoints_list2, scores_list2, bbox_list2 = run_inference(
            movenet, np.array(target_image)
        )
        result_image2 = render(target_image, keypoints_list2, scores_list2, bbox_list2)
        image_place2.image(result_image2, channels="BGR")

        ret, frame = cap.read()
        flip_frame = cv2.flip(frame, 1)
        if not ret:
            break

        # 推論実行
        keypoints_list1, scores_list2, bbox_list3 = run_inference(movenet, flip_frame)

        # 画像レンダリング
        result_image1 = render(
            flip_frame, keypoints_list1, scores_list2, bbox_list3, line_on=True
        )

        image_place3.image(result_image1, channels="BGR")

        for value1, value2 in zip(keypoints_list1, keypoints_list2):
            for in_value1, in_value2 in zip(value1, value2):
                neary_score = abs(in_value1[0] - in_value2[0]) + abs(
                    in_value1[1] - in_value2[1]
                )
        # text_place.write(f"## {neary_score}")
        if neary_score < 50:
            correct_sound()
            st.balloons()
            time.sleep(2)
            change_image()

    cap.release()


def main():
    start_image = "./start.png"
    with open(start_image, "rb") as image3:
        encoded = base64.b64encode(image3.read()).decode()
        start_logo = f"data:image/jpeg;base64,{encoded}"
    logo_place = st.empty()
    with logo_place:
        clicked = clickable_images(
            [start_logo],
            titles="",
            div_style={
                "display": "flex",
                "justify-content": "center",
                "flex-wrap": "wrap",
            },
            key="final",
        )
    if clicked == 0:
        logo_place = st.empty()
        game()


def run_inference(model, image):
    # 画像の前処理
    input_image = cv2.resize(image, dsize=(128, 128))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, 0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # 推論実行・結果取得
    outputs = model(input_image)
    keypoints = np.squeeze(outputs["output_0"].numpy())

    image_height, image_width = image.shape[:2]
    keypoints_list, scores_list, bbox_list = [], [], []

    # 検出した人物ごとにキーポイントのフォーマット処理
    for kp in keypoints:
        keypoints = []
        scores = []
        for index in range(17):
            kp_x = int(image_width * kp[index * 3 + 1])
            kp_y = int(image_height * kp[index * 3 + 0])
            score = kp[index * 3 + 2]
            keypoints.append([kp_x, kp_y])
            scores.append(score)
        bbox_ymin = int(image_height * kp[51])
        bbox_xmin = int(image_width * kp[52])
        bbox_ymax = int(image_height * kp[53])
        bbox_xmax = int(image_width * kp[54])
        bbox_score = kp[55]

        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def render(image, keypoints_list, scores_list, bbox_list, line_on=False):
    render = image.copy()
    # img_white = np.ones((512, 512), np.uint8) * 250
    # render = img_white.copy()
    for i, (keypoints, scores, bbox) in enumerate(
        zip(keypoints_list, scores_list, bbox_list)
    ):
        if bbox[4] < 0.2:
            continue

        # cv2.rectangle(render, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # 0:nose, 1:left eye, 2:right eye, 3:left ear, 4:right ear, 5:left shoulder, 6:right shoulder, 7:left elbow, 8:right elbow, 9:left wrist, 10:right wrist,
        # 11:left hip, 12:right hip, 13:left knee, 14:right knee, 15:left ankle, 16:right ankle
        # 接続するキーポイントの組
        kp_links = [
            # (0, 1),
            # (0, 2),
            # (1, 3),
            # (2, 4),
            (0, 5),
            (0, 6),
            # (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (11, 12),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 12),
            (12, 14),
            (14, 16),
        ]
        if line_on:
            for kp_idx_1, kp_idx_2 in kp_links:
                kp_1 = keypoints[kp_idx_1]
                kp_2 = keypoints[kp_idx_2]
                score_1 = scores[kp_idx_1]
                score_2 = scores[kp_idx_2]
                if score_1 > KEYPOINT_THRESHOLD and score_2 > KEYPOINT_THRESHOLD:
                    cv2.line(render, tuple(kp_1), tuple(kp_2), (255, 0, 0), 30)

            if scores[0] > KEYPOINT_THRESHOLD:
                cv2.circle(
                    render,
                    tuple([keypoints[0][0], keypoints[0][1] - 40]),
                    40,
                    (255, 0, 0),
                    -1,
                )

    return render


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
    )

    st.markdown(
        """
            <style>
                .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
            </style>
            """,
        unsafe_allow_html=True,
    )

    if "image_path" not in st.session_state:
        st.session_state.image_path_list = glob.glob("image/*.png")
        st.session_state.image_path = st.session_state.image_path_list[0]

    audio_load()
    bgm_play()

    main()
