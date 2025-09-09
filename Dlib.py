import dlib
import numpy as np
import os

# モデルファイルをロード
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 必要なモデルファイルが存在するか確認
if not os.path.exists(predictor_path) or not os.path.exists(face_rec_model_path):
    print("エラー: モデルファイルが見つかりません。")
    print("以下のファイルをダウンロードし、スクリプトと同じフォルダに配置してください:")
    print(f"- {predictor_path}")
    print(f"- {face_rec_model_path}")
    exit()

# 顔検出器、顔のランドマーク予測器、顔識別モデルを初期化
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def get_face_descriptor(image_path):
    """
    指定された画像パスから顔の特徴量ベクトルを取得する関数
    """
    try:
        img = dlib.load_rgb_image(image_path)
    except RuntimeError:
        return None, "画像ファイルが見つからないか、読み込めません。"

    dets = detector(img, 1)

    if len(dets) == 0:
        return None, "顔が検出されませんでした。"

    shape = sp(img, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    # dlibのdescriptorをnumpy配列に変換して返す
    return np.array(face_descriptor), None

def are_same_person(descriptor1, descriptor2, threshold=0.3):
    """
    2つのNumPy配列間の距離を計算し、同一人物か判定する関数
    """
    # ユークリッド距離を計算
    distance = np.linalg.norm(descriptor1 - descriptor2)
    print(f"特徴量間の距離: {distance:.4f}")

    return distance < threshold