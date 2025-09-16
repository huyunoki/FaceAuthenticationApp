from fastapi import FastAPI, File, UploadFile, HTTPException
import dlib
import numpy as np
import cv2
import os
from io import BytesIO

# モデルファイルをロード (既存コードから流用)
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
if not os.path.exists(predictor_path) or not os.path.exists(face_rec_model_path):
    raise FileNotFoundError("モデルファイルが見つかりません。")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 登録済みの顔データを事前に読み込み
registered_faces = {}
base_dir = "face_data"
if os.path.exists(base_dir):
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):
            image_path = os.path.join(person_dir, "1.jpg")
            if os.path.exists(image_path):
                img = dlib.load_rgb_image(image_path)
                dets = detector(img, 1)
                if len(dets) > 0:
                    shape = sp(img, dets[0])
                    descriptor = facerec.compute_face_descriptor(img, shape)
                    registered_faces[person_name] = np.array(descriptor)
                print(f"『{person_name}』さんの登録完了。")

app = FastAPI()

def get_face_descriptor(img):
    """画像から顔の特徴量を取得するヘルパー関数"""
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = sp(img, dets[0])
    return np.array(facerec.compute_face_descriptor(img, shape))

@app.post("/authenticate/")
async def authenticate_face(file: UploadFile = File(...)):
    """
    アップロードされた顔画像を認証するAPIエンドポイント
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_descriptor = get_face_descriptor(rgb_img)
    if detected_descriptor is None:
        raise HTTPException(status_code=400, detail="画像から顔が検出できませんでした。")

    # 登録者と照合
    for name, registered_descriptor in registered_faces.items():
        distance = np.linalg.norm(registered_descriptor - detected_descriptor)
        if distance < 0.35: # 閾値は適宜調整してください
            return {"status": "success", "user": name, "distance": distance}

    return {"status": "failure", "message": "一致する人物が見つかりませんでした。"}