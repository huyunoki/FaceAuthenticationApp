import cv2
import dlib
import os
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

# Dlibのモデルファイルをロード (ファイルパスを適宜修正)
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(predictor_path) or not os.path.exists(face_rec_model_path):
    print("エラー: モデルファイルが見つかりません。")
    pass

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
    return np.array(face_descriptor), None

def are_same_person(descriptor1, descriptor2, threshold=0.3):
    """
    2つのNumPy配列間の距離を計算し、同一人物か判定する関数
    """
    distance = np.linalg.norm(descriptor1 - descriptor2)
    print(f"特徴量間の距離: {distance:.4f}")
    return distance < threshold

def load_registered_faces():
    """
    face_dataフォルダ内のすべての登録者の名前と特徴量を読み込む
    """
    registered_faces = {}
    base_dir = "face_data"
    if not os.path.exists(base_dir):
        print("エラー: 'face_data' フォルダが見つかりません。")
        return registered_faces
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir) and not person_name.startswith('.'):
            image_path = os.path.join(person_dir, "1.jpg")
            if os.path.exists(image_path):
                descriptor, error_msg = get_face_descriptor(image_path)
                if descriptor is not None:
                    registered_faces[person_name] = descriptor
                    print(f"『{person_name}』さんの登録画像を読み込みました。")
                else:
                    print(f"警告: 『{person_name}』さんの画像処理に失敗しました - {error_msg}")
            else:
                print(f"警告: 『{person_name}』さんの登録画像 '1.jpg' が見つかりませんでした。")
    return registered_faces

# 日本語フォントをロード (フォントパスを環境に合わせて修正)
font_path = r"C:\Windows\Fonts\meiryo.ttc"
if not os.path.exists(font_path):
    print("警告: 日本語フォントが見つかりません。")
    pass
font = ImageFont.truetype(font_path, 20)

# OpenCVの画像に日本語テキストを描画する関数
def draw_japanese_text(img, text, position, font, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 顔認証のメイン処理
def main():
    registered_faces = load_registered_faces()
    if not registered_faces:
        print("エラー: 認証対象の登録者が見つかりませんでした。")
        return None

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not capture.isOpened():
        print("カメラに接続できませんでした。")
        return None
    
    window_name = "Real-time Face Authentication"
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) # この行を追加
    
    start_time = time.time()
    while time.time() - start_time < 10:  # 10秒後にタイムアウト
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        dets = detector(frame, 1)
        
        result_text = "Finding a match..."
        text_color = (255, 255, 255)
        
        if len(dets) > 0:
            face_rect = dets[0]
            
            detected_face_img = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
            
            if detected_face_img.size > 0:
                temp_img_path = "temp_face.jpg"
                cv2.imwrite(temp_img_path, detected_face_img)
                detected_descriptor, error_msg = get_face_descriptor(temp_img_path)
                os.remove(temp_img_path)
                
                if detected_descriptor is not None:
                    for name, registered_descriptor in registered_faces.items():
                        if are_same_person(registered_descriptor, detected_descriptor):
                            result_text = f"認証成功: {name} さん"
                            text_color = (0, 255, 0)
                            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), text_color, 2)
                            frame = draw_japanese_text(frame, result_text, (50, 50), font, text_color)
                            cv2.imshow("Real-time Face Authentication", frame)
                            cv2.waitKey(2000)
                            capture.release()
                            cv2.destroyAllWindows()
                            return name
                            
                    result_text = "一致する人物が見つかりませんでした"
                    text_color = (0, 0, 255)
            
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), text_color, 2)
            frame = draw_japanese_text(frame, result_text, (50, 50), font, text_color)
            cv2.imshow("Real-time Face Authentication", frame)

        else:
            result_text = "顔を検出できませんでした"
            text_color = (255, 255, 255)
            frame = draw_japanese_text(frame, result_text, (50, 50), font, text_color)
            cv2.imshow("Real-time Face Authentication", frame)
        
        if cv2.waitKey(1) == 27:
            break
            
    capture.release()
    cv2.destroyAllWindows()
    return None