import cv2
import dlib
import os
import time
import numpy as np
from Dlib import get_face_descriptor
from PIL import ImageFont, ImageDraw, Image

# モデルファイルをロード
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 必要なモデルファイルが存在するか確認
if not os.path.exists(predictor_path):
    print("エラー: モデルファイルが見つかりません。")
    print(f"- {predictor_path}")
    exit()

# 保存先のルートディレクトリ
SAVE_ROOT_DIR = "face_data"
# 顔検出器とランドマーク予測器をロード
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# 日本語フォントをロード
font_path = r"C:\Windows\Fonts\meiryo.ttc"
if not os.path.exists(font_path):
    print("警告: 日本語フォントが見つかりません。")
    print(f"フォントパス: {font_path} を環境に合わせて修正してください。")
    exit()

font = ImageFont.truetype(font_path, 20)

# OpenCVの画像に日本語テキストを描画する関数
def draw_japanese_text(img, text, position, font, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def save_detected_face_with_name():
    """
    名前を入力して顔画像を保存する関数
    """
    person_name = input("保存する人物の名前を入力してください: ").strip()
    if not person_name:
        print("エラー: 名前が入力されていません。")
        return

    save_dir = os.path.join(SAVE_ROOT_DIR, person_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"ディレクトリ '{save_dir}' を作成しました。")

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not capture.isOpened():
        print("エラー: カメラに接続できませんでした。")
        return

    print("カメラを起動しました。高品質な顔が検出されると、画像が保存されます。")
    print(f"『{person_name}』さんの顔を映してください。終了するには 'Esc' キーを押してください。")

    # 💡 変更点1: ウィンドウを事前に作成し、名前を付ける 💡
    window_name = "face saver"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # 💡 変更点2: ウィンドウを常に最前面に設定 💡
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    image_count = 1
    face_detected_time = None
    frame_count = 0
    detection_frequency = 5
    dets = []
    
    live_status_text = "待機中..."
    live_status_color = (255, 255, 255) # 白

    while True:
        ret, frame = capture.read()
        if not ret:
            print("エラー: フレームの取得に失敗しました。")
            break

        frame = cv2.flip(frame, 1)
        
        if frame_count % detection_frequency == 0:
            dets = detector(frame, 1)
        frame_count += 1
        
        if len(dets) > 0:
            rect = dets[0]
            # ここに顔の枠を描画する行を追加
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

            shape = sp(frame, rect)
            
            is_good_quality = True
            
            center_x = (rect.left() + rect.right()) / 2
            center_y = (rect.top() + rect.bottom()) / 2
            frame_center_x = frame.shape[1] / 2
            frame_center_y = frame.shape[0] / 2
            
            if abs(center_x - frame_center_x) > 100 or abs(center_y - frame_center_y) > 100:
                is_good_quality = False
                live_status_text = "顔をもっと中央に!"
                live_status_color = (0, 165, 255) # オレンジ
            
            face_size = rect.area()
            min_size = 15000
            max_size = 80000
            if face_size < min_size:
                is_good_quality = False
                live_status_text = "顔をもっと近づけて!"
                live_status_color = (0, 0, 255) # 赤
            elif face_size > max_size:
                is_good_quality = False
                live_status_text = "顔をもっと離して!"
                live_status_color = (0, 0, 255) # 赤
                
            left_eye = shape.part(36)
            right_eye = shape.part(45)
            eye_y_diff = abs(left_eye.y - right_eye.y)
            eye_x_diff = abs(left_eye.x - right_eye.x)
            
            if eye_x_diff > 0 and (eye_y_diff / eye_x_diff) > 0.3:
                is_good_quality = False
                live_status_text = "顔をまっすぐに!"
                live_status_color = (0, 165, 255) # オレンジ
                
            if is_good_quality:
                live_status_text = "OK！画像を保存します"
                live_status_color = (0, 255, 0) # 緑
                
                if face_detected_time is None:
                    face_detected_time = time.time()
                
                if time.time() - face_detected_time > 2.0:
                    expanded_rect = dlib.rectangle(
                        max(0, rect.left() - 50), 
                        max(0, rect.top() - 50), 
                        min(frame.shape[1], rect.right() + 50), 
                        min(frame.shape[0], rect.bottom() + 50)
                    )
                    
                    face_img = frame[expanded_rect.top():expanded_rect.bottom(), expanded_rect.left():expanded_rect.right()]
                    
                    if face_img is not None and face_img.size > 0:
                        file_path = os.path.join(save_dir, f"{image_count}.jpg")
                        success = cv2.imwrite(file_path, face_img)
                        if success:
                            test_descriptor, test_error_msg = get_face_descriptor(file_path)
                            if test_descriptor is not None:
                                print(f"画像を保存しました: {file_path}")
                                print("✅ 自己検証に成功しました！この画像は認証に使用できます。")
                                image_count += 1
                                face_detected_time = None
                                live_status_text = "自己検証OK！"
                                live_status_color = (0, 255, 0)
                            else:
                                os.remove(file_path)
                                print(f"❌ 自己検証に失敗しました: {test_error_msg} -> 再撮影してください。")
                                face_detected_time = None
                                live_status_text = "自己検証NG"
                                live_status_color = (0, 0, 255)
                        else:
                            print(f"エラー: 画像の保存に失敗しました。ファイルパス: {file_path}")
                            face_detected_time = None
                    else:
                        print("警告: 検出された顔の切り抜きが空でした。")
                        face_detected_time = None
            else:
                face_detected_time = None
        else:
            live_status_text = "顔を検出できません"
            live_status_color = (255, 255, 255)
            face_detected_time = None

        frame = draw_japanese_text(frame, live_status_text, (50, 50), font, live_status_color)
        
        if len(dets) > 0:
            rect = dets[0]
            frame = draw_japanese_text(frame, "顔を検出しました", (rect.left(), rect.top() - 25), font, (0, 255, 0))

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_detected_face_with_name()