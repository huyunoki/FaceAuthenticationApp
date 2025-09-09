import cv2
import dlib
import os
import numpy as np
from Dlib import get_face_descriptor, are_same_person
from PIL import ImageFont, ImageDraw, Image

# 認証する人物の情報を事前に読み込む
def load_registered_faces():
    """
    face_dataフォルダ内のすべての登録者の名前と特徴量を読み込む
    """
    registered_faces = {}
    base_dir = "face_data"
    
    if not os.path.exists(base_dir):
        print("エラー: 'face_data' フォルダが見つかりません。")
        return registered_faces

    # 'face_data' フォルダ内のすべてのサブディレクトリ（名前）をリストアップ
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        
        # ディレクトリかつ隠しフォルダでないことを確認
        if os.path.isdir(person_dir) and not person_name.startswith('.'):
            image_path = os.path.join(person_dir, "1.jpg")
            
            if os.path.exists(image_path):
                print(f"『{person_name}』さんの登録画像を読み込み中...")
                descriptor, error_msg = get_face_descriptor(image_path)
                
                if descriptor is not None:
                    registered_faces[person_name] = descriptor
                else:
                    print(f"警告: 『{person_name}』さんの画像処理に失敗しました - {error_msg}")
            else:
                print(f"警告: 『{person_name}』さんの登録画像 '1.jpg' が見つかりませんでした。")
    
    return registered_faces

# 日本語フォントをロード
font_path = r"C:\Windows\Fonts\meiryo.ttc"
if not os.path.exists(font_path):
    print("警告: 日本語フォントが見つかりません。")
    print(f"フォントパス: {font_path} を環境に合わせて修正してください。")
    # フォントが見つからない場合、処理を続行しない
    exit()

font = ImageFont.truetype(font_path, 20)

# OpenCVの画像に日本語テキストを描画する関数
def draw_japanese_text(img, text, position, font, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    # 登録された顔の特徴量をすべて事前に読み込む
    registered_faces = load_registered_faces()

    if not registered_faces:
        print("エラー: 認証対象の登録者が見つかりませんでした。")
        return

    # カメラからビデオキャプチャを開始
    capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not capture.isOpened():
        print("カメラに接続できませんでした。")
        return
    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # dlibの顔検出器をロード
    detector = dlib.get_frontal_face_detector()
    
    print("認証を開始します。カメラに顔を映してください。")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        frame = cv2.flip(frame, 1)

        # リアルタイムで顔を検出
        dets = detector(frame, 1)
        
        result_text = "Finding a match..."
        text_color = (255, 255, 255) # 白
        match_found = False

        if len(dets) > 0:
            face_rect = dets[0]
            
            # 検出された顔の画像を一時的に保存して特徴量を取得
            detected_face_img = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
            if detected_face_img.size > 0:
                temp_img_path = "temp_face.jpg"
                cv2.imwrite(temp_img_path, detected_face_img)
                detected_descriptor, error_msg = get_face_descriptor(temp_img_path)
                os.remove(temp_img_path)

                if detected_descriptor is not None:
                    # すべての登録者と検出した顔を比較
                    for name, registered_descriptor in registered_faces.items():
                        if are_same_person(registered_descriptor, detected_descriptor):
                            result_text = f"認証成功: {name} さんです！"
                            text_color = (0, 255, 0) # 緑
                            match_found = True
                            break # 一致する人が見つかったらループを抜ける
            
            # 認証結果の表示
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), text_color, 2)
        
        if not match_found:
            if len(dets) > 0:
                result_text = "一致する人物が見つかりませんでした"
                text_color = (0, 0, 255) # 赤
            else:
                result_text = "顔を検出できませんでした"
                text_color = (255, 255, 255) # 白

        # Pillowを使用して日本語テキストを描画
        frame = draw_japanese_text(frame, result_text, (50, 50), font, text_color)

        cv2.imshow("Real-time Face Authentication", frame)
        
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()