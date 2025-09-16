from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authentication import main as run_face_authentication_main

# FastAPIアプリのインスタンスを作成
app = FastAPI()

# 静的ファイル（CSS, JS）を配信するディレクトリを設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# テンプレートディレクトリを設定
templates = Jinja2Templates(directory="templates")

# ルートURL ("/") にアクセスされたときに、HTMLページを表示する
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 「出勤」ボタンが押されたときのリクエストを処理
@app.post("/attendance")
async def handle_attendance():
    print("出勤ボタンが押されました。顔認証を開始します。")
    # authentication.pyのmain関数を直接呼び出す
    run_face_authentication_main()
    return {"status": "success", "message": "出勤処理が完了しました。"}

# 「退勤」ボタンが押されたときのリクエストを処理
@app.post("/leaving")
async def handle_leaving():
    print("退勤ボタンが押されました。顔認証を開始します。")
    # authentication.pyのmain関数を直接呼び出す
    run_face_authentication_main()
    return {"status": "success", "message": "退勤処理が完了しました。"}

# 「顔面新規登録」ボタンが押されたときのリクエストを処理
@app.post("/register")
async def handle_register():
    # 顔面新規登録のロジックをここに実装
    print("顔面新規登録ボタンが押されました。")
    return {"status": "success", "message": "顔面新規登録を開始します。"}