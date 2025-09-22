# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authentication import main as run_face_authentication_main
from datetime import datetime
from urllib.parse import urlencode

# FastAPIアプリのインスタンスを作成
app = FastAPI()

# 静的ファイル（CSS, JS）を配信するディレクトリを設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# テンプレートディレクトリを設定
templates = Jinja2Templates(directory="templates")

# ルートURL ("/") にアクセスされたときに、HTMLページを表示する
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, status: str = None, message: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "status": status, "message": message})

# 「出勤」ボタンが押されたときのリクエストを処理
@app.post("/attendance")
async def handle_attendance():
    print("\n出勤ボタンが押されました。顔認証を開始します。")
    authenticated_user = run_face_authentication_main()
    
    if authenticated_user:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ 【成功】 {authenticated_user} さんが出勤しました。時刻: {current_time}")
        message = f"{authenticated_user}さん、出勤しました。"
        return RedirectResponse(url=f"/?status=success&message={urlencode({'m': message})}", status_code=303)
    else:
        print("❌ 【失敗】 認証に失敗しました。")
        message = "認証に失敗しました。"
        return RedirectResponse(url=f"/?status=failure&message={urlencode({'m': message})}", status_code=303)

# 「退勤」ボタンが押されたときのリクエストを処理
@app.post("/leaving")
async def handle_leaving():
    print("\n退勤ボタンが押されました。顔認証を開始します。")
    authenticated_user = run_face_authentication_main()

    if authenticated_user:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ 【成功】 {authenticated_user} さんが退勤しました。時刻: {current_time}")
        message = f"{authenticated_user}さん、退勤しました。"
        return RedirectResponse(url=f"/?status=success&message={urlencode({'m': message})}", status_code=303)
    else:
        print("❌ 【失敗】 認証に失敗しました。")
        message = "認証に失敗しました。"
        return RedirectResponse(url=f"/?status=failure&message={urlencode({'m': message})}", status_code=303)

# 「顔面新規登録」ボタンが押されたときのリクエストを処理
@app.post("/register")
async def handle_register():
    print("顔面新規登録ボタンが押されました。")
    return {"status": "success", "message": "顔面新規登録を開始します。"}