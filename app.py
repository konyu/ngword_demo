import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

st.set_page_config(page_title="Gemini Chat App", page_icon="🤖")

def check_password():
    """パスワード認証を行う関数"""

    def password_entered():
        """パスワードが入力されたときの処理"""
        auth_user = os.getenv("AUTH_USERNAME")
        auth_pass = os.getenv("AUTH_PASSWORD")

        if (st.session_state["username"] == auth_user and
            st.session_state["password"] == auth_pass):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔐 ログイン")
        st.write("このアプリケーションにアクセスするにはログインが必要です")

        with st.form("login_form"):
            st.text_input("ユーザー名", key="username")
            st.text_input("パスワード", type="password", key="password")
            st.form_submit_button("ログイン", on_click=password_entered)

        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 ユーザー名またはパスワードが正しくありません")

        st.markdown("---")
        st.caption("管理者から認証情報を取得してください")
        return False

    return True

if not check_password():
    st.stop()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🤖 Gemini Chat Application")
st.write("StreamlitとGemini APIを使用したチャットアプリケーション (2025年8月最新モデル対応)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash")

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "images" in message and message["images"]:
            cols = st.columns(min(len(message["images"]), 3))
            for idx, img_data in enumerate(message["images"]):
                with cols[idx % 3]:
                    st.image(img_data, width=200)
        st.markdown(message["content"])

with st.container():
    col1, col2 = st.columns([1, 6])

    with col1:
        uploaded_file = st.file_uploader(
            "画像を添付",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            key="image_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if uploaded_file not in st.session_state.uploaded_images:
                st.session_state.uploaded_images.append(uploaded_file)
                st.success("画像を追加しました")
                st.rerun()

    with col2:
        if st.session_state.uploaded_images:
            st.write("添付画像:")
            cols = st.columns(min(len(st.session_state.uploaded_images), 5))
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                with cols[idx % 5]:
                    st.image(img_file, width=100)
                    if st.button("削除", key=f"remove_{idx}"):
                        st.session_state.uploaded_images.pop(idx)
                        st.rerun()

if prompt := st.chat_input("メッセージを入力してください"):
    images_to_send = []
    image_data_for_history = []

    if st.session_state.uploaded_images:
        for img_file in st.session_state.uploaded_images:
            img_bytes = img_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            images_to_send.append(img)
            image_data_for_history.append(img_bytes)
            img_file.seek(0)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "images": image_data_for_history if image_data_for_history else None
    })

    with st.chat_message("user"):
        if image_data_for_history:
            cols = st.columns(min(len(image_data_for_history), 3))
            for idx, img_data in enumerate(image_data_for_history):
                with cols[idx % 3]:
                    st.image(img_data, width=200)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            if images_to_send:
                content = images_to_send + [prompt]
                response = st.session_state.model.generate_content(content)
            else:
                response = st.session_state.model.generate_content(prompt)

            full_response = response.text

            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            st.session_state.uploaded_images = []

        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

with st.sidebar:
    st.header("設定")

    if st.button("ログアウト", type="secondary"):
        st.session_state["password_correct"] = False
        st.rerun()

    st.markdown("---")

    model_name = st.selectbox(
        "モデル選択",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b"
        ],
        index=0
    )

    if st.button("モデルを変更"):
        st.session_state.model = genai.GenerativeModel(model_name)
        st.success(f"モデルを {model_name} に変更しました")

    if st.button("会話履歴をクリア"):
        st.session_state.messages = []
        st.session_state.uploaded_images = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. 画像を添付（オプション）
    2. メッセージを入力
    3. Enterキーで送信
    4. Geminiからの応答を待つ

    ### 対応画像形式
    PNG, JPG, JPEG, GIF, WebP

    ### 利用可能モデル
    - **Gemini 2.5 Flash**: 最新・高速・バランス型
    - **Gemini 2.5 Pro**: 最高性能・複雑タスク対応
    - **Gemini 2.5 Flash Lite**: 低コスト・低遅延
    - **Gemini 2.0 Flash**: 前世代の高速モデル
    - **Gemini 1.5 Flash**: レガシーモデル
    - **Gemini 1.5 Flash-8B**: 軽量モデル
    """)

    st.markdown("---")
    st.caption("💡 画像を添付してAIに質問できます")