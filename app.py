import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Gemini Chat App", page_icon="🤖")

st.title("🤖 Gemini Chat Application")
st.write("StreamlitとGemini APIを使用したチャットアプリケーション (2025年8月最新モデル対応)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("メッセージを入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            response = st.session_state.model.generate_content(prompt)
            full_response = response.text
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

with st.sidebar:
    st.header("設定")
    
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
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    1. メッセージを入力欄に入力
    2. Enterキーを押して送信
    3. Geminiからの応答を待つ
    4. 必要に応じてモデルを変更
    
    ### 利用可能モデル
    - **Gemini 2.5 Flash**: 最新・高速・バランス型
    - **Gemini 2.5 Pro**: 最高性能・複雑タスク対応
    - **Gemini 2.5 Flash Lite**: 低コスト・低遅延
    - **Gemini 2.0 Flash**: 前世代の高速モデル
    - **Gemini 1.5 Flash**: レガシーモデル
    - **Gemini 1.5 Flash-8B**: 軽量モデル
    """)