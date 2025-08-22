import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import json
import re
import chromadb
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

def analyze_image_with_prompt(image, prompt):
    """画像を定型プロンプトで分析してJSONデータを取得"""
    try:
        model = st.session_state.model
        content = [image, prompt]
        response = model.generate_content(content)

        # JSONデータの抽出
        response_text = response.text

        # ```json から ```までの部分を抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # {}で囲まれたJSONを探す
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

        # JSONをパース
        try:
            json_data = json.loads(json_str)
            return json_data, response_text
        except json.JSONDecodeError:
            return None, response_text

    except Exception as e:
        return None, f"エラーが発生しました: {str(e)}"

def check_ngwords_in_query_strings(query_strings, threshold=0.3):
    """query_stringリストをNGワード検索でチェック"""
    if not query_strings:
        return []
    
    vectorizer, collection = load_vectorizer_and_db()
    if vectorizer is None or collection is None:
        return []
    
    all_ngword_results = []
    for query in query_strings:
        if query and query.strip():
            ngword_results = search_ng_words(query.strip(), vectorizer, collection, threshold, 5)
            if ngword_results:
                all_ngword_results.append({
                    'query': query,
                    'ngwords': ngword_results
                })
    
    return all_ngword_results

def display_json_data(json_data):
    """JSONデータをリスト形式で表示（新しい配列形式に対応）"""
    if isinstance(json_data, list):
        st.subheader("📋 分析結果")

        for item in json_data:
            if isinstance(item, dict) and 'id' in item:
                # 新しい形式のJSONデータ
                item_id = item.get('id', 'N/A')
                objects = item.get('object', [])
                texts = item.get('text', [])
                source = item.get('source', 'N/A')
                query_strings = item.get('query_string', [])

                # オブジェクトのラベルを取得してタイトルに使用
                title_parts = []
                if objects:
                    for obj in objects:
                        if isinstance(obj, dict):
                            label = obj.get('label', '')
                            if label:
                                title_parts.append(label)

                # テキストからもタイトル要素を追加
                if texts and len(texts) > 0:
                    title_parts.extend(texts[:2])  # 最初の2つのテキストを追加

                title = ' / '.join(title_parts) if title_parts else f"ブロック {item_id}"

                # NGワード検索を実行
                ngword_results = check_ngwords_in_query_strings(query_strings)
                has_ngwords = len(ngword_results) > 0
                
                # NGワードがある場合は警告アイコンを追加
                warning_icon = "⚠️ " if has_ngwords else ""

                with st.expander(f"🔍 {warning_icon}{title} (ID: {item_id})", expanded=True):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # オブジェクト情報
                        if objects:
                            st.write("**🎯 検出されたオブジェクト:**")
                            for obj in objects:
                                if isinstance(obj, dict):
                                    label = obj.get('label', 'N/A')
                                    category = obj.get('category', 'N/A')

                                    # カテゴリに応じたアイコン
                                    category_icons = {
                                        '人の顔': '👤',
                                        '化粧品容器': '🧴',
                                        'その他': '📦'
                                    }
                                    icon = category_icons.get(category, '📦')

                                    st.write(f"  {icon} **{label}** ({category})")
                        else:
                            st.write("**🎯 検出されたオブジェクト:** なし")

                        # テキスト情報
                        if texts:
                            st.write("**📝 抽出されたテキスト:**")
                            for idx, text in enumerate(texts, 1):
                                st.write(f"  {idx}. {text}")
                        else:
                            st.write("**📝 抽出されたテキスト:** なし")

                    with col2:
                        # ソース情報
                        st.write(f"**📁 ソース:** {source}")

                        # クエリ文字列
                        if query_strings:
                            st.write("**🔍 クエリ文字列:**")
                            for idx, query in enumerate(query_strings, 1):
                                st.code(query, language="text")
                        else:
                            st.write("**🔍 クエリ文字列:** なし")
                    
                    # NGワード検索結果の表示
                    if has_ngwords:
                        st.markdown("---")
                        st.write("**🚫 NGワード検索結果:**")
                        
                        for ngword_result in ngword_results:
                            query = ngword_result['query']
                            ngwords = ngword_result['ngwords']
                            
                            st.write(f"**クエリ:** `{query}`")
                            
                            # リスクレベル別の色分け
                            risk_colors = {
                                'high': '🔴',
                                'mid': '🟡', 
                                'low': '🟢'
                            }
                            
                            for ngword in ngwords[:3]:  # 上位3件のみ表示
                                risk_icon = risk_colors.get(ngword['risk_level'], '⚪')
                                st.warning(f"{risk_icon} **{ngword['ng_word']}** (類似度: {ngword['similarity']:.3f}) → {ngword['replacement']}")
                            
                            if len(ngwords) > 3:
                                st.caption(f"...他 {len(ngwords) - 3} 件のNGワードが検出されました")
                    else:
                        st.markdown("---")
                        st.success("✅ NGワードは検出されませんでした")
            else:
                # 従来の形式（後方互換性のため）
                with st.expander(f"{item.get('title', item.get('name', f'アイテム'))}", expanded=True):
                    for k, v in item.items():
                        if k not in ['title', 'name']:
                            st.write(f"**{k}**: {v}")

    elif isinstance(json_data, dict):
        # 辞書形式の場合（後方互換性）
        for key, value in json_data.items():
            if isinstance(value, list):
                st.subheader(f"📋 {key}")
                for idx, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        with st.expander(f"{idx}. {item.get('title', item.get('name', f'アイテム{idx}'))}", expanded=True):
                            for k, v in item.items():
                                if k not in ['title', 'name']:
                                    st.write(f"**{k}**: {v}")
                    else:
                        st.write(f"{idx}. {item}")
            elif isinstance(value, dict):
                st.subheader(f"📋 {key}")
                for k, v in value.items():
                    st.write(f"**{k}**: {v}")
            else:
                st.write(f"**{key}**: {value}")

if not check_password():
    st.stop()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🤖 Gemini Multi-Tool Application")
st.write("画像分析・ファイル分析・文書処理などの多機能ツール")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash")

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if "file_analysis_results" not in st.session_state:
    st.session_state.file_analysis_results = []

# NGワード検索用のクラスと関数
class SimpleJapaneseVectorizer:
    """シンプルな日本語ベクトル化クラス"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=5000,
            lowercase=False,
            token_pattern=None
        )
        self.fitted = False

    def fit_transform(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return vectors.toarray()

    def transform(self, texts):
        if not self.fitted:
            raise ValueError("Vectorizer has not been fitted yet")
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()

@st.cache_resource
def load_vectorizer_and_db():
    """ベクトライザーとデータベースをロード（キャッシュ機能付き）"""
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("ng_words_simple")

        return vectorizer, collection
    except Exception as e:
        return None, None

def search_ng_words(query, vectorizer, collection, threshold=0.3, max_results=10):
    """NGワードを検索"""
    try:
        query_embedding = vectorizer.transform([query])

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=max_results
        )

        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance

            if similarity >= threshold:
                result = {
                    'ng_word': metadata['ng_word'],
                    'replacement': metadata['replacement'],
                    'reason': metadata['reason'],
                    'risk_level': metadata['risk_level'],
                    'similarity': similarity,
                    'distance': distance
                }
                search_results.append(result)

        return search_results

    except Exception as e:
        return []

def display_ngword_search_results(results, query):
    """NGワード検索結果を表示"""
    if not results:
        st.warning("該当するNGワードが見つかりませんでした。")
        st.info("💡 類似度しきい値を下げるか、別のキーワードで検索してみてください。")
        return

    st.success(f"🎯 '{query}' に類似するNGワードを {len(results)} 件発見しました")

    risk_colors = {
        'high': '🔴',
        'mid': '🟡',
        'low': '🟢'
    }

    for i, result in enumerate(results, 1):
        risk_icon = risk_colors.get(result['risk_level'], '⚪')

        with st.expander(
            f"{risk_icon} {i}. 「{result['ng_word']}」 (類似度: {result['similarity']:.3f})",
            expanded=i<=3
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**❌ NGワード:**")
                st.code(result['ng_word'], language="text")

                st.write("**✅ 置換候補:**")
                st.code(result['replacement'], language="text")

            with col2:
                st.write("**📋 詳細情報:**")
                st.write(f"• **リスクレベル:** {result['risk_level'].upper()}")
                st.write(f"• **類似度:** {result['similarity']:.4f}")
                st.write(f"• **距離:** {result['distance']:.4f}")

            st.write("**🔍 NG理由:**")
            st.info(result['reason'])

# メインタブ
tab1, tab2, tab3 = st.tabs(["🖼️ 画像分析", "📄 ファイル分析", "🚫 NGワード検索"])

with tab1:
    st.subheader("🖼️ 画像分析機能")

    # カスタムプロンプトの設定
    with st.expander("🔧 分析プロンプト設定", expanded=False):
        default_prompt = """この画像を詳細に分析して、以下のJSON形式で結果を返してください：

画像からその画像のブロックごとに、オブジェクトと文字列を抽出し、以下のJSON形式で出力してください。
ルール:
- idは1からの連番
- objectは存在する場合のみ出力
- objectは { "label": "<15文字以内>", "category": "<人の顔/化粧品容器/その他>" }
- textは必ず配列
- query_stringはtextの要素ごとに1対1で生成
  - objectあり → "category text"
  - objectなし → "text"
JSON形式:
[
  {
    "id": <number>,
    "object": [ { "label": "<string>", "category": "<string>" } ],
    "text": ["<string>", "<string>"],
    "source": "<filename>",
    "query_string": ["<string>", "<string>"]
  }
]

必ず有効なJSONフォーマットで回答してください。"""

        analysis_prompt = st.text_area(
            "分析用プロンプト",
            value=default_prompt,
            height=200
        )

    # 画像アップロード
    uploaded_file = st.file_uploader(
        "📸 画像をアップロードして分析開始",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        key="main_uploader"
    )

    if uploaded_file is not None:
        # 画像表示
        st.image(uploaded_file, caption="アップロード画像", width=400)

        # 分析実行セクション
        st.markdown("---")
        st.subheader("🔍 分析結果")

        # 分析実行
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes))

        with st.spinner("画像を分析しています..."):
            json_data, raw_response = analyze_image_with_prompt(img, analysis_prompt)

        if json_data:
            st.success("✅ 分析完了！")

            # JSONデータをリスト表示
            display_json_data(json_data)

            # 生データの表示（オプション）
            with st.expander("📄 生レスポンス", expanded=False):
                st.text(raw_response)

            # 分析結果を履歴に保存
            from datetime import datetime
            st.session_state.analysis_results.append({
                "image": img_bytes,
                "json_data": json_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        else:
            st.error("❌ JSON形式での分析に失敗しました")
            st.text("生レスポンス:")
            st.text(raw_response)

    # 分析履歴
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📚 画像分析履歴")

        for idx, result in enumerate(reversed(st.session_state.analysis_results[-5:]), 1):  # 最新5件
            with st.expander(f"分析結果 {idx}", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(result["image"], width=200)
                with col2:
                    display_json_data(result["json_data"])

with tab2:
    st.subheader("📄 ファイル分析機能")
    st.write("テキストファイル、CSV、JSONなどを分析します")

    # 固定プロンプトの設定
    with st.expander("🔧 ファイル分析プロンプト設定", expanded=False):
        file_default_prompt = """アップロードされたファイルの内容を詳細に分析して、以下の観点で評価してください：

1. **ファイルの種類と構造**
2. **データの品質と整合性**
3. **主要な特徴と統計情報**
4. **潜在的な問題点や改善提案**
5. **ビジネス価値や活用方法**

分析は日本語で、わかりやすく構造化して回答してください。"""

        file_analysis_prompt = st.text_area(
            "ファイル分析用プロンプト",
            value=file_default_prompt,
            height=150
        )

    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "📁 ファイルをアップロードして分析開始",
        type=['txt', 'csv', 'json', 'md', 'py', 'js', 'html', 'xml'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("**📁 ファイル情報:**")
            st.write(f"• **名前:** {uploaded_file.name}")
            st.write(f"• **サイズ:** {uploaded_file.size:,} bytes")
            st.write(f"• **タイプ:** {uploaded_file.type}")

            # ファイル内容のプレビュー
            try:
                file_content = uploaded_file.read()
                if uploaded_file.type.startswith('text') or uploaded_file.name.endswith(('.txt', '.csv', '.json', '.md', '.py', '.js', '.html', '.xml')):
                    content_str = file_content.decode('utf-8')
                    st.write("**📄 内容プレビュー:**")
                    st.text(content_str[:500] + ("..." if len(content_str) > 500 else ""))
                else:
                    content_str = str(file_content[:1000])
                    st.write("**📄 バイナリ内容 (一部):**")
                    st.text(content_str[:200] + "...")
            except Exception as e:
                st.error(f"ファイル読み込みエラー: {str(e)}")
                content_str = ""

        with col2:
            if content_str and st.button("🔍 ファイル分析実行", type="primary"):
                st.subheader("🔍 分析中...")

                with st.spinner("Geminiでファイルを分析しています..."):
                    try:
                        # Geminiにファイル内容と固定プロンプトを送信
                        full_prompt = f"{file_analysis_prompt}\n\n【ファイル名】: {uploaded_file.name}\n【ファイル内容】:\n{content_str}"

                        response = st.session_state.model.generate_content(full_prompt)
                        analysis_result = response.text

                        st.success("✅ ファイル分析完了！")

                        # 分析結果を表示
                        st.markdown("---")
                        st.markdown("### 📋 分析結果")
                        st.markdown(analysis_result)

                        # 生レスポンスの表示（オプション）
                        with st.expander("📄 生レスポンス", expanded=False):
                            st.text(analysis_result)

                        # 分析結果を履歴に保存
                        from datetime import datetime
                        st.session_state.file_analysis_results.append({
                            "filename": uploaded_file.name,
                            "file_size": uploaded_file.size,
                            "file_type": uploaded_file.type,
                            "analysis_result": analysis_result,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    except Exception as e:
                        st.error(f"❌ 分析エラーが発生しました: {str(e)}")

    # ファイル分析履歴
    if st.session_state.file_analysis_results:
        st.markdown("---")
        st.subheader("📚 ファイル分析履歴")

        for idx, result in enumerate(reversed(st.session_state.file_analysis_results[-5:]), 1):  # 最新5件
            with st.expander(f"{result['filename']} ({result['timestamp']})", expanded=False):
                st.write(f"**ファイルサイズ:** {result['file_size']:,} bytes")
                st.write(f"**ファイルタイプ:** {result['file_type']}")
                st.markdown("**分析結果:**")
                st.markdown(result['analysis_result'])

with tab3:
    st.subheader("🚫 NGワード検索機能")
    st.write("広告・マーケティング文言のNGワードをベクトル類似性で検出します")

    # データベース状況の確認
    if not os.path.exists("./chroma_db") or not os.path.exists("tfidf_vectorizer.pkl"):
        st.error("❌ NGワードデータベースが見つかりません。")
        st.info("管理者に連絡してデータベースをセットアップしてもらってください。")
    else:
        # 検索パラメータ
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "検索したいキーワードやフレーズを入力してください",
                placeholder="例: 美白効果、ニキビが治る、アンチエイジング、肌荒れ改善"
            )

        with col2:
            search_button = st.button("🔍 検索実行", type="primary")

        # 詳細設定
        with st.expander("⚙️ 検索設定", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                threshold = st.slider(
                    "類似度しきい値",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="この値以上の類似度を持つNGワードを表示します"
                )

            with col2:
                max_results = st.number_input(
                    "最大表示件数",
                    min_value=1,
                    max_value=50,
                    value=10
                )

        # 検索実行
        if (search_button or query) and query.strip():
            vectorizer, collection = load_vectorizer_and_db()

            if vectorizer is not None and collection is not None:
                with st.spinner("NGワードを検索中..."):
                    results = search_ng_words(query, vectorizer, collection, threshold, max_results)

                display_ngword_search_results(results, query)

                # 検索のヒント
                if not results:
                    st.info("""
                    💡 **検索のヒント:**
                    - TF-IDFは文字レベルの類似性を見るため、完全一致に近い表現がより高い類似度を示します
                    - 類似度しきい値を下げてみてください (推奨: 0.1-0.4)
                    - 同じ意味でも表現が異なる場合は検出されにくい可能性があります
                    """)
            else:
                st.error("❌ データベースの読み込みに失敗しました。")

        # バッチチェック機能
        st.markdown("---")
        st.subheader("📝 バッチチェック")
        st.write("複数のテキストを一度にチェックできます")

        batch_text = st.text_area(
            "チェックしたいテキスト（1行1項目）",
            height=150,
            placeholder="美白効果抜群\nニキビが治る\nアンチエイジング効果\n肌荒れ改善\n副作用なし"
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_threshold = st.slider(
                "バッチ類似度しきい値",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05
            )
        with col2:
            batch_max_results = st.number_input(
                "バッチ最大結果数",
                min_value=1,
                max_value=20,
                value=5
            )

        if st.button("🚀 バッチチェック実行") and batch_text.strip():
            vectorizer, collection = load_vectorizer_and_db()
            if vectorizer is None or collection is None:
                st.error("❌ データベースの読み込みに失敗しました。")
            else:
                lines = [line.strip() for line in batch_text.split('\n') if line.strip()]

                results_data = []
                progress_bar = st.progress(0)

                for i, line in enumerate(lines):
                    results = search_ng_words(line, vectorizer, collection, batch_threshold, batch_max_results)

                    if results:
                        for result in results:
                            results_data.append({
                                'チェック対象': line,
                                'NGワード': result['ng_word'],
                                '置換候補': result['replacement'],
                                'リスクレベル': result['risk_level'],
                                '類似度': f"{result['similarity']:.3f}"
                            })
                    else:
                        results_data.append({
                            'チェック対象': line,
                            'NGワード': '(検出なし)',
                            '置換候補': '問題なし',
                            'リスクレベル': 'safe',
                            '類似度': '0.000'
                        })

                    progress_bar.progress((i + 1) / len(lines))

                if results_data:
                    st.write(f"### 📊 バッチチェック結果: {len(lines)} 項目を検査")

                    df = pd.DataFrame(results_data)

                    # リスクレベル別のカウント
                    risk_counts = df[df['リスクレベル'] != 'safe']['リスクレベル'].value_counts()
                    if not risk_counts.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🔴 高リスク", risk_counts.get('high', 0))
                        with col2:
                            st.metric("🟡 中リスク", risk_counts.get('mid', 0))
                        with col3:
                            st.metric("🟢 低リスク", risk_counts.get('low', 0))
                        with col4:
                            st.metric("✅ 安全", len(df[df['リスクレベル'] == 'safe']))

                    # 結果の表示
                    st.dataframe(df, use_container_width=True)

                    # CSV出力機能
                    from datetime import datetime
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 結果をCSVでダウンロード",
                        data=csv,
                        file_name=f"ngword_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

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

    if st.button("分析履歴をクリア"):
        st.session_state.analysis_results = []
        st.session_state.file_analysis_results = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 使い方")
    st.markdown("""
    **🖼️ 画像分析:**
    1. 分析用プロンプトを設定
    2. 画像をアップロード
    3. JSON形式で自動分析

    **📄 ファイル分析:**
    1. プロンプトを設定（カスタム可）
    2. ファイルをアップロード
    3. Geminiによる詳細分析

    ### 対応フォーマット
    **画像**: PNG, JPG, JPEG, GIF, WebP
    **ファイル**: TXT, CSV, JSON, MD, PY, JS, HTML, XML
    """)

    st.markdown("---")
    st.caption("💡 AIによるマルチメディア分析ツール")