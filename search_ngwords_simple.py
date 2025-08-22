#!/usr/bin/env python3
"""
シンプルNGワード検索・動作確認スクリプト
TF-IDFベクトル化を使ったChromaDBでのNGワード検索
"""

import streamlit as st
import chromadb
import json
import os
import pandas as pd
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SimpleJapaneseVectorizer:
    """シンプルな日本語ベクトル化クラス"""

    def __init__(self):
        # 日本語対応のTF-IDFベクトライザー
        self.vectorizer = TfidfVectorizer(
            analyzer='char',  # 文字レベルでの解析
            ngram_range=(2, 4),  # 2-4文字のn-gram
            max_features=5000,  # 特徴量の最大数
            lowercase=False,  # 日本語では小文字変換は不要
            token_pattern=None  # デフォルトのトークナイザーを無効化
        )
        self.fitted = False

    def fit_transform(self, texts):
        """テキストをベクトル化（訓練+変換）"""
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return vectors.toarray()  # 密な配列に変換

    def transform(self, texts):
        """テキストをベクトル化（変換のみ）"""
        if not self.fitted:
            raise ValueError("Vectorizer has not been fitted yet")
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()

# ページ設定
st.set_page_config(
    page_title="NGワード検索システム (TF-IDF版)",
    page_icon="🚫",
    layout="wide"
)

@st.cache_resource
def load_vectorizer_and_db():
    """ベクトライザーとデータベースをロード（キャッシュ機能付き）"""
    try:
        # TF-IDFベクトライザーの読み込み
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # ChromaDBクライアントの初期化
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("ng_words_simple")

        return vectorizer, collection
    except Exception as e:
        st.error(f"ベクトライザーまたはデータベースの読み込みに失敗しました: {str(e)}")
        return None, None

def search_ng_words(query, vectorizer, collection, threshold=0.3, max_results=10):
    """NGワードを検索"""
    try:
        # クエリをベクトル化
        query_embedding = vectorizer.transform([query])

        # ChromaDBで類似検索
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=max_results
        )

        # 結果を整理
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance  # コサイン類似度に変換

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
        st.error(f"検索中にエラーが発生しました: {str(e)}")
        return []

def display_search_results(results, query):
    """検索結果を表示"""
    if not results:
        st.warning("該当するNGワードが見つかりませんでした。")
        st.info("💡 類似度しきい値を下げるか、別のキーワードで検索してみてください。")
        return

    st.success(f"🎯 '{query}' に類似するNGワードを {len(results)} 件発見しました")

    # リスクレベル別に色分け
    risk_colors = {
        'high': '🔴',
        'mid': '🟡',
        'low': '🟢'
    }

    for i, result in enumerate(results, 1):
        risk_icon = risk_colors.get(result['risk_level'], '⚪')

        with st.expander(
            f"{risk_icon} {i}. 「{result['ng_word']}」 (類似度: {result['similarity']:.3f})",
            expanded=i<=3  # 上位3件は展開表示
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

def batch_check_mode():
    """バッチチェックモード"""
    st.subheader("📝 バッチチェックモード")
    st.write("複数のテキストを一度にチェックできます")

    # テキストエリアで複数行入力
    batch_text = st.text_area(
        "チェックしたいテキスト（1行1項目）",
        height=200,
        placeholder="美白効果抜群\nニキビが治る\nアンチエイジング効果\n肌荒れ改善\n副作用なし"
    )

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        batch_threshold = st.slider(
            "類似度しきい値",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05
        )

    with col2:
        batch_max_results = st.number_input(
            "最大結果数",
            min_value=1,
            max_value=20,
            value=5
        )

    if st.button("🚀 バッチチェック実行") and batch_text.strip():
        vectorizer, collection = load_vectorizer_and_db()
        if vectorizer is None or collection is None:
            return

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
                # NGワードが見つからない場合も記録
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
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 結果をCSVでダウンロード",
                data=csv,
                file_name=f"ngword_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """メイン関数"""
    st.title("🚫 NGワード検索システム (TF-IDF版)")
    st.write("広告・マーケティング文言のNGワードをTF-IDFベクトル化で検出します")

    # データベース状況の確認
    if not os.path.exists("./chroma_db"):
        st.error("❌ ChromaDBが見つかりません。先に `setup_ngword_db_simple.py` を実行してください。")
        st.code("python setup_ngword_db_simple.py")
        return

    if not os.path.exists("tfidf_vectorizer.pkl"):
        st.error("❌ TF-IDFベクトライザーが見つかりません。先に `setup_ngword_db_simple.py` を実行してください。")
        return

    # 設定情報の表示
    if os.path.exists("ngword_db_config.json"):
        with open("ngword_db_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # with st.expander("📊 データベース情報", expanded=False):
        #     col1, col2, col3, col4 = st.columns(4)
        #     with col1:
        #         st.metric("総NGワード数", config.get("total_records", "不明"))
        #     with col2:
        #         st.metric("ベクトル次元", config.get("vector_dim", "不明"))
        #     with col3:
        #         st.metric("モデル", "TF-IDF")
        #     with col4:
        #         st.metric("最終更新", config.get("created_at", "不明")[:10])

    # メインタブ
    tab1, tab2, tab3 = st.tabs(["🔍 単語検索", "📝 バッチチェック", "📈 統計情報"])

    with tab1:
        st.subheader("🔍 単語検索モード")

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
                    help="この値以上の類似度を持つNGワードを表示します（TF-IDFでは低めの値を推奨）"
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

                display_search_results(results, query)

                # 検索のヒント
                if not results:
                    st.info("""
                    💡 **検索のヒント:**
                    - TF-IDFは文字レベルの類似性を見るため、完全一致に近い表現がより高い類似度を示します
                    - 類似度しきい値を下げてみてください (推奨: 0.1-0.4)
                    - 同じ意味でも表現が異なる場合は検出されにくい可能性があります
                    """)

    with tab2:
        batch_check_mode()

    with tab3:
        st.subheader("📈 統計情報")

        vectorizer, collection = load_vectorizer_and_db()
        if vectorizer is not None and collection is not None:
            total_count = collection.count()
            st.metric("総NGワード数", total_count)

            # サンプル表示
            sample_results = collection.get(limit=5)
            if sample_results['metadatas']:
                st.write("### 📝 サンプルNGワード")
                sample_df = pd.DataFrame([
                    {
                        'NGワード': meta['ng_word'],
                        '置換候補': meta['replacement'][:50] + "..." if len(meta['replacement']) > 50 else meta['replacement'],
                        'リスクレベル': meta['risk_level']
                    }
                    for meta in sample_results['metadatas'][:5]
                ])
                st.dataframe(sample_df, use_container_width=True)

            # # TF-IDFベクトライザーの情報
            # if vectorizer is not None:
            #     st.write("### 🔧 TF-IDFベクトライザー情報")
            #     col1, col2, col3 = st.columns(3)
            #     with col1:
            #         st.metric("N-gram範囲", "2-4文字")
            #     with col2:
            #         st.metric("最大特徴量", vectorizer.vectorizer.max_features)
            #     with col3:
            #         actual_features = len(vectorizer.vectorizer.get_feature_names_out()) if hasattr(vectorizer.vectorizer, 'get_feature_names_out') else "不明"
            #         st.metric("実際の特徴量", actual_features)

if __name__ == "__main__":
    main()