#!/usr/bin/env python3
"""
シンプルなNGワードデータベース構築スクリプト
TF-IDFベクトル化を使用してNGワードデータをChromaDBに保存
"""

import pandas as pd
import chromadb
import os
import json
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def setup_ngword_database():
    """NGワードデータベースをセットアップ"""
    try:
        # CSVファイルの読み込み
        logger.info("CSVファイル読み込み開始...")
        df = pd.read_csv('input.csv', encoding='utf-8')
        logger.info(f"読み込み完了: {len(df)}件のNGワードデータ")
        
        # シンプルベクトライザーの初期化
        logger.info("ベクトライザーの初期化...")
        vectorizer = SimpleJapaneseVectorizer()
        logger.info("ベクトライザーの初期化完了")
        
        # ChromaDBクライアントの初期化
        logger.info("ChromaDBクライアント初期化...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # コレクション作成（既存があれば削除して再作成）
        collection_name = "ng_words_simple"
        try:
            chroma_client.delete_collection(collection_name)
            logger.info("既存のコレクションを削除しました")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "NGワードのTF-IDFベクトル検索コレクション"}
        )
        logger.info("新しいコレクションを作成しました")
        
        # データの準備
        ng_words = df['ng_word'].tolist()
        replacements = df['replacement'].tolist()
        reasons = df['reason'].tolist()
        risk_levels = df['risk_level'].tolist()
        
        # 各NGワードの埋め込みベクトルを生成
        logger.info("TF-IDFベクトル生成中...")
        embeddings = vectorizer.fit_transform(ng_words)
        logger.info(f"TF-IDFベクトル生成完了: {embeddings.shape}")
        
        # メタデータの準備
        metadatas = []
        for i, (ng_word, replacement, reason, risk_level) in enumerate(zip(ng_words, replacements, reasons, risk_levels)):
            metadata = {
                "ng_word": ng_word,
                "replacement": replacement, 
                "reason": reason,
                "risk_level": risk_level,
                "created_at": datetime.now().isoformat()
            }
            metadatas.append(metadata)
        
        # ChromaDBにデータを追加
        logger.info("ChromaDBにデータを追加中...")
        collection.add(
            embeddings=embeddings.tolist(),
            documents=ng_words,  # 検索対象のテキスト
            metadatas=metadatas,
            ids=[f"ng_word_{i}" for i in range(len(ng_words))]
        )
        
        logger.info(f"データベース構築完了: {len(ng_words)}件のNGワードを追加")
        
        # 統計情報の表示
        risk_counts = df['risk_level'].value_counts()
        logger.info("リスクレベル別統計:")
        for level, count in risk_counts.items():
            logger.info(f"  {level}: {count}件")
        
        # ベクトライザーを保存
        import pickle
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("TF-IDFベクトライザーを保存しました")
        
        # 設定情報を保存
        config = {
            "model_name": "TF-IDF (char-level, 2-4 ngram)",
            "collection_name": collection_name,
            "total_records": len(ng_words),
            "vector_dim": embeddings.shape[1],
            "created_at": datetime.now().isoformat(),
            "csv_file": "input.csv",
            "vectorizer_file": "tfidf_vectorizer.pkl"
        }
        
        with open("ngword_db_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info("設定ファイル 'ngword_db_config.json' を保存しました")
        return True
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"詳細: {traceback.format_exc()}")
        return False

def verify_database():
    """データベースの動作確認"""
    try:
        logger.info("データベースの動作確認開始...")
        
        # ChromaDBクライアントの初期化
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("ng_words_simple")
        
        # ベクトライザーの読み込み
        import pickle
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        # コレクションの基本情報
        count = collection.count()
        logger.info(f"コレクション内のレコード数: {count}")
        
        # サンプル検索
        test_queries = ["ニキビが良くなる", "シミが薄くなる", "美白効果", "アンチエイジング"]
        
        for query in test_queries:
            logger.info(f"\n検索テスト: '{query}'")
            query_embedding = vectorizer.transform([query])
            
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3
            )
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                similarity = 1 - results['distances'][0][i]  # コサイン類似度に変換
                logger.info(f"  {i+1}. NGワード: '{metadata['ng_word']}' (類似度: {similarity:.3f})")
                logger.info(f"     置換候補: '{metadata['replacement']}'")
                logger.info(f"     リスクレベル: {metadata['risk_level']}")
        
        logger.info("\nデータベース動作確認完了")
        return True
        
    except Exception as e:
        logger.error(f"動作確認でエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"詳細: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("シンプルNGワードデータベース構築スクリプト")
    print("=" * 60)
    
    # データベース構築
    if setup_ngword_database():
        print("\n✅ データベース構築が成功しました")
        
        # 動作確認
        if verify_database():
            print("✅ 動作確認も完了しました")
        else:
            print("❌ 動作確認に失敗しました")
    else:
        print("❌ データベース構築に失敗しました")
    
    print("\n" + "=" * 60)