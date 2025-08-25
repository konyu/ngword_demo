#!/usr/bin/env python3
"""
NGワードデータベース構築スクリプト (FAISS版)
input.csvからデータを読み込み、FAISSインデックスを構築して単一ファイルに保存
"""

import pandas as pd
import os
import json
from datetime import datetime
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# FAISSのインポート
try:
    import faiss
except ImportError:
    print("FAISSがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install faiss-cpu")
    exit(1)

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

def setup_ngword_faiss():
    """NGワードデータベースをFAISSで構築"""
    try:
        # CSVファイルの読み込み
        logger.info("CSVファイル読み込み開始...")
        if not os.path.exists('input.csv'):
            logger.error("input.csvが見つかりません")
            return False
            
        df = pd.read_csv('input.csv', encoding='utf-8')
        logger.info(f"読み込み完了: {len(df)}件のNGワードデータ")
        
        # シンプルベクトライザーの初期化
        logger.info("ベクトライザーの初期化...")
        vectorizer = SimpleJapaneseVectorizer()
        logger.info("ベクトライザーの初期化完了")
        
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
                "id": i,
                "ng_word": ng_word,
                "replacement": replacement, 
                "reason": reason,
                "risk_level": risk_level,
                "created_at": datetime.now().isoformat()
            }
            metadatas.append(metadata)
        
        # FAISSインデックスの作成
        logger.info("FAISSインデックス構築中...")
        dimension = embeddings.shape[1]
        
        # コサイン類似度用のインデックス（L2正規化 + 内積）
        index = faiss.IndexFlatIP(dimension)
        
        # ベクトルをL2正規化（コサイン類似度のため）
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # インデックスにベクトルを追加
        index.add(embeddings_normalized)
        logger.info(f"FAISSインデックス構築完了: {index.ntotal}件のベクトルを追加")
        
        # 全データを単一ファイルに保存
        ngword_data = {
            'faiss_index': index,
            'vectorizer': vectorizer,
            'metadatas': metadatas,
            'config': {
                'total_records': len(ng_words),
                'vector_dim': dimension,
                'created_at': datetime.now().isoformat(),
                'description': 'NGワードFAISSデータベース（TF-IDF + コサイン類似度）'
            }
        }
        
        # 既存ファイルを削除
        output_file = "ngword_faiss.pkl"
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info("既存のFAISSファイルを削除しました")
        
        # ファイル保存
        logger.info("FAISSデータベースをファイルに保存中...")
        with open(output_file, 'wb') as f:
            pickle.dump(ngword_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # ファイルサイズ確認
        file_size = os.path.getsize(output_file)
        logger.info(f"保存完了: {output_file} ({file_size:,} bytes)")
        
        # 設定情報をJSON形式でも保存（確認用）
        config_file = "ngword_faiss_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(ngword_data['config'], f, ensure_ascii=False, indent=2)
        logger.info(f"設定情報保存: {config_file}")
        
        # テスト検索の実行
        logger.info("テスト検索実行中...")
        test_query = "美白効果"
        query_vector = vectorizer.transform([test_query]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = index.search(query_vector, 5)
        logger.info(f"テスト検索結果 (クエリ: '{test_query}'):")
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx >= 0:  # 有効なインデックス
                similarity = distance  # 正規化済み内積 = コサイン類似度
                ng_word = metadatas[idx]['ng_word']
                logger.info(f"  {i+1}. {ng_word} (類似度: {similarity:.3f})")
        
        logger.info("✅ NGワードFAISSデータベース構築が完了しました！")
        return True
        
    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 NGワードFAISSデータベース構築を開始します...")
    success = setup_ngword_faiss()
    
    if success:
        print("\n✅ 構築完了！以下のファイルが生成されました:")
        print("  - ngword_faiss.pkl (メインデータベース)")
        print("  - ngword_faiss_config.json (設定情報)")
        print("\n次のステップ:")
        print("  1. requirements.txtを更新してfaiss-cpuを追加")
        print("  2. app.pyをFAISS対応に更新")
        print("  3. 古いChromaDBファイルを削除")
    else:
        print("\n❌ 構築に失敗しました。ログを確認してください。")
        exit(1)