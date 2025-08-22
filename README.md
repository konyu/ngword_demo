# Gemini Chat Application

StreamlitとGoogle Gemini APIを使用したチャットアプリケーション

## 機能

- 🤖 最新のGemini 2.5モデルに対応
- 🖼️ 画像添付機能（画像を分析・質問可能）
- 💬 会話履歴の保持
- 🔄 モデルの動的切り替え
- 🔐 ベーシック認証によるアクセス制限
- 🎨 直感的なUI/UX

## 使用技術

- Streamlit 1.48.1
- Google Generative AI (Gemini API)
- Python 3.9+

## セットアップ

### 1. Gemini API キーの取得

[Google AI Studio](https://makersuite.google.com/app/apikey)でAPIキーを取得してください。

### 2. 環境変数の設定

Streamlit Community Cloudでデプロイする場合：
1. アプリの設定画面で「Secrets」を開く
2. 以下を追加：
```toml
GEMINI_API_KEY = "your-api-key-here"

# ベーシック認証（オプション）
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "your-secure-password"
```

ローカルで実行する場合：
`.env`ファイルを作成し、以下を記入：
```
GEMINI_API_KEY=your-api-key-here

# ベーシック認証（オプション）
AUTH_USERNAME=admin
AUTH_PASSWORD=your-secure-password
```

**注意事項：**
- `AUTH_USERNAME`と`AUTH_PASSWORD`を設定しない場合、デフォルト値（admin/password）が使用されます
- 本番環境では必ず強力なパスワードを設定してください

### 3. ローカルでの実行

```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# アプリの起動
streamlit run app.py
```

## デプロイ

このアプリはStreamlit Community Cloudで簡単にデプロイできます。

1. このリポジトリをGitHubにプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud)にサインイン
3. 「New app」をクリック
4. リポジトリを選択してデプロイ
5. Secretsに環境変数を設定：
   - `GEMINI_API_KEY`: Gemini APIキー（必須）
   - `AUTH_USERNAME`: 認証用ユーザー名（オプション）
   - `AUTH_PASSWORD`: 認証用パスワード（オプション）

## 利用可能なモデル

- **Gemini 2.5 Flash**: 最新・高速・バランス型
- **Gemini 2.5 Pro**: 最高性能・複雑タスク対応
- **Gemini 2.5 Flash Lite**: 低コスト・低遅延
- **Gemini 2.0 Flash**: 前世代の高速モデル
- **Gemini 1.5 Flash**: レガシーモデル
- **Gemini 1.5 Flash-8B**: 軽量モデル

## ライセンス

MIT

## 作成者

2025年8月作成