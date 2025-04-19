# Language Processing API

日本語と英語のテキスト処理を行うためのAPIサービスです。形態素解析、単語ベクトル取得、テキスト類似度計算などの機能を提供します。

## 基本情報

- ベースURL: `/v1`
- 認証: すべてのAPIリクエストには `X-API-KEY` ヘッダーが必要です
- コンテンツタイプ: `application/json`

## API エンドポイント

### 1. 形態素解析 API

テキストを形態素解析してその結果を返します。

- **エンドポイント**: `/v1/parse`
- **メソッド**: POST
- **リクエスト本文**:

```json
{
    "text": "解析するテキスト"
}
```

- **レスポンス例**:

```json
{
    "results": [
        {
            "surface": "解析",
            "feature": "名詞,サ変接続,*,*,*,*,解析,カイセキ,カイセキ"
        },
        {
            "surface": "する",
            "feature": "動詞,自立,*,*,サ変・スル,基本形,する,スル,スル"
        },
        {
            "surface": "テキスト",
            "feature": "名詞,一般,*,*,*,*,テキスト,テキスト,テキスト"
        }
    ]
}
```

### 2. 単語ベクトル取得 API

単語のベクトル表現を取得します。モデルに単語が存在しない場合は分かち書きを行い平均ベクトルを返します。

- **エンドポイント**: `/v1/vector/word`
- **メソッド**: POST
- **リクエスト本文**:

```json
{
    "word": "単語"
}
```

- **レスポンス例（成功時）**:

```json
{
    "success": true,
    "vector": [0.123, 0.456, ...]
}
```

- **レスポンス例（成功・複合語）**:

```json
{
    "success": true,
    "vector": [0.123, 0.456, ...],
    "split_words": ["単", "語"],
    "is_compound_word": true
}
```

- **レスポンス例（失敗時）**:

```json
{
    "success": false,
    "error": "No words found in model vocabulary from word: 単語"
}
```

### 3. テキスト平均ベクトル取得 API

複数のテキストから平均ベクトルを計算します。

- **エンドポイント**: `/v1/vector/texts`
- **メソッド**: POST
- **リクエスト本文**:

```json
{
    "texts": ["テキスト1", "テキスト2", ...]
}
```

- **レスポンス例（成功時）**:

```json
{
    "success": true,
    "vector": [0.123, 0.456, ...],
    "processed_words": [
        {
            "original": "テキスト1",
            "used": ["テキスト"]
        },
        ...
    ]
}
```

- **レスポンス例（失敗時）**:

```json
{
    "success": false,
    "error": "No words found in model vocabulary: テキスト1, テキスト2"
}
```

### 4. 単語類似度計算 API

単語ペア間の類似度を計算します。

- **エンドポイント**: `/v1/similarity`
- **メソッド**: POST
- **リクエスト本文**:

```json
{
    "pairs": [
        ["単語1", "単語2"],
        ["単語3", "単語4"],
        ...
    ]
}
```

- **レスポンス例**:

```json
{
    "pairs": [
        {
            "word1": "単語1",
            "word2": "単語2",
            "similarity": 0.765,
            "error": ""
        },
        {
            "word1": "単語3",
            "word2": "単語4",
            "similarity": 0.432,
            "error": ""
        },
        ...
    ],
    "error": ""
}
```

## セットアップと実行

### 環境変数

以下の環境変数を `.env` ファイルに設定できます：

- `SECRET_KEY`: アプリケーションの秘密鍵
- `API_KEYS`: カンマ区切りのAPI鍵リスト（例: `key1,key2,key3`）
- `MODEL_PATH`: 単語ベクトルモデルのパス
- `MECAB_DICT_PATH`: MeCab辞書のパス
- `PORT`: サーバーのポート番号（デフォルト: 8080）
- `FLASK_ENV`: 開発環境の場合は `development`

### ファイルを編集

```shell
sudo nano /etc/systemd/system/language_processing_api_run.service
```

### 再起動

```shell
sudo systemctl restart language_processing_api_run.service
```

### 実行

```shell
gunicorn --bind 0.0.0.0:3000 --workers 1 --threads 8 --timeout 0 main:app
```
