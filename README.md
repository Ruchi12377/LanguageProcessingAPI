# Language Processing API

## ファイルを編集

```shell
sudo nano /etc/systemd/system/language_processing_api_run.service
```

## 再起動

```shell
sudo systemctl restart language_processing_api_run.service
```

## 実行

```shell
gunicorn --bind 0.0.0.0:3000 --workers 1 --threads 8 --timeout 0 main:app
```
