## 改修中（Under the construction）

Schema is here: http://go.bubbl.us/a76bd4/28ec?/FX-trading-bot


# 使い方
```
$python3 bot.py auth.json
```
実行には認証用のjson形式の認証キー（auth.json）が必要となります。
認証キーには以下の3つが格納されています。
- LINE notify token
- Oanda id
- Oanda token

jsonの内容は以下のようになっています。
```
{
 "line_token": "your LINE notify token", 
 "oanda_id": "XXX-XXX-XXXXXXX-XXX",
 "oanda_token": "your oanda token"
}
```


#### 必要なもの（required）
- Oanda Rest API V20のアカウントとトークン
- (LINE通知機能を使用する場合）LINE notify用のトークン
- Oanda Rest API V20 account and token
- LINE notify token
