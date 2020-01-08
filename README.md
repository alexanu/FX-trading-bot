# Octopus
##### OctopusはPython製のFXトレードbotです
##### "Octopus" is FX trading bot made by Python
#### 特徴（feature）
- シングル/マルチタイムフレームでの解析が可能
- Pythonを使った自作のアルゴリズムの実装が可能
- 時系列解析に用いられるニューラルネットワークの一種であるRNN/LSTMが使用可能
- LINE notifyを用いた取引状況の簡易的な通知が可能
- It can analyze both single and multi timeframe candle stick data.
- You can implement your original trading algorythm to the bot by using Python
- You can use RNN or LSTM to analyze candle stick data
- It send trade imformation to you by using LINE notify

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

#### 必要なモジュール（please install following modules before you run "Octopus"）
- NumPy
- Pandas
- TensorFlow
- matplotlib
- mpl_finance
- tqdm
- scipy
- scikit-learn
以上の依存するモジュールはrequirements.txtを用いてまとめて取得することができる
```
$pip3 install -r requirements.txt
```

#### 実装済み
- トレンドラインの自動生成
- 波形の振動を検知するアルゴリズムを追加
- RNN/LSTMの学習器、予測器
- 本番環境/バーチャル環境の切り替え

#### 今後の予定
- close用のアルゴリズム
- RNNの多層化->統計処理のアルゴリズム実装のためしばらく休止
- 並列化->Pythonのスレッド並列実装のコストとメリットが見合わないため先送り
- tqdmからfastprogressへ切り替え

#### bug fix
- Connectionが切れたときの復帰処理の漏れを修正
