# Octopus
##### Python+TensorFlow製のFXトレードbot
#### 特徴
- シングル/マルチタイムフレームでの解析が可能
- Pythonを使った自作のアルゴリズムの実装が可能
- 時系列解析に用いられるニューラルネットワークの一種であるRNN/LSTMが使用可能
- LINE notifyを用いた取引状況の簡易的な通知が可能

#### 必要なもの
- Oanda Rest API V20のアカウントとトークン
- (LINE通知機能を使用する場合）LINE notify用のトークン

#### 必要なモジュール
- NumPy
- Pandas
- TensorFlow
- matplotlib
- mpl_finance
- sys
- json
- tqdm
- requests
- scipy
- time
- sklearn

#### 実装済み
- トレンドラインの自動生成
- 波形の振動を検知するアルゴリズムを追加
- RNN/LSTMの学習器、予測器

#### 実装予定
- close用のアルゴリズム
- RNNの多層化
