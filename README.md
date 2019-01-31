# Octopus

## 概要
###### Python+TensorFlow製のFXトレードbot。複数の時間足に対してマルチタイムフレームでの為替データの解析が可能。 現在は時系列データの解析に効果が期待されるRecurrent Neural Networkの一種であるLSTMを用いた予測や線形回帰を用いたトレンドラインの自動生成、取引の自動化が可能。 LINE notifyを用いた取引状況の簡易な通知が可能。 tickデータの取得にOandaのAPIを使用しているため、bot利用にはOanda Rest API V20対応のアカウントが必要。 
