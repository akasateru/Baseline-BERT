
# Baseline-BERT
## 説明
入力に文章とクラス名を繋げたものを用いて、文章がクラス名に関連しているかどうかを判定する２値分類器を構築。クラス情報は、'this text is about (hypothesis)'。クラス名が2つある場合,'this text is about (hypothesis) or (hypothesis)'。hypothesisはwordnetから取得。各データごとに1つの不正解データを作成。不正解データは10クラスの中から正解データを除いた9クラスの中からランダムに選択。

## BERT model
BERTの事前学習済みモデルは、公式のbert-base-uncasedを使用。  
https://github.com/google-research/bert

## 学習データ(fine tuning data)
yahoo topic datasetを使用。クラス数10、データ数130万（各6万5千）  
https://github.com/yinwenpeng/BenchmarkingZeroShot
- topic/train_pu_half_v0.txt
- topic/train_pu_half_v1.txt

## テストデータ
### DBpediaデータセット
DBpedia2014から14クラスを選択したもの。クラス情報は単語間にスペースを挿入。学習データは、各40,000、計560,000、テストデータは、各5,000、計70,000。
https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k
から取得。
- dbpedia_csv
    - class.txt
    - readme.txt
    - train.csv
    - test.csv

## 前処理
- 括弧内文章の削除
- 記号文字の削除
- 著者名の削除
- スペースの調整

## パラメータ
- BATCH_SIZE = 32
- EPOCHS = 20
- LR = 5e-5
- SEQ_LEN = 64
- BERT_DIM = 768

## 使い方
```
set -e
python make_dataset.py
python BERT_matching_train.py
python BERT_matching_test.py
```