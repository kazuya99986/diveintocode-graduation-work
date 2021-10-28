# レビュー分析アプリケーション【Review Analyzer】

## 制作の目的
このアプリケーションはDIVE INTO CODE（機械学習エンジニアコース2107）の卒業課題として作成しました。初学者のため、高度な機能は備えておりませんが、レビュー分析に興味がある方、またお困りの方へ少しでもお力添えとなれば幸いです  

## アプリケーションの概要
レビューに対する感情分析をするのがメインの機能になります。レビューをポジティブ、ネガティブ、ニュートラルの3クラスに分類します。
またnlplotを用いた各種可視化グラフ、ロジスティック回帰を用いた説明変数（頻出キーワード）の決定係数算出の機能が含まれます。

## 使用したデータセット
以下データセットにて学習しました

- [東北大学 乾研究室日本語BERTモデル（v2）]
- [リクルート社「じゃらん」クチコミに基づく学術研究用データセット]

[東北大学 乾研究室日本語BERTモデル（v2）]: https://huggingface.co/cl-tohoku/bert-base-japanese-v2
[リクルート社「じゃらん」クチコミに基づく学術研究用データセット]:  https://github.com/megagonlabs/jrte-corpus


## プログラムのインストール
当プログラムをローカル環境へインストールしてください。  
また以下より追加データをダウンロードの上、`base_model`フォルダに設置してください。  
  
[pytorch_model.bin]

[pytorch_model.bin]: https://drive.google.com/file/d/1MuHOvyAHhka8cVGeKSQC-KzPOFhwDe1Y/view?usp=sharing

## ライブラリのインストール
必要ライブラリをインストールしてください  
  
`$ pip install -r requirements.txt`

## インポート
当ライブラリをインポートしてください  
  
`import review_analyzer`

## 推論データの用意
推論データ（レビュー文章）を以下の型にて用意ください　　

`list[str]`

## インスタンス化
Analyzerクラスをインスタンス化してください。

`インスタンス変数.review_analyzer.Analyzer()`  

ご自身の学習モデルを使用されたい方はそのモデルが保存されたディレクトリのパスを`model_PATH`へ指定してください。  
またその他の引数は以下の通りです。 ※全て任意設定です

    model_PATH: str
                使用するモデルが保存されたPATH
    stopwords : list[str]
                nlplotでの可視化の除外するキーワード
    wakati_ng_type: list[str]
                mecabでの分かち書きの際に除外する品詞種別（品詞名一覧：https://taku910.github.io/mecab/posid.html)
                デフォルト設定：['助詞','助動詞','動詞','記号','接頭辞','接続詞','代名詞','連体詞']
    coef_type : list[str]
                決定係数算出の際の説明変数
                デフォルト設定：['普通名詞']
    coef_num　: int
                決定係数算出の際、ネガティブ・ポジテイブそれぞれ上位何キーワードを説明変数に用いるか
                ※少ないサンプル数の際に、大きい値を設定すると結果が思わしくないものとなります（目安：サンプル数1000の際には、10程度）
                ※ネガティブ・ポジテイブそれぞれからキーワードを抽出し、重複排除を行う為、入力された値と実際出力される値は異なります
    coef_vocabulary: list[str]
                決定係数算出の際の説明変数（計算の際には出現数に置き換えられます）
                ※coef_vocabularyが設定された場合は、coef_numの設定は無視されます


## 推論
`text_input`に推論データを設定の上、以下コードを実行してください。

`インスタンス変数.review_analyzer.analyze(text_input)`  

【学習時間の目安】  
＜GPU使用時＞1分/5000レビュー  
＜CPUのみ＞1時間/5000レビュー  

※メモリがクラッシュしてしまう場合は引数の`batch_size`へ少ない値（4や8など）を設定してください（デフォルト=32）

## インスタンス変数一覧

    df: 入力データや分析結果のまとめ（DataFrame）
      <カラム説明>
      text_num: レビューの番号（番号が同一の場合、同一のレビューとなります）
      text    : レビューを　。　！　？　で区切った短文の文章です
      label・・・　短文に対する分析結果です(2:ポジティブ、1：ニュートラル、0：ネガティブ）
      total_evaluation・・・レビュー全体に対する分析結果です。短文の評価結果をもとに分類されています
      wakati・・・形態素解析によるわかち書きです（情報が少ないと思われる品詞の単語は削除）
      
    coef_vocabulary: ロジスティック回帰の説明変数に設定された単語
    coef: 決定係数
    frequency_pos: ポジティブ（短文）に出現した単語と出現回数（dict） ※wakatiにて削除された単語は対象外
    frequency_neg: ネガティブ（短文）に出現した単語と出現回数（dict） ※wakatiにて削除された単語は対象外

 ## メソッド一覧
     nlplot_wordcloud: ワードクラウド（頻出単語の可視化）を表示します
        <argument>
        label: 'positive' or 'neutral' or 'negative'
        
     nlplot_conetwork: 共起ネットワーク（文字通しの結び付き）を表示します
        <argument>
        label: 'positive' or 'neutral' or 'negative'
        
     nlplot_bigram: 出現数の多い単語（unigram）を表示します
        <argument>
        label: 'positive' or 'neutral' or 'negative'
        top_n: int(上位いくつの単語を対象とするか）
        
     nlplot_unigram: 出現数が多く、また結びつきの強い2単語を表示します
        <argument>
        label: 'positive' or 'neutral' or 'negative'
        top_n: int(上位いくつの単語を対象とするか）
        
     coef_graph: ネガティブ、ポジティブ、それぞれ出現頻度多い単語が結果にどう影響しているかを表すグラフを表示します
     
     calc_coef: ロジステック回帰を用いた決定係数の算出を行います ※analyzeメソッドで同時に実行されます
        <argument>
        coef_vocabulary: list[str] 学習の対象とする単語 ※analyzeメソッド実行時に自動抽出されたリストを上書くようになります
        coef_num: int いくつの単語を対象とするか ※coef_vocabularyと同時に入力された場合、coef_vocabularyの単語数が優先されます
        
     word_search: 指定した単語が含まれるレビューが検索できます
        <argument>
        word:list[str]
        label: 'positive' or 'neutral' or 'negative' or 'all'
        output_num: int 検索件数(デフォルト:10)
        how: 'and' or 'or' アンド検索orオア検索(デフォルト:and)






