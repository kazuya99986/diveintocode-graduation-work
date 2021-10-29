import pandas as pd
import numpy as np
import torch
import re
import unicodedata
import nlplot
import collections
import plotly.graph_objects as go

from fugashi import Tagger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, AdamW, Trainer, TrainingArguments



class Make_model:
    '''
    ファインチューニングモデルを作成
    Parameters
    ----------
    model_PATH: str
                使用する事前学種モデル
    '''
    def __init__(self, model_name):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name, tokenizer='mecab')


    def fit(self, texts, labels, max_length=128, epochs=5,train_batch_size=32, eval_batch_size=64):
        '''
        Parameters
        ----------
        texts       : list[str]
                    学習データ（文章）
        label       : list[int]
                    学習データ（ラベル）※Labelは 2 (Positive), 1 (Neutral), 0 (Negative)としてください
        max_length  : int
                    最大文字数
        epochs      : int
                    学習エポック数
        train_batch_size: int
                    学習データバッチ数
        eval_batch_size: int
                    検証データバッチ数
        '''
        train_docs, test_docs, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=0)

        train_encodings = self.tokenizer(train_docs,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length).to(self.device)

        test_encodings = self.tokenizer(test_docs,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length).to(self.device)

        train_dataset = JpSentiDataset(train_encodings, train_labels)
        test_dataset = JpSentiDataset(test_encodings, test_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_steps=500,
            weight_decay=0.05,
            save_total_limit=1,
            dataloader_pin_memory=False,
            evaluation_strategy="epoch",
            logging_steps=500,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        save_directory = './original_model'
        self.tokenizer.save_pretrained(save_directory)
        self.model.save_pretrained(save_directory)



    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }



class JpSentiDataset(torch.utils.data.Dataset):
    '''
    学習データセットのエンコーディング
    '''
    def __init__(self, encodings, labels=False):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.labels:
            return len(self.labels)
        else:
            return self.encodings["input_ids"].shape[0]




class Analyzer:
    '''
    Parameters
    ----------
    model_PATH: str
                使用するモデルがあるパスを指定
    stopwords : list(str), shape (1D)
                nlplotでの可視化の除外するキーワード
    wakati_ng_type: list(str), shape(1D)
                mecabでの分かち書きの際に除外する品詞種別（品詞名一覧：https://taku910.github.io/mecab/posid.html)
                デフォルト設定：['助詞','助動詞','接続詞','動詞','記号','EOS']　※EOSも含めてください
    coef_type : list(str), shape(1D)
                決定係数算出の際の説明変数
                デフォルト設定：['名詞-普通名詞']
    coef_num　: int
                決定係数算出の際、ネガティブ・ポジテイブそれぞれ上位何キーワードを説明変数に用いるか
                ※少ないサンプル数の際に、大きい値を設定すると結果が思わしくないものとなります（目安：サンプル数1000の際には、10程度）
                ※ネガティブ・ポジテイブそれぞれからキーワードを抽出し、重複排除を行う為、入力された値と実際出力される値は異なります
    coef_vocabulary: list(str), shape(1D)
                決定係数算出の際の説明変数（計算の際には出現数に置き換えられます）
                ※coef_vocabularyが設定された場合は、coef_numの設定は無視されます
    '''

    def __init__(self, model_PATH='base_model/', stopwords=False, wakati_ng_type=False, coef_type=False, coef_num=False, coef_vocabulary=False):

        model_PATH = model_PATH
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_PATH, local_files_only=True)
        self.model = BertForSequenceClassification.from_pretrained(model_PATH,).to(device)

        if not stopwords:
            self.stopwords = []
        else:
            self.stopwords = stopwords

        self.wakati_ng_type = wakati_ng_type
        self.coef_type = coef_type
        self.coef_num = coef_num
        self.coef_vocabulary = coef_vocabulary

        self.df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
        self.frequency_pos = {}
        self.frequency_neg = {}



    def lower_text(self, text):
        '''
        前処理
        大文字アルファベットを小文字へ変換
        Parameters
        ----------
        text     : str
                   変換する文字列
        '''
        return text.lower()


    def clean_url(self, text):
        '''
        前処理
        urlの正規化
        Parameters
        ----------
        text     : str
                   変換する文字列
        '''
        return re.sub(r'http\S+', '', text)


    def clean_symbol(self, text):
        '''
        前処理
        記号を削除
        Parameters
        ----------
        text     : str
                   変換する文字列
        '''
        symbols = '["#$%&\'(),-./:;<=>@\\]^_`{|}、]'
        return re.sub(symbols, '', text)


    def normalize_unicode(self, text, form='NFKC'):
        '''
        前処理
        Unicode正規化
        Parameters
        ----------
        text     : str
                   変換する文字列
        '''
        return unicodedata.normalize(form, text)


    def normalize_number(self, text):
        '''
        前処理
        連続する数字は0へ置換
        Parameters
        ----------
        text     : str
                   変換する文字列
        '''
        return re.sub(r'\d+', '0', text)


    def analyze(self, text_input, batch_size=32):
        '''
        メイン処理
        1. 前処理
        2. レビューを'。'などで区切る
        3. 区切り文での推論（感情極性ラベル（positive, negative, neutral）に分類）
        4. 総合スコアリング（レビュー全体の評価決定）
        Parameters
        ----------
        text_list   : list(str)
                      分析する文章（レビュー）
        '''
        texts = []
        text_num = []
        self.total_evaluation = []

        for i in range(len(text_input)):
            sentence = self.lower_text(text_input[i])
            sentence = self.clean_url(sentence)
            sentence = self.clean_symbol(sentence)
            sentence = self.normalize_number(sentence)
            sentence = self.normalize_unicode(sentence)

            sentence = re.split('[。!?]', sentence)
            for small_sentence in sentence:
                if small_sentence != '':
                    texts.append(small_sentence)
                    text_num.append(i)

        encoding = self.tokenizer(texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=128)

        text_encoded = JpSentiDataset(encoding)

        predict_args = TrainingArguments(
            output_dir='./results',
            do_predict=True,
            # per_device_train_batch_size=32,
            per_device_eval_batch_size=batch_size,
        )

        trainer = Trainer(model=self.model, args=predict_args)
        predictions = trainer.predict(text_encoded)

        labels = np.argmax(predictions.predictions, axis=1).tolist()

        self.df = pd.DataFrame({'text_num':text_num,'text':texts,'label':labels})

        self.sentence_length = text_num[-1]+1
        temp_eva = []
        for i in range(self.sentence_length):
            temp_df = self.df[self.df['text_num']==i]
            eva = 0
            for m, sentence_label in enumerate(temp_df['label']):
                if sentence_label == 2:
                    eva += 1
                elif sentence_label == 1:
                    eva += 0
                elif sentence_label == 0:
                    eva += -1

            if eva > 0:
                evaluation = 'positive'
            elif eva == 0:
                evaluation = 'neutral'
            elif eva < 0:
                evaluation = 'negative'


            self.total_evaluation.append(evaluation)
            [temp_eva.append(evaluation) for _ in range(m+1)]

        self.df = pd.concat([self.df, pd.DataFrame({'total_evaluation':temp_eva})], axis=1)


        self.fugashi_wakati()
        self.nlplot_graph()


        pos_num = self.total_evaluation.count('positive')
        neg_num = self.total_evaluation.count('negative')
        neu_num = self.total_evaluation.count('neutral')
        print()
        print('【総合評価】')
        print('ポジティブ: {}   {:.1f}%'.format(pos_num, pos_num*100 /self.sentence_length))
        print('ネガティブ: {}   {:.1f}%'.format(neg_num, neg_num*100 /self.sentence_length))
        print('ニュートラル: {}   {:.1f}%'.format(neu_num, neu_num*100 /self.sentence_length))
        print('【短文評価】')
        print('ポジティブ: {}   {:.1f}%'.format(self.df[self.df['label']==2].shape[0], self.df[self.df['label']==2].shape[0]*100 / len(self.df)))
        print('ネガティブ: {}   {:.1f}%'.format(self.df[self.df['label']==0].shape[0], self.df[self.df['label']==0].shape[0]*100 /len(self.df)))
        print('ニュートラル: {}   {:.1f}%'.format(self.df[self.df['label']==1].shape[0], self.df[self.df['label']==1].shape[0]*100 /len(self.df)))
        print('※短文とは、1レビューを\'。\'などで区切った文のことです')


        self.calc_coef()


        print()
        print('【ポジティブレビューの頻出単語】')
        print([sorted(self.frequency_pos.items(), key=lambda x:x[1], reverse=True)[i][0] for i in range(20)])
        print()
        print('【ネガティブレビューの頻出単語】')
        print([sorted(self.frequency_neg.items(), key=lambda x:x[1], reverse=True)[i][0] for i in range(20)])





    def fugashi_wakati(self):
        '''
        fugashiを用いたわかち書き
        以下2通りでわかち書き
        1. '助詞' '助動詞' '接続詞' '副詞' '動詞' '記号'を除いたもの・・・self.dfへ追加
        2. '名詞-普通名詞'を含むもの・・・出現回数と対にして、dictへ保存
        '''
        if not self.wakati_ng_type:
            self.wakati_ng_type = ['助詞','助動詞','動詞','記号','接頭辞','接続詞','代名詞','連体詞']

        if not self.coef_type:
            self.coef_type = '普通名詞'

        tagger = Tagger('-Owakati')
        wakati_list = []

        for i,text in enumerate(self.df['text']):
            temp = []
            for x in tagger(text):
                for m,ng_type in enumerate(self.wakati_ng_type):
                    if x.pos.split(',')[0] == ng_type:
                        break
                    if m == len(self.wakati_ng_type) -1:
                        temp.append(str(x))

                if self.df['label'][i] == 2:
                    if x.pos.split(',')[1] == self.coef_type:
                        if str(x) == self.stopwords:
                            continue
                        elif str(x) not in self.frequency_pos.keys():
                            self.frequency_pos[str(x)] = 1
                        else:
                            self.frequency_pos[str(x)] += 1

                elif self.df['label'][i] == 0:
                    if x.pos.split(',')[1] == self.coef_type:
                        if str(x) == self.stopwords:
                            continue
                        elif str(x) not in self.frequency_neg.keys():
                            self.frequency_neg[str(x)] = 1
                        else:
                            self.frequency_neg[str(x)] += 1

            wakati_list.append(temp)
        self.df = pd.concat([self.df, pd.DataFrame({'wakati':wakati_list})], axis=1)



    def calc_coef(self, coef_vocabulary=False, coef_num=False):
        '''
        ロジスティック回帰の決定係数を用いて、結果に対する主要キーワードの寄与度を算出
        Parameters
        ----------
        coef_num　: int
                    決定係数算出の際、ネガティブ・ポジテイブそれぞれ上位何キーワードを説明変数に用いるか
                    ※少ないサンプル数の際に、大きい値を設定すると結果が思わしくないものとなります（目安：サンプル数1000の際には、10程度）
                    ※ネガティブ・ポジテイブそれぞれからキーワードを抽出し、重複排除を行う為、入力された値と実際出力される値は異なります
        coef_vocabulary: list(str), shape(1D)
                    決定係数算出の際の説明変数（計算の際には出現数に置き換えられます）
                    ※coef_vocabularyが設定された場合は、coef_numの設定は無視されます
        '''
        if coef_num:
            self.coef_num = coef_num

        if not self.coef_num:
            self.coef_num = int(3 + self.sentence_length * 0.003)

        if coef_vocabulary:
            self.coef_vocabulary = coef_vocabulary

        if not self.coef_vocabulary:
            pos_top_keys = [sorted(self.frequency_pos.items(), key=lambda x:x[1], reverse=True)[i][0] for i in range(self.coef_num)]
            neg_top_keys = [sorted(self.frequency_neg.items(), key=lambda x:x[1], reverse=True)[i][0] for i in range(self.coef_num)]

            self.coef_vocabulary = list(set(pos_top_keys + neg_top_keys))

        pos_array = np.zeros((self.sentence_length,len(self.coef_vocabulary)))
        neg_array = np.zeros((self.sentence_length,len(self.coef_vocabulary)))

        for i in range(self.sentence_length):
            temp_df = self.df[self.df['text_num']==i]
            for m in range(len(temp_df)):
                if temp_df.iloc[m,2] == 2:
                    for j in range(len(self.coef_vocabulary)):
                        if self.coef_vocabulary[j] in temp_df.iloc[m,4]:
                            pos_array[i,j] += 1

                elif temp_df.iloc[m,2] == 0:
                    for j in range(len(self.coef_vocabulary)):
                        if self.coef_vocabulary[j] in temp_df.iloc[m,4]:
                            neg_array[i,j] += 1

        df_pos_neg = pd.concat([pd.DataFrame(pos_array),pd.DataFrame(neg_array), pd.DataFrame(self.total_evaluation, columns=['total_evaluation'])],axis=1)

        most_common_label = collections.Counter(self.total_evaluation).most_common()[0][0]
        if most_common_label == 'neutral':
            most_common_label = collections.Counter(self.total_evaluation).most_common()[1][0]

        if most_common_label == 'positive':
            df_neg = df_pos_neg[df_pos_neg['total_evaluation'] == 'negative']
            df_pos = df_pos_neg[df_pos_neg['total_evaluation'] == 'positive'].sample(len(df_neg))

        elif most_common_label == 'negative':
            df_pos = df_pos_neg[df_pos_neg['total_evaluation'] == 'positive']
            df_neg = df_pos_neg[df_pos_neg['total_evaluation'] == 'negative'].sample(len(df_pos))

        self.df_pos_neg = pd.concat([df_pos,df_neg])
        y = np.where(self.df_pos_neg['total_evaluation'] == 'positive', 1, 0)
        X = self.df_pos_neg.drop('total_evaluation', axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

        clf = LogisticRegression(random_state=0)
        clf.fit(X_train, y_train)
        self.coef = clf.coef_

        Y_pred = clf.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, Y_pred, average='binary')

        print()
        print('【ロジスティック回帰】')
        print('accuracy_score:  {:.1f}%'.format(accuracy_score(y_test, Y_pred)*100))
        print('precision_score:  {:.1f}%'.format(precision*100))
        print('recall_score:  {:.1f}%'.format(recall*100))
        print('f1:  {:.1f}%'.format(f1*100))
        print('※coef_graph()メソッドで決定係数グラフ参照可')




    def nlplot_graph(self):
        '''
        nlplotインスタンス化、グラフ作成
        '''
        positive = self.df[self.df['label']==2]
        neutral = self.df[self.df['label']==1]
        negative = self.df[self.df['label']==0]

        if len(positive) > 0:
            self.nlp_pos = nlplot.NLPlot(positive, target_col='wakati')
            self.nlp_pos.build_graph(stopwords=self.stopwords, min_edge_frequency= self.clac_min_edge_frequency(positive) )

        if len(neutral) > 0:
            self.nlp_neu = nlplot.NLPlot(neutral, target_col='wakati')
            self.nlp_neu.build_graph(stopwords=self.stopwords, min_edge_frequency= self.clac_min_edge_frequency(neutral) )

        if len(negative) > 0:
            self.nlp_neg = nlplot.NLPlot(negative, target_col='wakati')
            self.nlp_neg.build_graph(stopwords=self.stopwords, min_edge_frequency= self.clac_min_edge_frequency(negative) )




    def clac_min_edge_frequency(self, label):
        '''
        nlplotのmin_edge_frequencyを算出
        Parameters
        ----------
        label     : DataFrame
                    ラベルを条件に抽出されたデータフレーム
        '''
        if len(label) < 100:
            return 1
        elif len(label) < 500:
            return 2
        else:
            return int(10 + len(label) * 0.003)



    def nlplot_wordcloud(self, label):
        '''
        nlplot、ワードクラウド作成
        '''
        if label == 'positive':
            pram = self.nlp_pos
        elif label == 'neutral':
            pram = self.nlp_neu
        elif label == 'negative':
            pram = self.nlp_neg
        else:
            raise ValueError(
            f'そのlabelは存在しません。labelは\'positive\', \'neutral\',\'negative\'のいずれかを選択してください'
            )

        pram.wordcloud(
            max_words=100,
            max_font_size=100,
            colormap='tab20_r',
            stopwords=self.stopwords,
            )



    def nlplot_conetwork(self, label):
        '''
        nlplot、共起ネットワーク作成
        Parameters
        ----------
        label     : str('positive' or 'neutral' or 'negative')
                    表示したいラベル
        '''
        if label == 'positive':
            pram = self.nlp_pos
        elif label == 'neutral':
            pram = self.nlp_neu
        elif label == 'negative':
            pram = self.nlp_neg
        else:
            raise ValueError(
            f'そのlabelは存在しません。labelは\'positive\', \'neutral\',\'negative\'のいずれかを選択してください'
            )

        pram.co_network(
            title=f'Co-occurrence network of {label}s',
            sizing = 100, #default:100,
        )



    def nlplot_unigram(self, label, top_n=50):
        '''
        nlplot、uniqramのグラフ作成
        Parameters
        ----------
        label     : str('positive' or 'neutral' or 'negative')
                    表示したいラベル
        top_n     : int
                    表示したい件数
        '''
        if label == 'positive':
            pram = self.nlp_pos
        elif label == 'neutral':
            pram = self.nlp_neu
        elif label == 'negative':
            pram = self.nlp_neg
        else:
            raise ValueError(
            f'そのlabelは存在しません。labelは\'positive\', \'neutral\',\'negative\'のいずれかを選択してください'
            )

        return pram.bar_ngram(
            title=f'uni-gram of {label}s Top50',
            xaxis_label='word_count',
            yaxis_label='word',
            ngram=1,
            top_n=top_n,
            stopwords=self.stopwords,
        )



    def nlplot_bigram(self, label, top_n=50):
        '''
        nlplot、biqramのグラフ作成
        Parameters
        ----------
        label     : str('positive' or 'neutral' or 'negative')
                    表示したいラベル
        top_n     : int
                    表示したい件数
        '''
        if label == 'positive':
            pram = self.nlp_pos
        elif label == 'neutral':
            pram = self.nlp_neu
        elif label == 'negative':
            pram = self.nlp_neg
        else:
            raise ValueError(
            f'そのlabelは存在しません。labelは\'positive\', \'neutral\',\'negative\'のいずれかを選択してください'
            )

        pram.bar_ngram(
            title=f'bi-gram of {label}s Top50',
            xaxis_label='word_count',
            yaxis_label='word',
            ngram=2,
            top_n=top_n,
            stopwords=self.stopwords,
        )



    def coef_graph(self):
        '''
        ロジスティック回帰、決定係数のグラフ作成
        '''
        x = self.coef_vocabulary
        y = self.coef[0]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y[:int(len(y)/2)],
                        marker_color='#5256EF',
                        name='Positive'))
        fig.add_trace(go.Bar(x=x, y=y[int(len(y)/2):],
                        marker_color='crimson',
                        name='Negative'
                        ))
        fig.show()




    def word_search(self, word, label='all', output_num=10, how='and'):
        '''
        キーワード検索
        Parameters
        ----------
        word        : str ※複数の場合は要素内でカンマ区切り
                      list
                    検索したい文字列
        label       : str('all' or 'positive' or 'neutral' or 'negative')
                    検索したいラベル
        output_num  : int
                    表示したい件数
        how         : str（'and' or 'or')
                    検索タイプ（and検索　or or検索）
        '''
        if type(word) is str:
            word = list(word.split(','))
        elif type(word) is not list:
            return 'wordはlist、もしくはstr（複数ワード検索の場合は要素内でカンマ区切り）型で入力してください'

        if label == 'positive':
            label_onlist = 2
        elif label == 'neutral':
            label_onlist = 1
        elif label == 'negative':
            label_onlist = 0
        else:
            label_onlist = False

        if label_onlist:
            texts = self.df[self.df['label'] == label_onlist]['text'].sample(frac=1)
        else:
            texts = self.df['text'].sample(frac=1)

        text_list = []


        if how == 'and':
            for text in texts:
                if all(map(text.__contains__, (word))):
                    text_list.append(text)
                    if len(text_list) == output_num:
                        return text_list

        if how == 'or':
            for text in texts:
                if any(map(text.__contains__, (word))):
                    text_list.append(text)
                    if len(text_list) == output_num:
                        return text_list

        if text_list == []:
            raise ValueError(
            f'そのキーワードが含まれる文章は存在しません'
            )

        return text_list
