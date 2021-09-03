import pandas as pd
import spacy
import os

class Num_Preprocessor():

    def extract_date(self,sent,nlp):
        doc = nlp(sent)
        ents = [(e.text,e.start_char,e.start_char+len(e.text)) for e in doc.ents if e.label_=='DATE' or e.label_=='TIME']
        extraced_date_sent = sent
        for ent in ents:
            extraced_date_sent = extraced_date_sent.replace(ent[0],'')
        return extraced_date_sent,ents

    def check_phrase(self,phrase,remove_tokens):
        result_phrase = phrase
        for remove_token in remove_tokens:
            if phrase[-1] == remove_token:
                result_phrase = result_phrase[0:-1]
                break
        return result_phrase

    def check_unit(self,phrase,non_unit=['千','萬','億']):
        result_phrase = ''
        for word in phrase:
            if word in non_unit:
                result_phrase += word
        return result_phrase

    def check_non_num_chinese(self,phrase,remove_phrases=['一些','多少','大多','數','大多數','多','多數','眾多',
                                                          '一半','數百萬','數千萬','數以萬計','大量','許','諸多','很多','幾年','若干','半數','幾','幾票',
                                                          '部分','部份','兩半','許多','同一','一點','一家','一大堆','大部分','約半','組數十','好幾部','著好幾','多年','一大','一起','整','整數'],num_unit = ['十','千','萬','億']):
        result_phrase = phrase
        for remove_phrase in remove_phrases:
            if result_phrase == remove_phrase:
                return ''
        return result_phrase


    def add_num_phrase_procedure(self,num_phrase, num_count, num_dict,non_needed_phrase = ['','一','one','一些','多少','大多','數','大多數','多','多數','眾多',
                                                          '一半','數百萬','數千萬','數以萬計','大量','許','諸多','很多','幾年','若干','半數','幾','幾票',
                                                          '部分','部份','兩半','許多','同一','一點','一家','一大堆','大部分','約半','組數十','好幾部','著好幾','多年','一大','一起','整','整數']):
        if num_phrase not in non_needed_phrase:
            num_dict[f'<NUM{num_count}>'] = num_phrase
            num_phrase = ''
            num_count += 1
        is_phrase = False
        return num_phrase,num_count,num_dict,is_phrase

    def extract_num(self,sent, nlp, split_sign =' ', remove_tokens = ['元','%']):
        extracted_sent,date_ents = self.extract_date(sent,nlp)
        doc = nlp(extracted_sent)
        num_count = 0
        is_phrase = False
        num_phrase = ''
        num_dict = dict()
        for token in doc:
            if token.pos_== 'NUM' and token.tag_!='OD':
                if is_phrase:
                    if token.tag_ == 'M':
                        #'M' means a unit, we need to check is there a number are tokenized as unit ex. 萬元 the model may see 萬元 as a kind of dollar.
                        num_phrase += split_sign + self.check_unit(token.text)
                        num_phrase,num_count,num_dict,is_phrase = self.add_num_phrase_procedure(num_phrase,num_count,num_dict)
                    else:
                        num_phrase += split_sign + self.check_non_num_chinese(token.text)
                else:
                    if token.tag_=='CD':
                        #'CD' means a number, but we need to check model tokenize adj of number as cd. ex. many 多少 一些 大部分 
                        is_phrase = True
                        num_phrase = self.check_non_num_chinese(token.text)
                        if num_phrase == '':
                            is_phrase = False
            else:
                if is_phrase:
                    num_phrase = self.check_phrase(num_phrase,remove_tokens)
                    num_phrase, num_count, num_dict, is_phrase = self.add_num_phrase_procedure(num_phrase, num_count, num_dict)

        if is_phrase:
        # the last phrase isn't added into phrase_dict
            num_phrase = self.check_phrase(num_phrase, remove_tokens)
            num_phrase, num_count, num_dict, is_phrase = self.add_num_phrase_procedure(num_phrase, num_count, num_dict)

        return num_dict,date_ents

    def replace_num(self,sent,num_dict,ents):
        result_sent = sent
        for i in range(len(ents)-1,-1,-1):
            ent_start = ents[i][1]
            ent_end = ents[i][2]
            result_sent = result_sent[:ent_start] + '<DATE>' + result_sent[ent_end:]
        for key in num_dict:
            result_sent = result_sent.replace(num_dict[key],key,1)
        for ent in ents:
            result_sent = result_sent.replace('<DATE>',ent[0],1)
        return result_sent

    def fill_num(self,sent,num_dict):
        resulted_sent = sent
        for key in num_dict:
            resulted_sent = resulted_sent.replace(key,num_dict[key],1)
        return resulted_sent

    def show_tokens_pos(self,sent,nlp):
        doc = nlp(sent)
        for token in doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                  token.shape_, token.is_alpha, token.is_stop)

    def show_tokens_ents(self,sent,nlp):
        doc = nlp(sent)
        ents=[(ent.text, ent.label_)for ent in doc.ents]
        print(ents)


def english_example():
    sent = '2019年5月27日放款新台幣5千萬元佔73%，轉入台北富邦銀行同名賬戶，用於業務周轉。'
    doc = zh_nlp(sent)
    for token in doc:
        print(token.idx)
        print(dir(token))
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)
    print(sent)
    zh_num_dict = num_preprocessor.extract_num(sent, zh_nlp, split_sign='')
    numed_zh_sent = num_preprocessor.replace_num(sent, zh_num_dict)
    print(numed_zh_sent)
    print(zh_num_dict)
    # print(num_preprocessor.fill_num(numed_zh_sent, en_num_dict))

def chinese_example():
    #那名失蹤近<NUM0>的小女孩，如今推斷應該已不在人世。
    #該鎮所有人已被困在家一個月。
    #所有離海岸不足100米的房子都有被洪水襲擊的危險。
    #被「男性健康」雜誌形容為「平均每戶學位數量多過溫度計上的刻度」的這座美國中西部大都會，擠掉德州的普蘭諾與北卡羅萊納州的萊里，在該雜誌的人口教育程度最佳的全<NUM0>大城市排名中，拿下第一名。
    #法國最近終止了Minitel服務，這種由法國自行發展的網路前身早在網際網路出現10年前，就為數 < NUM0 > 國民眾提供了線上金融、旅遊訂位，甚至性愛聊天室等服務。
    #「以前一個客人上門，我們總是能夠賣<NUM0>、<NUM1>樣產品給他們。現在，唉，他們通常就只會買一樣。我們確實感受經濟不景氣帶來的衝擊。」他說。
    sent = '當電力公司打開球中成千上萬的小燈時，人們恐中喊著“喔”和“啊”．'
    #sent = '最大面額1000先令的新版紙鈔在六月推出，這項行動旨在迫使逃稅者、奸商和犯罪集團，釋出積存的來歷不明財富。肯亞央行六月表示，約有2.18億張千元先令在市面流通。'
    zh_num_dict, zh_date_ents = num_preprocessor.extract_num(sent, zh_nlp, split_sign='')
    numed_zh_sent = num_preprocessor.replace_num(sent, zh_num_dict,zh_date_ents)
    doc = zh_nlp(sent)
    print(sent)
    num_preprocessor.show_tokens_pos(sent,zh_nlp)
    print(zh_num_dict,zh_date_ents)
    print(numed_zh_sent)


if __name__=='__main__':
    # load long and short corpus
    training_data_path = 'training_data/'
    en_files = ['financial-long.src','financial-short.src']
    zh_files = ['financial-long.trg','financial-short.trg']
    en_corpus = []
    zh_corpus = []
    
    for en_file, zh_file in zip(en_files, zh_files):
        en_fp = open(os.path.join(training_data_path,en_file))
        zh_fp = open(os.path.join(training_data_path, zh_file))
    
        for en_data, zh_data in zip(en_fp,zh_fp):
            en_corpus.append(en_data.strip())
            zh_corpus.append(zh_data.strip())
    df = pd.DataFrame({'english_sent':en_corpus,'chinese_sent':zh_corpus})

    # initialize chinese and english spacy nlp model.
    zh_nlp = spacy.load('zh_core_web_trf')
    en_nlp = spacy.load('en_core_web_trf')

    num_preprocessor = Num_Preprocessor()
    numed_zh_sents = []
    zh_num_dicts = []
    count = 0
    print(df.shape[0])

    for en_sen,zh_sen in zip(df['english_sent'],df['chinese_sent']):
        zh_num_dict, zh_date_ents = num_preprocessor.extract_num(zh_sen, zh_nlp, split_sign='')
        en_num_dict, en_date_ents = num_preprocessor.extract_num(en_sen,en_nlp,split_sign=' ')
        numed_zh_sent = num_preprocessor.replace_num(zh_sen, zh_num_dict,zh_date_ents)
        numed_en_sent = num_preprocessor.replace_num(en_sen,en_num_dict,en_date_ents)
        numed_zh_sents.append(numed_zh_sent)
        zh_num_dicts.append(zh_num_dict)
        count +=1
        if count % 100 ==0:
            print(count)
    df['numed_chinese_sent'] = numed_zh_sents
    df['zh_num_dict'] = zh_num_dicts
    df.to_csv('training_data/numed_training_data.csv',index=False)



