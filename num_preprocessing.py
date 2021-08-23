import pandas as pd
import spacy
import os

class Num_Preprocessor():

    def extract_date(self,sent,nlp):
        doc = nlp(sent)
        ents = [e.text for e in doc.ents if e.label_=='DATE' or e.label_=='TIME']
        extraced_date_sent = sent
        for ent in ents:
            extraced_date_sent = extraced_date_sent.replace(ent,'')
        return extraced_date_sent

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

    def check_non_num_chinese(self,phrase,remove_phrases=['一些','多少','大多數','多',]):
        result_phrase = phrase
        for remove_phrase in remove_phrases:
            if result_phrase == remove_phrase:
                return ''
        return result_phrase

    def add_num_phrase_procedure(self,num_phrase, num_count, num_dict,non_needed_phrase = ['','一','1','one']):
        if num_phrase != '' and num_phrase != '一' and num_phrase != '1' and num_phrase != 'one':
            num_dict[f'<NUM{num_count}>'] = num_phrase
            num_phrase = ''
            num_count += 1
        is_phrase = False
        return num_phrase,num_count,num_dict,is_phrase

    def extract_num(self,sent, nlp, split_sign =' ', remove_tokens = ['元','%']):
        extracted_sent = self.extract_date(sent,nlp)
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

        return num_dict

    def replace_num(self,sent,num_dict):
        result_sent = sent
        for key in num_dict:
            result_sent = result_sent.replace(num_dict[key],key,1)
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
            en_corpus.append(en_data.sentip())
            zh_corpus.append(zh_data.sentip())
    df = pd.DataFrame({'english_sent':en_corpus,'chinese_sent':zh_corpus})
    
    # initialize chinese and english spacy nlp model.
    zh_nlp = spacy.load('zh_core_web_trf')
    en_nlp = spacy.load('en_core_web_trf')
    num_preprocessor = Num_Preprocessor()
    sent = 'On May 27, 2019, it will release NTD 50 million (73%) and transfer it to the account of the same name in Taipei Fubon Bank for operational turnover.'
    en_num_dict = num_preprocessor.extract_num(sent, en_nlp)
    print(num_preprocessor.replace_num(sent, en_num_dict))
    print(en_num_dict)
    
    sent = '2019年5月27日放款新台幣5千萬元佔73%，轉入台北富邦銀行同名賬戶，用於業務周轉。'
    doc = zh_nlp(sent)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)
    print(sent)
    zh_num_dict = num_preprocessor.extract_num(sent, zh_nlp, split_sign='')
    numed_zh_sent = num_preprocessor.replace_num(sent, zh_num_dict)
    print(numed_zh_sent)
    print(zh_num_dict)
    print(num_preprocessor.fill_num(numed_zh_sent, en_num_dict))
    count = 0
    for en_sen,zh_sen in zip(df['english_sent'],df['chinese_sent']):
        zh_num_dict = num_preprocessor.extract_num(zh_sen, zh_nlp, split_sign='')
        en_num_dict = num_preprocessor.extract_num(en_sen,en_nlp,split_sign=' ')
        numed_zh_sent = num_preprocessor.replace_num(zh_sen, zh_num_dict)
        numed_en_sent = num_preprocessor.replace_num(en_sen,en_num_dict)
        if len(zh_num_dict)!=0 |len(en_num_dict)!=0:
            doc = zh_nlp(zh_sen)
            for token in doc:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop)
            doc = en_nlp(en_sen)
            for token in doc:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                      token.shape_, token.is_alpha, token.is_stop)
    
            print(zh_num_dict,en_num_dict)
            print(numed_zh_sent)
            print(numed_en_sent)
            count +=1
            if count > 1:
                break
    
    sent = '首爾房價一年上漲18%'
    zh_num_dict = num_preprocessor.extract_num(sent, zh_nlp, split_sign='')
    numed_zh_sent = num_preprocessor.replace_num(sent, zh_num_dict)
    doc = zh_nlp(sent)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print(numed_zh_sent)