"""        DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                    Version 2, December 2004 

 Copyright (C) 2004 YC <github.com/YC-Lammy> 

 Everyone is permitted to copy and distribute verbatim or modified 
 copies of this license document, and changing it is allowed as long 
 as the name is changed. 

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

  0. You just DO WHAT THE FUCK YOU WANT TO.
"""
import math
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import download
download('stopwords')
download('wordnet')
from operator import itemgetter

def extract_keyword(text, number=5):
    stop_words = set(stopwords.words('english'))
    total_words = text.split()
    total_word_length = len(total_words)
    total_sentences = tokenize.sent_tokenize(text)
    total_sent_len = len(total_sentences)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
    #print(tf_score)

    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())

    #print(idf_score)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    #print(tf_idf_score)

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    result = get_top_n(tf_idf_score, number)
    print(result)
    return result
