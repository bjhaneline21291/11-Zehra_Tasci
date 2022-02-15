//        DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
//                    Version 2, December 2004 
//
// Copyright (C) 2004 YC <github.com/YC-Lammy> 
//
// Everyone is permitted to copy and distribute verbatim or modified 
// copies of this license document, and changing it is allowed as long 
// as the name is changed. 
//
//            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
//   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 
//
//  0. You just DO WHAT THE FUCK YOU WANT TO.


var extract_keyword = function(text, number = 5) {
    const stopwords = ['what', 'until', 'a', "haven't", 'will', 'wasn', 'hers', 'after', 'myself', 'below', 'mustn', 'she', 'here', 'if', 'these', 'only', 'above', "wouldn't", "mustn't", "it's", "mightn't", 'to', 'was', 'in', "won't", 'itself', 'can', 'mightn', 'ourselves', 'be', 'which', 'some', 'those', "should've", "you'd", 'again', 've', 'haven', 'not', 'hadn', 'shouldn', 'my', 'has', 'are', 'further', "she's", 'now', "wasn't", 'themselves', 'its', "shan't", 'ain', 'an', "isn't", 'there', 'your', 'doing', 'once', 'that', "you've", 'no', 'were', 'just', 'them', 'her', 'between', 'been', 'into', 'from', 'while', 'on', 'against', 'am', 'out', 'should', 'this', 'then', 'as', 'm', 'being', 'doesn', "hadn't", 'before', 'who', 'with', 'down', 'nor', 'is', 'me', 'or', 'wouldn', 'own', "hasn't", 'did', "weren't", 'his', 'hasn', 'isn', "aren't", 'for', 'during', 'ours', 're', 'o', "needn't", 'up', 'under', 'each', 'have', 'same', 'off', 'where', 'but', 'ma', 'most', 'y', 'such', 'by', 'they', "doesn't", 'few', 'him', "you're", 's', 'very', 'll', 'don', 'than', 'when', 'd', 'through', 'having', 'it', 'weren', 'too', "didn't", 'their', 'you', 'and', 'himself', 'yours', 'other', 'so', 'more', 't', 'all', 'herself', 'the', 'theirs', 'aren', 'whom', 'about', 'won', 'yourselves', 'our', "don't", 'over', 'shan', 'we', 'why', "shouldn't", 'because', 'any', 'how', 'had', 'at', 'he', 'of', 'yourself', 'does', 'both', 'didn', "couldn't", "that'll", 'couldn', "you'll", 'i', 'needn', 'do'];

    var total_words = text.split(' ');
    const total_word_length = total_words.length;
    var total_sentences = text.split('.');
    for (i = 0; i < total_sentences.length; i++) {
        total_sentences[i] += '.';
    }

    const total_sent_len = total_sentences.length;
    var final = [];

    var check_sent = function(word, sentences) {

        for (x in sentences) {
            x = sentences[x];
            var y = true;
            if (x.includes(word) === false) {
                y = false;
                break;
            }
            final.push(y);
        }
        var num = 0;
        for (i = 0; i < final.length; i++) {
            if (final[i] === true) {
                num += 1;
            }
        }
        return num;
    }

    var idf_score = {};
    var tf_score = {};

    for (each_word in total_words) {
        each_word = total_words[each_word];
        each_word = each_word.replace('.', '');

        if (stopwords.includes(each_word) === false) {

            if (each_word in tf_score) {
                tf_score[each_word] += 1;
            } else {
                tf_score[each_word] = 1;
            }

            if (each_word in idf_score) {
                idf_score[each_word] = check_sent(each_word, total_sentences);
            } else {
                idf_score[each_word] = 1;
            }
        }
    }
    for (key in tf_score) {
        var value = tf_score[key];
        tf_score[key] = value / total_word_length;
    }
    for (key in idf_score) {
        var value = idf_score[key];
        idf_score[key] = Math.log(parseInt(total_sent_len / value));
    }

    var tf_idf_score = {};
    for (key in tf_score) {
        if (idf_score[key]) {
            tf_idf_score[key] = idf_score[key] * tf_score[key];
        } else {
            tf_idf_score[key] = 0;
        }
    }
    var get_top_n = function(dict_elem, number) {
        var items = Object.keys(dict_elem).map(function(key) {
            return [key, dict_elem[key]];
        });

        // Sort the array based on the second element
        items.sort(function(first, second) {
            return second[1] - first[1];
        });

        // Create a new array with only the first 5 items
        var result = items.slice(0, number);
        return result;
    }
    var result = get_top_n(tf_idf_score, number);
    console.log(result);
    return result;
}