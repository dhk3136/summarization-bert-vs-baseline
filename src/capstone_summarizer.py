# here come the libraries
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  
import pandas as pd
import numpy as np
import pdb


# feeding text, cleaning text, placing in dict
def _create_dictionary_table(text_string) -> dict:
    
    # removing stop words (NLTK version)
    stop_words = set(stopwords.words("english"))
    
    # splitting words into tokens
    words = word_tokenize(text_string)
    
    # converting to word roots
    stem = PorterStemmer()
            
    # new dict for word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table


def _calculate_sentence_scores(sentences, frequency_table) -> dict:   

    # the heart of the word frequency algorithm
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]

        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words

    return sentence_weight


def _calculate_average_score(sentence_weight) -> int:
   
    # average score for sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    # retrieving sentence avg from original text
    average_score = (sum_values / len(sentence_weight))

    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''
    
    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary

def _get_article_gist(sentences, sentence_weight):
    s_max = 0
    for k, v in sentence_weight.items():
        if v > s_max:
            s_max = v
            k_max = k
            
    for i, sentence in enumerate(sentences):
        beg = sentence[:7]
        if beg == k_max:
            ind_sent_max_s = i
            
    return sentences[ind_sent_max_s]


def _run_article_summary(article_content):
#     pdb.set_trace()
    # constructing dict for word frequency table
    frequency_table = _create_dictionary_table(article_content)

    # tokenizing
    sentences = sent_tokenize(article_content)

    # algorithm scoring for sentences by word
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    # optional parameter
    threshold = _calculate_average_score(sentence_scores)
    
    # generating summaries
    article_summary = _get_article_summary(sentences, sentence_scores, 1.5 * threshold)
    article_summary = _get_article_gist(sentences, sentence_scores)

    return article_summary



if __name__ == '__main__':
    # loading dataset
    text = pd.read_csv('data/wiki_corpus.csv')

    # separating summary generation from summary target
    X = text['text']
    y = text['headline']
    
    text_string = X
    text_string = [each for each in X]
    print(text_string)
    