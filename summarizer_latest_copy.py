#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas


# In[ ]:


# code citation: https://blog.floydhub.com/gentle-introduction-to-text-summarization-in-machine-learning/

#importing libraries
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  
import pandas as pd
import numpy as np
import pdb

# #fetching the content from the URL
# fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/20th_century')


# article_read = fetched_data.read()

# #parsing the URL content and storing in a variable
# article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')

# #returning <p> tags
# paragraphs = article_parsed.find_all('p')

# article_content = ''

# #looping through the paragraphs and adding them to the variable
# for p in paragraphs:  
#     article_content += p.text

#text = pd.read_csv('../wikihow-dataset/wikihowSep.csv')
# text = text.iloc[:1000]



# It is clear that ROUGE-N is a recall-related measure because the denominator of the equation is the
# total sum of the number of n-grams occurring at the reference summary side.
# ie, Rouge-1 = matches / total num of words in reference/gold model

# iter_summaries = 0
# def loop_summaries(text_string, true_lst):
#     for word in text_string:
#         for chars in true_lst:
#             print(word, chars)
#             if word == chars:
#                 iter_summaries += 1
#     return iter_summaries / len(text_string)  # OR OTHER WAY AROUND
# #     return iter_summaries
     

# def loop(text_string):
#     for articles in text_string:

        
def _create_dictionary_table(text_string) -> dict:
    
    #removing stop words
    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(text_string)
    
    #reducing words to their root form
    stem = PorterStemmer()
    
    #creating dictionary for the word frequency table
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

    #algorithm for scoring a sentence by its words
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
   
    #calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    #getting sentence average value from source text
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

def _get_article_gist(sentences, sentence_weights):
    s_max = 0
    for k, v in sentence_weights.items():
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
    #creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article_content)

    #tokenizing the sentences
    sentences = sent_tokenize(article_content)

    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    #getting the threshold
    threshold = _calculate_average_score(sentence_scores)
    
    #producing the summary
#     article_summary = _get_article_summary(sentences, sentence_scores, 1.5 * threshold)
    article_summary = _get_article_gist(sentences, sentence_scores)

    return article_summary

    
#     def rouge_1(nums_of_overlap_words, nums_of_reference_words):
#         a = nums_of_overlap_words / nums_of_reference_words
#         for each in a:
#             b = rouge_1_list.append(each)
#             return b
        
        

if __name__ == '__main__':
    # text.to_csv('1st_1000_2_cols.csv', columns=['headline', 'text'], index=False)
    text = pd.read_csv('1st_1000_2_cols.csv')

    X = text['text']
    y = text['headline']
    
    text_string = X.iloc[0]
    
    article_content = X.iloc[0]
    true_summary = y.iloc[0]
    all_articles = X.iloc[0:]

    
    sentences = sent_tokenize(text_string)
    frequency_table = _create_dictionary_table(text_string)
    print(_calculate_sentence_scores(sentences, frequency_table))
    print(sentence_weight)
    print(_calculate_average_score(sentence_weight))
    
    arr = X.iloc[0:].tolist()
#     print(arr)
   # article_content = X.str.cat(sep='#')
  
    summary_results = _run_article_summary(article_content)
    print(article_content)
    print(summary_results)
    print(true_summary)

#     print(_run_article_summary(article_content))

    sum_lst = summary_results.split()
    true_lst = true_summary.split()
    print(f'\n{sum_lst}')
    print(f'{true_lst}\n')

    nums_of_overlap_words = len([word for word in sum_lst if word in true_lst])
    nums_of_reference_words = (len(true_lst))
    nums_of_system_words = len((sum_lst))
    
    rouge_recall = nums_of_overlap_words/nums_of_reference_words
    rouge_precision = nums_of_overlap_words / nums_of_system_words
    
    recall = []
    recall.append(rouge_recall)
    print(recall)

    print(f'Rouge Recall:\n{rouge_recall}\n\nRouge Precison:\n{rouge_precision}')
    
    print(_create_dictionary_table(text_string))


# In[ ]:





# In[ ]:




