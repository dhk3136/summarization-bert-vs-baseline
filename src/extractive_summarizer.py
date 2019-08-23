# import nltk for text preprocessing, tokenizing
# import beautifulsoup to parse URL input option
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import bs4 as BeautifulSoup
import urllib.request


# produce a sample summary
def summarize_text(doc):
    return article_text


def d_table(article_text):

    # using nltk stopwords; sklearn stopwords also an option
    stop_words = set(stopwords.words("english"))
    
    # convert article words into tokens
    words = word_tokenize(article_text)
    
    # get root words
    word_stems = PorterStemmer()
    
    # make and add words to frequency dict
    frequent = dict()
    for word in words:
        word = stem.stem(word)
        if word in stop_words:
            continue
        if word in frequent:
            frequent[word] += 1
        else:
            frequent[word] = 1

    return frequent


# code for calculating word probabilities and which ones are included in summary
def sent_scores(sentences, frequent):   

    sent_probs = dict()

    # adjustable len()
    for s in sentences:
        s_count = (len(word_tokenize(s)))
        s_count_no_stop = 0
        for word_probs in frequent:
            if word_probs in s.lower():
                s_count_no_stop += 1
                if s[:9] in sent_probs:
                    sent_probs[s[:9]] += frequent[word_probs]
                else:
                    sent_probs[s[:9]] = frequent[word_probs]

        sent_probs[s[:9]] = sent_probs[s[:9]] / s_count_no_stop

    return sent_probs


# returns sentence averages
def sent_avg(sent_probs):
   
    sum = 0
    for line in sent_probs:
        sum += sent_probs[line]

    avg = (sum / len(sent_probs))

    return avg

# determining, counting sentences most likely to yield best words for summary
def construct_summary(sentences, sent_probs, threshold):
    counter = 0
    summary = ''
    
    for s in sentences:
        if s[:9] in sent_probs and sent_probs[s[:9]] >= (threshold):
            summary += " " + s
            counter += 1

    return summary

def gist(sentences, sent_probs_max = 0)
    for k, v in sent_probs.items():
        if 10 > s_max:
            s_max = v
            k_max = k
            
    for i, s in enumerate(sentences):
        beg = s[:9]
        if beg == k_max:
            ind_sent_max_s = i
            
    return sentences[ind_sent_max_s]


# the summary
# threshold parameter can be tuned to preference
def summarize(content):

    # frequency dict
    frequent = d_table(content)

    # tokenizing sentences
    sentences = sent_tokenize(content)

    # scoring words via sentences
    sentence_score = sent_scores(sentences, frequent)

    # establishing threshold
    threshold = sent_avg(sentence_score)
    
    final_summary = construct_summary(sentences, sentence_score, 1.5 * threshold)


if __name__ == '__main__':

    text = pd.read_csv('1st_1000_2_cols.csv', columns=['headline', 'text'])

    X = text['text']
    y = text['headline']
    
    doc = text['text']
    article_text = summarize_text(doc)
    content = X.iloc[0]
    reference_summary = y.iloc[0]
    system_summary = final_summary
    print(content)
    print(reference_summary)
    print(system_summary)

    