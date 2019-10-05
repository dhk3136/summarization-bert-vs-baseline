import numpy as np
from summarizer import SingleModel
import bs4 as bs  
import urllib.request  
import re
import argparse
import nltk
import logging
import heapq


# choose BERT or vanilla summarizer
choose_bert_or_vanilla = input('Please enter 1 to use the BERT summarizer or 2 for the Vanilla summarizer:\n')

# BERT summarizer
if choose_bert_or_vanilla == '1':
    print('Welcome to the BERT Summarizer!\n')
    
    # choose URL, text file, or copy/paste input
    choose_summarizer = input('Enter A for URL input:\nEnter B for text file input:\nEnter C for copy/paste input:\n(case-sensitive)\n')

    if choose_summarizer == 'A':
        print('\nOption A processes and summarizes web articles.\nBest results are obtained by entering a URL of a text-centric website (e.g., Wikipedia, news sites, blogs.)\n')
        print("NB: A good rule of thumb... if your browser can display the webpage via its 'Reader View,' it's likely the summarizer is compatible with the site.")
        print("If not, feel free to start over and select one of the other two summary options; it's just as simple to use the Text Document Summarizer or the Copy/Paste Summarizer.\n")
        print("Either enter the URL of the page you'd like to summarize or feel free to copy/paste one of the URLs below to take the summarizer out for a spin:")
        print('https://en.wikipedia.org/wiki/Chernobyl_(miniseries)\nhttps://www.blog.google/products/assistant/more-information-about-our-processes-safeguard-speech-data\n')
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        # taking input from website article
        url = input('URL to summarize:\n')
        print('Summarizing...')

        # fetching and reading in data from URL
        scraped_data = urllib.request.urlopen(url)  
        article = scraped_data.read()

        # using beautifulsoup to parse article
        parsed_article = bs.BeautifulSoup(article,'lxml')
        paragraphs = parsed_article.find_all('p')

        # iterating and appending text to string
        article_text = ""

        for p in paragraphs:  
            article_text += p.text

        # model default params
        # use bert-base-uncased for smaller, less resource-intensive model
        model = SingleModel(
            model='bert-large-uncased',
            vector_size=None,
            hidden=-2,
            reduce_option='mean'
    )

        # passing in full text to model
        m = model(article_text)

        # creating final summary with a ratio of 0.13
        summary_file = '\n\nSUMMARY:\n' + m

        # printing summary and full text for comparison
        print(f'\nSUMMARY:\n{model(article_text)}\n')
        print(f'FULL TEXT:\n', article_text)
        
        # appending summary output to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)


    # text file summary option
    elif choose_summarizer == 'B':
        print('\nOption B summarizes text documents from Google Drive or your local machine.\n')
        print("\nSimply provide the path to the text document you'd like to summarize.\nOr feel free to copy/paste the filename of one of the sample documents below:\n")
        print('full_text/google_pixel_photobooth_article.txt')
        print('full_text_samples/wash_post_youth_trending_away_from_news.txt\n')
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        document = input('Enter your <path/to/file.txt> here:\n')

        # reading in text file
        with open(document, 'r') as d:
            text_data = d.read()

        # importing model and passing in full text
        model = SingleModel()
        m = model(text_data)

        # creating final summary with a ratio of 0.13
        summary_file = '\n\nSUMMARY:\n' + m

        # printing summary and full text output for comparison
        print(f'\nSUMMARY:\n{model(text_data)}\n')
        print(f'FULL TEXT:\n', text_data)

        # appending summary output to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)


    # copy/paste string input option
    elif choose_summarizer == 'C':
        print("\nFor Option C, simply copy/paste any text you'd like to summarize below.\n\n")
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        text_copy_paste = input('INPUT:\n')
        text_copy_paste = 'Please wait while your summary is processing...'

        # importing model and passing in full-text string
        model = SingleModel()
        m = model(text_copy_paste)

        # creating final summary with a ratio of 0.13
        summary_file = '\n\nSUMMARY:\n' + m

        # printing summary and full text output for comparison
        print(f'\n\nSUMMARY:\n{model(text_copy_paste)}\n')
        print(f'FULL TEXT:\n', text_copy_paste)

        # appending summary output to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)

    else:
        print('\nMust choose from A, B, or C')


# vanilla summarizer
if choose_bert_or_vanilla == '2':
    print('Welcome to the Vanilla Summarizer!\n')

    # choose URL, text file, or string input
    choose_input = input('Enter A for URL input:\nEnter B for text file input:\nEnter C for copy/paste input:\n(case-sensitive)\n')

    if choose_input == 'A':
        print('\nOption A processes and summarizes web articles.\nBest results are obtained by entering a URL of a text-centric website (e.g., Wikipedia, news sites, blogs.)\n')
        print("NB: A good rule of thumb... if your browser can display the webpage via its 'Reader View,' it's likely the summarizer is compatible with the site.")
        print("If not, feel free to start over and select one of the other two summary options; it's just as simple to use the Text Document Summarizer or the Copy/Paste Summarizer.\n")
        print("Either enter the URL of the page you'd like to summarize or feel free to copy/paste one of the URLs below to take the summarizer out for a spin:")
        print('https://en.wikipedia.org/wiki/Chernobyl_(miniseries)\nhttps://www.blog.google/products/assistant/more-information-about-our-processes-safeguard-speech-data\n')
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        # enter URL
        url = input('URL to summarize: \n')
        print('Summarizing...')

        # fetching and reading in data from URL
        scraped_data = urllib.request.urlopen(url)  
        article = scraped_data.read()

        # using beautifulsoup to parse article
        parsed_article = bs.BeautifulSoup(article,'lxml')
        paragraphs = parsed_article.find_all('p')

        # iterating and appending to full-text string
        article_text = ""

        for p in paragraphs:  
            article_text += p.text

        # text clean up
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
        article_text = re.sub(r'\s+', ' ', article_text)  

        processed_article = re.sub('[^a-zA-Z]', ' ', article_text )  
        processed_article = re.sub(r'\s+', ' ', processed_article)

        # sentence-level tokenization of full text
        sentence_list = nltk.sent_tokenize(article_text)  

        # NLTK stopwords
        stopwords = nltk.corpus.stopwords.words('english')

        # creating term frequency dict
        word_frequencies = {}  
        for word in nltk.word_tokenize(processed_article):  
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        # adding term frequency ratio as dict values
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

        # ranking sentences for summary inclusion
        sentence_scores = {}  
        for sent in sentence_list:  
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        # creating final summary with default 4 highest-scoring sentences
        summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
        summary_sentences = ''.join(summary_sentences)
        summary_file = '\n\nSUMMARY:\n' + summary_sentences

        # printing summary and full text for comparison
        print(f'\nSUMMARY:\n{summary_sentences}\n\n')
        print(f'FULL TEXT:')
        print(article_text)

        # appending summary output to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)


    # text file input option
    elif choose_input == 'B':
        print('\nOption B summarizes text documents from Google Drive or your local machine.\n')
        print("\nSimply provide the path to the text document you'd like to summarize.\nOr feel free to copy/paste the filename of one of the sample documents below:\n")
        print('full_text/google_pixel_photobooth_article.txt')
        print('full_text/wash_post_youth_trending_away_from_news.txt\n')
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        document = input('Please enter your <path/to/file.txt> here:\n')

        # reading in text file
        with open(document, 'r') as d:
            text_data = d.read()

        # text clean up
        text_data = re.sub(r'\[[0-9]*\]', ' ', text_data)  
        text_data = re.sub(r'\s+', ' ', text_data)  

        processed_article = re.sub('[^a-zA-Z]', ' ', text_data )  
        processed_article = re.sub(r'\s+', ' ', processed_article)

        # sentence-level tokenization of full text
        sentence_list = nltk.sent_tokenize(text_data)  

        # NLTK stopword list
        stopwords = nltk.corpus.stopwords.words('english')

        # creating term frequency dict
        word_frequencies = {}  
        for word in nltk.word_tokenize(processed_article):  
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        # adding term frequency ratios as dict values
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

        # ranking sentences for summary inclusion
        sentence_scores = {}  
        for sent in sentence_list:  
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        # creating final summary with default 4 highest-scoring sentences
        summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
        summary_sentences = ''.join(summary_sentences)
        summary_file = '\n\nSUMMARY:\n' + summary_sentences

        # printing summary and full-text output for comparison
        print(f'\nSUMMARY:\n{summary_sentences}\n')
        print(f'FULL TEXT:')
        print(text_data)

        # appending summary to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)


    # copy/paste string input option
    elif choose_input == 'C':
        print("\nFor Option C, simply copy/paste any text you'd like to summarize below.\n\n")
        print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

        # reading in text as string
        text_copy_paste = input('INPUT:\n')
        text_copy_paste = str(text_copy_paste)

        # text processing and clean up
        text_copy_paste = re.sub(r'\[[0-9]*\]', ' ', text_copy_paste)  
        text_copy_paste = re.sub(r'\s+', ' ', text_copy_paste)  

        processed_article = re.sub('[^a-zA-Z]', ' ', text_copy_paste )  
        processed_article = re.sub(r'\s+', ' ', processed_article)

        # sentence-level tokenization of full text
        sentence_list = nltk.sent_tokenize(text_copy_paste)  

        # NLTK stopword list; optionally can use sklearn stopwords
        stopwords = nltk.corpus.stopwords.words('english')

        # creating term frequency dict
        word_frequencies = {}  
        for word in nltk.word_tokenize(processed_article):  
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())
        
        # adding term frequency ratios as dict values
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

        # final ranking for summary sentence inclusion
        sentence_scores = {}  
        for sent in sentence_list:  
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        # creating final summary with default 4 highest-scoring sentences
        summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
        summary_sentences = ''.join(summary_sentences)
        summary_file = '\n\nSUMMARY:\n' + summary_sentences

        # printing summary and full-text output for comparison
        print(f'\nSUMMARY:\n{summary_sentences}\n')
        print(f'FULL TEXT:')
        print(text_copy_paste)

        # appending summary to text file
        with open('your_summaries/summary.txt', 'a') as summary_output:
            for line in summary_file:
                summary_output.write(line)

    else:
        print('\nMust choose from A, B, or C')
