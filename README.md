![text_summary_graphic](img/textsummarygraphicred.png)

# Building an NLP Extractive Text Summarization Model--from the Ground Up

## Overview  
The timing of this text summarization project coincides with a special era in Natural Language Processing (NLP), during sudden and enormous gains in model performance, and in particular, within Transfer Learning methods utilizing recently released pretrained models (e.g., BERT, XLNet, OpenAI). A couple of years ago, practitioners in Computer Vision experienced the beginning of a similar leap in model performance while NLP progress remained stagnant in comparison. But much has changed: As Sebastian Ruder writes:  

### "NLP's ImageNet moment has arrived."
  
This project has a two-fold aim: first, through an analysis of extractive summarization algorithms to provide informed research and the context of the current state of NLP in its present and dramatic transformation which seems to change on a weekly basis. Second, to produce two summarization models as a way of conducting a study of the relationship between word and sentence probabilities, tokenization, contextual proximity, semantic performance, and syntactic representation.

The first summarizer serves as a baseline model, a simple algorithm that only uses NLTK for processing and does not rely on any form of training or machine learning--just straightforward probabilities for word and sentence inclusion into a final summary.

The second summarizer is quite the opposite: at its bare minimum, it is an enormous, pretrained, and unsupervised language model with state-of-the-art architecture with transfer learning as its intention. Its name is BERT. BERT's authors claim the pre-trained versions excel at NLP tasks without a pre-specified intent, but that the models still perform extremely well on "downstream tasks." And for the most part, they're right. Tasks such as classification, question-answering, Named Entity Recognition all do very well without further manipulation. It also performs very well on benchmarking metrics such as SQuAD, MNLI, and MRPC. BERT was trained on the entire Wikipedia corpus as well as the entire Toronto Book corpus. Its parameters are massive. It comes in a variety of sizes and includes multi-lingual options. For this project I utilized the biggest model, BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340 million parameters. That's right: 340 million hyperparameters.

The [paper's](https://arxiv.org/abs/1810.04805) authors state:

 > "BERT outperforms previous methods because it is the first *unsupervised, deeply bidirectional* system for pre-training NLP."

On top of my BERT-Large model, I utilize K-Means as a way of graphing relationships and most importantly, measuring distances using metrics such as eigenvalue calculations and cosine similarity. The model also borrows from OpenAI's tokenizer. Although much is made of BERT's ingesting of enormous corpora, additional training using domain-specific data helps performance as is the case with most pretrained models. I chose a limited array of standardized NLP summarization datasets, the BBC News corpus.  

## "OK, enough of all this long text. I thought this project was about summarization!"

Fair enough. If you'd like to jump straight into the summarizer, here are some simple instructions. Launching the Colab notebook is the quickest and easiest way to take the summarizer for a spin. Once you follow the setup instructions in the notebook, you'll be presented with three options for summarization:

- A) Reading in a URL from a text-centric site such as Wikipedia, news sites (without a paywall), blogs, etc;
- B) Loading your own text document for summarization;
- C) Simple copy/paste of any kind of text directly into the prompt

For all options, you'll see both the full text and summary printed in the notebook's output. Additionally, the summarizer will output a text file of your summary which you can find in the directory `your_summaries`.

## TLDR;?
BERT considers context as part of its inherent design. Not only is BERT an unsupervised model, it also is a transformer model. No convolutions. No recurrent movement. It's bidirectional capability allows it to consider tokens from both the left *and* right of a sentence, for example. In other words, it surpasses earlier models capable only of predicting context from one-sided input. In addition, BERT masks (drops) 15 percent of its tokens and predicts on those, further pushing to model relationships among tokens and creating a sophisticated framework for generating probabilities and text output.

Further, BERT's hyperparameter tuning significantly increases performance based on specific NLP tasks such as GLUE. Many of these parameter adjustments are also freely available, although additional (and expensive) training time is required to take advantage. Unfortunately, summarization is not on the list of specified tasks nor is there any guidance on tuning.

Computational socio-linguist, Rachael Tatman, lists summarization first on her short list of the most difficult NLP tasks. This includes the lack of reproducibility options from the original paper, deeply flawed metrics that emphasize lexical overlap at the expensive of measuring the capture of underlying meanings (semantics) which in turn relies on contextual measurement (more on these problems below). As such, methods related to question-answering, next word prediction are adapted to create BERT summarizers. 

Given the choice, most organizations would not opt to train such a model from scratch for the sake of posterity. If you'd like to check out a showcase of the latest and greatest, I highly recommend [huggingFace's repository](https://github.com/huggingface/pytorch-transformers) which ports these models into PyTorch libraries (as opposed to Tensorflow).  

As such, I have no problem at all with their use. In fact, the release of these models is in accord with the general spirit of open source distribution from which so many benefit. In other words, I'm far from a desire to judge the new and shiny, and clearly, I'm benefitting from the change, as well.

Here's how I consider pretraining in general:  

> *PRETRAINED MODELS = TRANSFER LEARNING - (TIME COMPRESSION + MATERIAL RESOURCE)*

Take, for example, that BERT takes four days to train from scratch on __16 TPUs__. Keep in mind, this process happens in-house at Google. While those extensive resources are astounding, consider that it takes __2 weeks__ on a single TPU to achieve the same on the *small* BERT-base model at a cost of around $500. Is this cost and resource-prohibitive? I'd wager it is for most of us. I'll let you do the math!

## Results  
Here are a few examples comparing the output from the two summarizers taking the same input.  


## Measurement and Metrics: How good is my summarizer?  
If you're interested in running an entire corpus through the summarizers, here are some important considerations.
 - In NLP summarization, models are evaluated by a variety of metrics. As such, the number of datasets with which to test a summarizer is somewhat limited and tied to these metrics. This constraint exists because benchmarking requires both a large set of news articles to be summarized and their accompanying "gold" summaries (i.e., reference or target summaries). In other words, extensive human labor in labeling these reference summaries is involved, as there currently is no other way to compare your summary with a "gold" summary. 
 
 - If you wish to benchmark either of these algorithms, ROUGE (favors Recall) or BLEU (favors Precision) or METEOR are standard places to start.
  - Be forewarned, however, that these metrics are fraught with criticisms by NLP practioners. For example, Tatman incisively notes that BLEU's single biggest flaw is also the most important thing it's supposed to measure: meaning. In other words, BLEU and ROUGE do not take into account accuracy regarding *the underlying meaning* of a sentence in its scoring. And within the NLP summarization task, we know that's a big problem. In fact, summarization depends on context, meaning, and their probabilities to produce coherent, useful, and human-readible summaries. Instead, these metrics calculate scores based on the number of overlapping or intersecting words between your summary and a reference summary. How *close* you got to a good summary is measured by common unigrams, bigrams, trigrams, and 4-grams. In addition, this mere counting and calculating of intersections overlooks another major problem: syntactic structure. It is possible, in other words, to achieve a high score on three different sentences containing the same words. Yet, it's possible that only one of those sentences actually makes any grammatical sense. Out of order words that match the reference summary are still matches--and are correspondingly rewarded for it. Even a lengthy sentence capturing both the underlying (semantic) meaning and syntactic structure can be penalized based on the length of the sentence versus two sentences that struggle to capture each of those essential tasks.

 - Although there is no shortage of criticisms of these metrics by NLP practitioners, these metrics are still around for two main reasons:
  - 1) They are ubiquitous. They are widely-used in a practice that loves its standardization (NLP). While there are pockets of proposed alternatives, they are a minority, and there is little movement to replace them wholesale.
  - 2) Replacing these metrics with more sophisticated ones would significantly increase the amount of compute it takes to measure NLP tasks. ROUGE and BLEU are lightweight with simple calculations you could do by hand given enough time. Thus, the immediate gratification of obtaining a score further detracts from potentially sound alternatives.

Returning to Ruder, he posts this admonition at the top of his NLP Summarization site:  

> __Warning: Evaluation Metrics__

> For summarization, automatic metrics such as ROUGE and METEOR have serious limitations:

 - They only assess content selection and do not account for other quality aspects, such as fluency, grammaticality, coherence, etc.
 - To assess content selection, they rely mostly on lexical overlap, although an abstractive summary could express they same content as a reference without any lexical overlap.
 - Given the subjectiveness of summarization and the correspondingly low agreement between annotators, the metrics were designed to be used with multiple reference summaries per input. However, recent datasets such as CNN/DailyMail and Gigaword provide only a single reference.

Still, I encourage you to try one or two of these metrics over your dataset so you can make your own informed opinion and judgment.  

Here are a few common starting places:

Datasets:  
 - CNN/DailyMail: A very common dataset for NLP summarization. Includes both full text articles and their accompanying reference (gold) summaries. Preprocessed and DIY options for constructing the dataset are available [here]()
 -  

## Bibliography and Acknowledgements:  
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* In NAACL-HLT 2018.

