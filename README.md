![text_summary_graphic](img/textsummarygraphicred.png)

# BERT vs Vanilla Summarization: A Semantics-First Approach

## Overview  
The timing of this text summarization project coincides with a special era in Natural Language Processing (NLP), during sudden and enormous gains in model performance, and in particular, within Transfer Learning methods utilizing recently released pretrained models (e.g., BERT, XLNet, GPT-2). As NLP researcher Sebastian Ruder notes:  

### "NLP's ImageNet moment has arrived."
  
That is, just a couple of years ago, practitioners in Computer Vision experienced the beginning of a similar leap in model performance while NLP progress remained stagnant in comparison. But much has changed. 

## Purpose  
This project has a two-fold aim:  
- First, to produce two summarization models in order to study the relationship between word and sentence probabilities, token prediction methods, contextual proximity vs semantic inter-sentence coherence, and syntactic representation.  In particular, this project's focus centers on modeling with an emphasis on powerful pretrained networks which for the first time allows NLP to apply and to encourage the use of transfer learning methods. More on these considerations follow later in this document.  
- Second, through an analysis of extractive summarization algorithms to provide informed research within the context of the current state of NLP in its present and dramatic transformation occurring at breakneck speed on a weekly--and sometimes--daily basis.

## Method and Models  
The first summarizer serves as a baseline model, a simple algorithm solely using NLTK for processing that does not rely on any form of training or machine learning--just straightforward probabilities for word and sentence inclusion into a final summary.

The second summarizer is quite the opposite: at its bare minimum, it is an enormous, pretrained, and unsupervised language transformer with state-of-the-art architecture with transfer learning as its intention. Its name is BERT. BERT's authors claim the pre-trained versions excel at NLP tasks without a pre-specified intent, but that the models still perform extremely well on "downstream tasks." And for the most part, they're right. Tasks such as classification, question-answering, Named Entity Recognition all do very well without further manipulation. It also performs very well on benchmarking metrics such as SQuAD, MNLI, and MRPC. BERT was trained on the entire Wikipedia corpus as well as the entire Toronto Book corpus. Its parameters are massive. It comes in a variety of sizes and includes multi-lingual options. For this project I utilized the biggest model, BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340 million parameters. That's right: 340 million hyperparameters.

BERT also leads the way in pushing the Transfer Learning trend for NLP into the spotlight. Google's release of the architecture and source code allowed users to replicate the results--or close to it--released in its paper. As expected, TensorFlow, another Google product, was offered as the tensor library supporting BERT. Soon after, a PyTorch port of BERT, achieving identical performance, was released alongside several other popular transfer learning models (the aforementioned XLNet, XL Transformer, XLM, and transformer models, generally). BERT has been so successful that Facebook decided to make their XLM model a modified version of BERT intended for cross-lingual use.

### So how does BERT work?
The [paper's](https://arxiv.org/abs/1810.04805) authors state:

 > "BERT outperforms previous methods because it is the first *unsupervised, deeply bidirectional* system for pre-training NLP."

BERT considers context as part of its inherent design. Not only is BERT an unsupervised model, it also is a transformer model. No convolutions. No recurrent movement. Its bidirectional capability allows it to consider tokens from both the left *and* right of a word, for example. In other words, it surpasses earlier models capable only of predicting context from one-sided input or "shallow bidirectional" input. In addition, BERT masks (drops) 15 percent of its tokens and predicts on those, further pushing to model relationships among tokens and creating a sophisticated framework for generating probabilities and text output.


## Analysis  
Here, we'll use a summarized example from URL text, of which the full-text sample document I've included with summarizer: the Wikipedia page for the critically-acclaimed HBO series, Chernobyl.

BERT Summarizer:  
```python
"Chernobyl is a 2019 historical drama television miniseries created and written by Craig Mazin and directed by Johan Renck for HBO. The miniseries is based in large part on the recollections of Pripyat locals, as told by Belarusian Nobel laureate Svetlana Alexievich in her book Voices from Chernobyl.[2] Writer Craig Mazin began researching for the project in 2014, by reading books and government reports from inside and outside the Soviet Union. Director Johan Renck heavily criticised the amount of diverse and eye-catching modern windows in the houses, but was not concerned about removing them in post-production."
```

This is how BERT sees the first sentence:  

![bert_masked_tokens](img/BERT_token_predictions_1.jpg)  

Impressive, yes? Not only can BERT predict "drama" with a high probability, its capability really shines when predicting "miniseries" over "series," which to my mind is a distinction difficult even for a human annotator to make. Here's another one:

```python
"We just learned that one of these language reviewers has violated our data security policies by leaking confidential Dutch audio data."
```  

![bert_masked_tokens_2](img/BERT_token_predictions_2.jpg)  

BERT intentionally masks a small ratio of words so that it can predict probabilities of the correct word--in context. This is different from measuring distance as a metric of semantic similarity. In addition, BERT is ambidextrous (maybe too much on the anthropomorphism). The model can take input from both sides which expands its contextual framework resulting in a huge advantage over uni-directional input.  

On top of my BERT-Large model, I utilize K-Means as a way of graphing relationships and most importantly, measuring cluster proximity via edge weights and cosine similarity. The model also borrows from OpenAI's tokenizer. Although much is made of BERT's ingesting of enormous corpora, additional training using domain-specific data helps performance as is the case with most pretrained models. I chose from the limited array of standardized NLP summarization datasets. Given the extensive resources necessary to train BERT, not to mention hyperparameter tuning, the BBC News corpus was reasonable and not as large or unwieldy as the standard CNN/Dailymail dataset. In addition, as you'll read below (if you can summon the patience), my impetus for this project is far less about posting a nice ROUGE or BLEU score and far more about semantic, inter-sentence coherence--within news text or otherwise. Hence the title.

## "OK, what's up with all this readme text? Didn't you say this was about summarization!"

Fair enough. If you'd like to jump straight into the summarizer, here are some simple instructions. Launching the Colab notebook is the quickest and easiest way to take the summarizer for a spin. Everything you need to run the summarizer is contained within the notebook. Once you follow the simple setup instructions there, you'll be presented with three options for summarization:

- A) Reading in a URL from a text-centric site such as Wikipedia, news sites (without a paywall), blogs, etc;
- B) Loading your own text document for summarization;
- C) Simple copy/paste of any kind of text directly into the prompt

For all options, you'll get your summary printed in the notebook's output. But that's not helpful when you want to retain your summary. As such, the summarizer will output a text file of your summary which you can find in the directory `your_summaries`.

## TL;DR?
BERT's hyperparameter tuning significantly increases performance based on specific NLP tasks such as GLUE. Many of these parameter adjustments are also freely available, although additional (and expensive) training time is required to take advantage. Unfortunately, summarization is not on the list of specified tasks nor is there any guidance on tuning.

Computational socio-linguist, Rachael Tatman, lists summarization first on her short list of the most difficult NLP tasks. This includes the lack of reproducibility options from the original paper, deeply flawed metrics that emphasize lexical overlap at the expensive of measuring the capture of underlying meanings (semantics) which in turn relies on contextual measurement (more on these problems below). As such, methods related to question-answering, next word prediction are adapted to create BERT summarizers. 

Given the choice, most organizations would not opt to train such a model from scratch for the sake of posterity. If you'd like to check out a showcase of the latest and greatest, I highly recommend [huggingFace's repository](https://github.com/huggingface/pytorch-transformers) which ports these models into PyTorch libraries (as opposed to Tensorflow).  

As such, I have no problem at all with their use. In fact, the release of these models is in accord with the general spirit of open source distribution from which so many benefit. In other words, I'm far from a desire to judge the new and shiny, and clearly, I'm benefitting from the change, as well.

Here's how I consider pretraining in general:  

> *PRETRAINED MODELS = TRANSFER LEARNING - (TIME COMPRESSION + MATERIAL RESOURCE)*

Take, for example, that BERT takes four days to train from scratch on __16 TPUs__. Keep in mind, this process happens in-house at Google. While those extensive resources are astounding, consider that it takes __2 weeks__ on a single TPU to achieve the same on the *small* BERT-base model at a cost of around $500. Is this cost and resource-prohibitive? I'd wager it is for most of us. I'll let you do the math!

Also, on this *very* day, three announcements were made regarding Transfer Learning. First, Google releases a distilled version of BERT, named ALBERT (this has got to stop) that is supposed to be more manageable in size, training, and footprint--while outscoring the original BERT in three NLU (Natural Language Understanding) benchmarks. Second, the popular HuggingFace organization released its pretrained models as compatible with TensorFlow 2.0 which now gives users an additional option to PyTorch. Third, AllenNLP, the well-known creators of ELMO, announced full compatibility and porting with the HuggingFace models. More significantly, they released Interpret, an interactive framework for visualizing and explaining what's happening under the hood of many state-of-the-art language models. This all happened TODAY.

## Results  
Here are a few additional examples comparing the output from the two summarizers taking the same input.


```python
Vanilla' Summarizer:
Vladimir Medinsky, Russian culture minister, whose father was one of the Chernobyl liquidators, called the series “Masterfully made” and “filmed with great respect for ordinary people”. The series centers around the Chernobyl nuclear disaster of April 1986 and the unprecedented cleanup efforts that followed. Simultaneously with the initial series announcement, it was confirmed that Jared Harris would star in the series.
```
*Full text is excluded here for space and length considerations. However, I've included those files in the repo's `fulltext` directory.*

After all my blathering about BERT's superhero qualities, why does he produce reasonable--but not astounding--summaries? Recent papers have criticized the model when it comes to inter-sentence coherence. While BERT excels at *most* NLP tasks, next-sentence prediction, contra Google's claims, apparently is not one of them:

 > "In the BERT paper, Google proposed a next-sentence prediction technique to improve the model’s performance in downstream tasks, but subsequent studies found this to be unreliable."

Hmm. I hadn't heard those criticism before I set out on this project. It would seem that next-sentence prediction could be important to a summarization task. Just *maybe*. Bitterness ensues.

## Measurement and Metrics: How good is my summarizer?  
If you're interested in running an entire corpus through the summarizers, here are some important considerations.
 - In NLP summarization, models are evaluated by a variety of metrics. As such, the number of datasets with which to test a summarizer is somewhat limited and tied to these metrics. This constraint exists because benchmarking requires both a large set of news articles to be summarized and their accompanying "gold" summaries (i.e., reference or target summaries). In other words, extensive human labor in labeling these reference summaries is involved, as there currently is no other way to compare your summary with a "gold" summary. 
 
If you wish to benchmark either of these algorithms, ROUGE (favors Recall) or BLEU (favors Precision) or METEOR are standard places to start.
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


SyncedReview
Google’s ALBERT Is a Leaner BERT; Achieves SOTA on 3 NLP Benchmarks
Tony Peng
Sept 27, 2019
https://medium.com/syncedreview/googles-albert-is-a-leaner-bert-achieves-sota-on-3-nlp-benchmarks-f64466dd583


BERT – State of the Art Language Model for NLP November 7, 2018 by Rani Horev - LyrnAI


Jeremy Howard, Sebastian Ruder:
Fine-tuned Language Models for Text Classification. CoRR abs/1801.06146 (2018)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin:
Attention Is All You Need. CoRR abs/1706.03762 (2017)

Guillaume Lample, Alexis Conneau:
Cross-lingual Language Model Pretraining. CoRR abs/1901.07291 (2019)

Eric Wallace, Jens Tuyls, Junlin Wang, Sanjay Subramanian, Matt Gardner, Sameer Singh:
AllenNLP Interpret: A Framework for Explaining Predictions of NLP Models. abs/1909.09251 Preprint. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova:
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018)

Andrew M. Dai, Quoc V. Le:
Semi-supervised Sequence Learning. CoRR abs/1511.01432 (2015)

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer:
Deep contextualized word representations. CoRR abs/1802.05365 (2018)

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever:
Improving Language Understanding by Generative Pre-Training. Preprint. (2018)

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov, Quoc V. Le:
XLNet: Generalized Autoregressive Pretraining for Language Understanding. CoRR abs/1906.08237 (2019)