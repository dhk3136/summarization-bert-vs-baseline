![text_summary_graphic](img/textsummarygraphicred.png)

# Building an NLP Extractive Text Summarization Model--from the Ground Up

## Overview  
The timing of this text summarization project coincides with a special era in Natural Language Processing (NLP), during sudden and enormous gains in model performance, and in particular, within Transfer Learning methods utilizing recently released pretrained models (e.g., BERT, XLNet, OpenAI). A couple of years ago, practitioners in Computer Vision experienced the beginning of a similar leap in model performance while NLP progress remained stagnant in comparison. But much has changed: As Sebastian Ruder writes:  

### "NLP's ImageNet moment has arrived."
  
This project has a two-fold aim: first, to provide informed research and the context of the current state of NLP in its present and dramatic transformation which seems to change on a weekly basis. Second, to produce my own text summarization algorithm as a way of conducting a deep study of the relationship between probabilities, tokenization, and word proximity--with the caveat that I'd be able to manipulate the algorithm without the hinderance of complex and hidden architectures featured in CNNs, LSTMs, and the new transformer designs. In other words, this project represents both research-based and practical applications in and around text summarization--not for benchmarking--as a pretrained model can provide that kind of top performance with ease. Rather, as a reference for practitioners who have little time to read academic papers and the technically inclined who wish to study the underpinnings of extractive text summarization.  

I sought to build an algorithm from the ground up and from which I could openly track, learn, code, and study code. In short, I wanted to see what was 'under the hood' before these high-performance models seal it shut for good. Distance metrics such as cosine similarity, and benchmark metrics that optimize for recall and precision, leave behind much of the baseline modeling process. For example, are all preprocessing routines equal in their ability to tokenize? How clean does clean text need to be for optimization? Can simple linear algebra help to search for and convert rare words into new, high-probability features?  

I'd wager that pretrained models (training from scratch takes 1-2 weeks on the fastest set of GPUs) are just on the horizon for professional use by data scientists in various organizations. They achieve state-of-the-art results on most Natural Language Processing "tasks" by standardized metrics, and given the choice, most organizations would not opt to train such a model from scratch for the sake of posterity. If you'd like to check out a showcase of the latest and greatest, I highly recommend [huggingFace's repository](https://github.com/huggingface/pytorch-transformers) which ports these models into PyTorch libraries (as opposed to Tensorflow).  

As such, I have no problem at all with their use. In fact, the release of these models is in accord with the general spirit of open source distribution from which so many benefit. In other words, I'm far from a desire to judge the new and shiny, and I'll likely benefit from the change, as well.

## Measurement and Metrics: How good is my summarizer?  
If you're interested in running an entire corpus through the summarizers, here are some important considerations.
 - In NLP summarization, models are evaluated by a variety of metrics. As such, the number of datasets with which to test a summarizer is somewhat limited and tied to these metrics. This constraint exists because benchmarking requires both a large set of news articles to be summarized and their accompanying "gold" summaries (i.e., reference or target summaries). In other words, extensive human labor in labeling these reference summaries is involved, as there currently is no other way to compare your summary with a "gold" summary. 
 
 - If you wish to benchmark either of these algorithms, ROUGE (favors Recall) or BLEU (favors Precision) or METEOR are standard places to start.
  - Be forewarned, however, that these metrics are fraught with criticisms by NLP practioners. For example, computational socio-linguist, Rachael Tatman, notes that BLEU's single biggest flaw is also the most important thing it's supposed to measure: meaning. In other words, BLEU and ROUGE do not take into account accuracy regarding *the underlying meaning* of a sentence in its scoring. And within the NLP summarization task, we know that's a big problem. In fact, summarization depends on context, meaning, and their probabilities to produce coherent, useful, and human-readible summaries. Instead, these metrics calculate scores based on the number of overlapping or intersecting words between your summary and a reference summary. How *close* you got to a good summary is measured by common unigrams, bigrams, trigrams, and 4-grams. In addition, this mere counting and calculating of intersections overlooks another major problem: syntactic structure. It is possible, in other words, to achieve a high score on three different sentences containing the same words. Yet, it's possible that only one of those sentences actually makes any grammatical sense. Out of order words that match the reference summary are still matches--and are correspondingly rewarded for it. Even a lengthy sentence capturing both the underlying (semantic) meaning and syntactic structure can be penalized based on the length of the sentence versus two sentences that struggle to capture each of those essential tasks.

 - Although there is no shortage of criticisms of these metrics by NLP practitioners, these metrics are still around for two main reasons:
  - 1) They are ubiquitous. They are widely-used in a practice that loves its standardization (NLP). While there are pockets of proposed alternatives, they are a minority, and there is little movement to replace them wholesale.
  - 2) Replacing these metrics with more sophisticated ones would significantly increase the amount of compute it takes to measure NLP tasks. ROUGE and BLEU are lightweight with simple calculations you could do by hand given enough time. Thus, the immediate gratification of obtaining a score further detracts from potentially sound alternatives.

Returning to Ruder, he posts this admonition at the top of his NLP Summarization site:  

> __Warning: Evaluation Metrics__

> For summarization, automatic metrics such as ROUGE and METEOR have serious limitations:

 > They only assess content selection and do not account for other quality aspects, such as fluency, grammaticality, coherence, etc.
 > To assess content selection, they rely mostly on lexical overlap, although an abstractive summary could express they same content as a reference without any lexical overlap.
 > Given the subjectiveness of summarization and the correspondingly low agreement between annotators, the metrics were designed to be used with multiple reference summaries per input. However, recent datasets such as CNN/DailyMail and Gigaword provide only a single reference.

