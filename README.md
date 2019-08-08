![img](img/textsummarygraphicred.png)

# Building an Extractive Text Summarization Model from the Ground Up

## Overview
The timing of this project coincides with a special era in Natural Language Processing (NLP), during sudden and enormous gains in model performance, and in particular, within the Transfer Learning method utilizing recently released pretrained models. A couple of years ago, practitioners in Computer Vision experienced the beginning of a similar leap in model performance while NLP progress remained stagnant in comparison. As Sebastian Ruder writes:  
  
### "NLP's ImageNet moment has arrived."  
  
Behind this NLP progress are a set of pretrained models that when adopted for Transfer Learning, have achieved state-of-the-art scores in a variety of NLP tasks. The trend kicked off with fast.ai's ULMfit LSTM in early 2018. Pretrained and available to all, this momentum witnessed each new model besting the previous. Elmo, BERT, XLNet, Transformer-XL, OpenAI GPT-1, GPT-2, and XLM are just a handful of these high-performance models. The most infamous of all, OpenAI's GPT-2, caused a controversy because its organization refused to release the full code, making it impossible for other researchers and Data Scientists to replicate and verify its results. The focus on GPT-2 was because it made headlines when it showcased its astounding ability to generate news article text virtually indistiguishable from a journalist's. In other words, it (nearly) passed the Turing Test. Eventually, OpenAI released a portion of its code, but not the portion enabling them to break a sound barrier in NLP progress. 

Digression aside, many (but not all) these pretrained models got a huge push in popularity when organizations like huggingFace began porting these models as PyTorch-based instead of Tensorflow for ostensible gains in speed. In addition huggingface aggregated many of these models into one repository (now called Pytorch-Transformers which you can find [here](https://github.com/huggingface/pytorch-transformers)). The repo's popularity has skyrocketed and continues to gain traction in the NLP community.


*complete readme currently in final edit and coming imminently...*