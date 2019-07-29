#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# code adopted from the paper, "Variations of the Similarity Function of
# TextRank for Automated Summarization (Barrios et al, 2018)"


# In[28]:


from summa import summarizer
import pandas as pd

df = pd.read_csv('wikihow3000.csv')


# In[246]:


# df.columns = ['overview', 'headline', 'text', 'something', 'title']


# In[259]:


# df = df.drop(['overview', 'something', 'title'], axis=1)


# In[18]:


# df.to_csv('wikihow3000.csv')


# In[43]:


df1 = pd.read_csv('sys_refs.csv', encoding='UTF-8', nrows=1000)


# In[47]:


df1.info()


# In[48]:


df1.columns = ['system', 'references']


# In[50]:


df1.head()


# In[51]:


df['text'] = df['text'][:1000]


# In[58]:


ref_summs = []
for each in df1['references']:
    ref_summs.append(str(each))
# print(text)
sys_summs = []
for each in df1['references'].values:
    sys_summs.append(summarizer.summarize(each))
print(system_sums)


# In[60]:


sys_summs = sys_summs[:1000]


# In[61]:


len(sys_summs)


# In[ ]:


# [each for each in system_sums if each == "None"]


# In[64]:


reference_join = ''.join([each for each in df1['references'].values])
reference_split = reference_join.split()
# print(reference_split)


# In[65]:


len(reference_split)


# In[ ]:


# rouge function / see below for alt rouge calculations; perhaps more coherent

def rouge_scores(nums_of_overlap_words, reference_sums, system_sums):
    for each in reference_sums:
        nums_of_reference_words
        for words in system_sums:
            nums_of_system_words = len((words)) 
            rouge_recall = nums_of_overlap_words/nums_of_reference_words
            rouge_precision = nums_of_overlap_words / nums_of_system_words
  
        # prints final rouge scores 
        print(f'Rouge Recall:\n{rouge_recall}\n\nRouge Precison:\n{rouge_precision}\nRouge f1:') 
    
nums_of_system_words = len((words))
nums_of_reference_words = (len(each))    
reference_sums = len([each for each in df['headline'].values if each in system_sums] and each != "''")
rouge_scores(nums_of_system_words, nums_of_reference_words, system_sums)


# In[66]:


# converting to string and back to a list to extract from pandas dataframe
sys_sum_join = ''.join(sys_summs)
sys_sum_split = sys_sum_join.split()

# print(system_sums_split) # long output, use sparingly

# more stopwords I added
stops = ['and', 'the', '.', ',', 'is', 'in', 'if', 'for', 'an', 'a', '-', ';', '!', '\n', "''"]

# words in common between system and referenc summaries

overlap = []
for each in sys_sum_split:
#     print(each)
    for word in reference_split:
          if each != stops:
                if each == word:
                    overlap.append(each)
    
            


# In[ ]:


# # overlap = []
# # for each in df['headline'].values:
# # #     headline.append(str(each))
# #     for word in system_sums:
# #         if each == word:
# #             overlap.append(each)
# # print(overlap)
# system_sums_join = ''.join(system_sums)
# system_sums_split = system_sums_join.split()
# # print(system_sums_split)

# overlap = []
# for each in system_sums_split:
# #     print(each)
#     for word in reference_split:
# #         print(word)
#         if each == word:
#             overlap.append(each)
            
# print(system_sums_split)
# print(reference_split)


# In[67]:


# system and reference matches for rouge

overlap_total = len(overlap)
print(overlap_total)


# In[68]:


# number of each

reference_total = len(reference_split)
system_sums_total = len(sys_sum_split)


# In[12]:





# In[69]:


# rouge metric recall score

rouge_recall = overlap_total / reference_total
print(rouge_recall)


# In[70]:


# rouge precision score

rouge_precision = overlap_total / system_sums_total
print(rouge_precision)


# In[71]:


# rouge f1-score

alpha = 0.5
rouge_f1 = rouge_precision * rouge_recall / (alpha * rouge_precision) + alpha * rouge_recall
print(rouge_f1)


# In[18]:


# troubleshooting mismatches in overlap (eventually fixed)
compare = pd.DataFrame(system_sums)
compare['system'] = ref_summ


# In[19]:


compare.columns = ['system', 'reference']

compare.tail()


# In[20]:


# exporting to a simplified format
df.to_csv('wikihow_filter.csv')


# In[23]:


# rechecking data
compare.info()


# In[24]:


# more data for missing rows
pd.set_option('display.max_rows', 3000)


# In[25]:


compare.tail()


# In[26]:


compare.info()


# In[27]:


# last but cleanest export
compare.to_csv('sys_ref_cols.csv')


# In[ ]:




