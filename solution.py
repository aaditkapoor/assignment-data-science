
# coding: utf-8

# ## We aim to find the keywords in file.pdf and create a distribution chart of the same.
# ### Using File: file.pdf
# ### Stopwords file: stopwords.txt

# In[349]:


# Importing
import numpy as np
import pandas as pd
import PyPDF2 # For extracting text
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize # For spliting
from string import punctuation
from collections import Counter


# In[259]:


# Loads the stopwords from file
def load_stopwords():
    fp = open("stopwords.txt","r")
    words = []
    for word in fp.readlines():
        words.append(word.replace("\n","")) # Getting words
    return words # filtered words

# Extracts text from the pdf
def extract_text_from_pdf():
    data = {}
    pdfFileObject = open('file.pdf', 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    count = pdfReader.numPages # Number of pages in the pdf
    print ("Number of pages in pdf: ", count)
    
    for i in range(count):
        data["page" + str(i)] = [] # Building empty data store variable in the form data = {'page0':[], 'page1':[]}
        
    for i in range(count):
        page = pdfReader.getPage(i) # Getting each page
        data["page"+str(i)].append(page.extractText().lower()) # Getting text from each page
    
    return data # Getting the extracted text


stopwords = load_stopwords() # Loading stopwords

# Removes stopwords and punctuation
def remove_stopwords_punc(sentences):
    global stopwords
    new = []
    for sentence in sentences:
        for word in sentence:
            if word not in stopwords and word not in punctuation:
                new.append(word) # Removing all the stopwords and assigning it into a list
    return new


# In[260]:


texts = extract_text_from_pdf() # Extracting text


# In[261]:


# We now have a list of lists
tokenized_words = [sentence[0].split() for sentence in texts.values()] # Splitting words into list
print ("length of tokenized_words: ", len(tokenized_words))


# In[262]:


filtered_words = remove_stopwords_punc(tokenized_words) # All the filtered words
print ("length of filtered words: ", len(filtered_words)) # length


# In[263]:


# Next we will find the most common words from these tokens
words_count = {}
for word in filtered_words:
    words_count[word] = 0 # Intializing all the words to 0 count


# In[264]:


# Getting common words count
repeating_words = list(set(filtered_words))   


# In[316]:


counter = Counter(filtered_words) # Creating a counter
print ("Getting the common words")
common_words = dict(counter.most_common())
vals = list(common_words.values())

def remove_ones(vals, mean=None):
    vals[:] = [val for val in vals if val != 1] # Removing all ones from list
    return vals

                
vals = remove_ones(vals) # Removing ones from common_words numbers
npvalues = np.array(vals) # Changing into numpy array


# In[317]:


max_occurence_count = np.max(npvalues) # max count
min_occurence_count = np.min(npvalues) # min
mean_occurence_count = np.mean(npvalues) # Mean value of occurences


print ("Max:", max_occurence_count)
print ("Min: ", min_occurence_count)
print ("Mean: ", mean_occurence_count)


# In[336]:


# Returns the final list of keywords and the max,min range of words
def get_max_min_word(common_words):
    max_word = ""
    min_word = ""
    keywords = [] # All the max. words
    for i in common_words:
        if common_words[i] == max_occurence_count:
            max_word = i
            keywords.append(max_word)
        if common_words[i] == min_occurence_count:
            min_word = i
   
    print ("Max word is: ", max_word)
    print ("Min word is: ", min_word)
    return keywords

# Calculates score of each possible keyword
def calculate_percentage(common_words):
    percentages = {}
    for k in common_words:
        if common_words[k] != 1: # Discarding all the '1' value common words
            percentages[k] = common_words[k] / len(vals) * 100 # Calculating percentage of the possible keywords
    return percentages


# In[334]:


keywords = get_max_min_word(common_words) # the important keyword


# In[355]:


percentages = calculate_percentage(common_words) # scores
csv_data = pd.DataFrame(list(percentages.items()), columns=['word','score']) # Changing into pandas datafram
csv_data.head()


# In[354]:


csv_data.to_csv("keyword_score.csv") # Saving

