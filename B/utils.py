import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# removes diacritic and unwanted things in cube brackets([Chorous]) and make all text lowercase
def clean_data(data):
  newData = []
  for text in data:
    text = re.sub(r'\[.*\]', '', text)
    newData.append(text)
  data = newData

  newData = []
  for text in data:
    text = text.lower()
    text = text.replace(r'\n', ' ')
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace('?', ' ')
    text = text.replace(':', ' ')
    text = text.replace('!', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('-', ' ')
    text = text.replace('+', ' ')
    text = text.replace('\\', ' ')
    text = text.replace('/', ' ')
    text = text.replace('|', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('"', ' ')
    text = text.replace('&', ' ')
    text = text.replace('~', ' ')
    text = text.replace("'", '')
    newData.append(text)
  data = newData

  return data

# splits texts into words
# def split_data(data):  
#   newData = []
#   for text in data:
#     text = text.split(' ')
#     newData.append(text)
#   return newData

def split_data(data):
    newData = []
    for text in data:
        # Ensure text is a string
        if isinstance(text, str):
            text = text.split(' ')
        else:
            # Handle non-string data: skip, convert to string, or other logic
            text = []  # Example: skip or handle differently
        newData.append(text)
    return newData

# stems given data by snowball stemmer
def stem(data): 
  stemmer = SnowballStemmer("english")
  newData = []
  for text in data:
    newText = []
    for word in text:
      newText.append(stemmer.stem(word))
    newData.append(newText)
  return newData

# removes stop words from given data
def remove_stopwords(data):
  stopWords = set(stopwords.words('english'))
  newData = []
  for text in data:
    newText = [word for word in text if not word in stopWords]
    newData.append(newText)
  return newData

# creates vocabulary of top threshhold used words in given data, if threshhold is not given, includes all words
def create_vocab(data, threshhold = None):
  vocabSizes = {} # dictionary with words and its number of occurrences
  for text in data:
    for word in text:
      if word not in vocabSizes:
        vocabSizes[word] = 1
      else:
        vocabSizes[word] += 1

  i = 0
  vocabTopWords = {} # dictionary with words and its index used in tf-idf vector
  for word in sorted(vocabSizes, key=vocabSizes.get, reverse=True):
    vocabTopWords[word] = i
    i += 1
    if threshhold is not None and i >= threshhold:
      break
  return vocabTopWords

# mapping from genre to class number
genre_to_class = {
    'Pop': 0,
    'Rock': 1,
    'Country': 2,
    'Electronic': 3,
    'Hip-Hop': 4
}
