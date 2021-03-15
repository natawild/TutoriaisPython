import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk 
import string 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report


print('_________________________Spam Filter____________________________')

messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))

for mess_no, message in enumerate(messages[:10]):
	print(mess_no,message)


print(messages[0])	

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
print(messages)


print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)

print(messages)


messages['length'].plot.hist(bins=50)
#plt.show()

print(messages['length'].describe())
messages[messages['length'] == 910]['message'].iloc[0]


messages.hist(column='length',by='label',bins=60,figsize=(12,4))
#plt.show()


mess = 'Sample message! Notice: it has punctuation.'

nopunc = [c for c in mess if c not in string.punctuation]
print(nopunc)


print(stopwords.words('english'))

nopunc = ''.join(nopunc)

x = ['a', 'b', 'c', 'd']
print(''.join(x))

nopunc.split()
#remove comum words 
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print(clean_mess)

def text_process(mess):
	'''1.remove punc 
	2. remove stop words 
	3. return list of clean text words 
	'''
	nopunc = [char for char in mess if char not in string.punctuation]
	nopunc = ''.join(nopunc)
	return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


print(messages['message'].head(5).apply(text_process))


print('_________________________Vectorization____________________________')

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)

bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)


print(bow_transformer.get_feature_names()[9554])

messages_bow = bow_transformer.transform(messages['message'])

print(messages_bow)

print('Shape of Sparse Matrix: ', messages_bow.shape)

print(messages_bow.nnz)

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfid4 = tfidf_transformer.transform(bow4)
print(tfid4)


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
spam_detect_model.predict(tfid4)[0]
messages['label'][3]
all_pred = spam_detect_model.predict(messages_tfidf)
print(all_pred)

msg_train, msg_test, label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)

print(msg_train)

pipeline = Pipeline([
	('bow', CountVectorizer(analyzer=text_process)),
	('tfidf', TfidfTransformer()),
	('classifier', MultinomialNB())
	])


pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test,predictions))

