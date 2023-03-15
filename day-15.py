# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:40:18 2023

@author: """
import nltk as nt
import pandas as pd
from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)
fdist.most_common()
import matplotlib.pyplot as plt
fdist.plot(30)
plt.show()
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))
stop_words.update(['.','?'])
print(stop_words)


filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_word)
len(filtered_word)
len(tokenized_word)

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
ps = PorterStemmer()

stemmed_words=[]
for w in filtered_word:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_word)
print("Stemmed Sentence:",stemmed_words)
filtered_word.append('welcomed')
wl=WordNetLemmatizer()
lem_words=[]
for w in filtered_word:
    lem_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_word)
print("Stemmed Sentence:",lem_words)

nt.pos_tag(lem_words)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
#Convert all the required text into a single string here 
#and store them in word_string
#you can specify fonts, stopwords, background color and other options
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white',width=1200,height=1000).generate(' '.join(tokenized_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()





from sklearn.datasets import load_iris
iris_data=load_iris()   #loading iris dataset from sklearn.datasets

iris_df = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(iris_df)

print(kmeans.cluster_centers_)

print(kmeans.cluster_centers_) #display cluster centers

plt.scatter(iris_df[y_kmeans   == 0, 0], iris_df[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(iris_df[y_kmeans   == 1, 0], iris_df[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(iris_df[y_kmeans   == 2, 0], iris_df[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')   #Visualising the clusters - On the first two columns

plt.scatter(kmeans.cluster_centers_[:,   0], kmeans.cluster_centers_[:,1],s = 100, c = 'black', label = 'Centroids')   #plotting the centroids of the clusters

plt.legend()

plt.show()

"""Dumping the model"""   #(saving our models)
#joblib
from sklearn.externals import joblib
joblib.dump(dt,"mydt.model",compress=5)
#calling model
model=joblib.load("mydt.model")
model.predict(x_test)
model.close()


"""pickle"""
import pickle
mymodel=open("RF.model",'wb')
pickle.dump(dt,mymodel)
m=pickle.load("RF.model")



mymodel=open("RF.model",'rb')

m=pickle.load(mymodel)


