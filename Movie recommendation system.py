#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
get_ipython().system('pip install nltk')


# In[27]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[28]:


movies.head()


# In[29]:


credits.head()


# In[30]:


credits.head(1)['cast'].values # to get the full caste and their characters


# In[31]:


credits.head(1)['crew'].values


# ## Data Pre Processing

# In[32]:


# we will merge the dataframes
movies=movies.merge(credits,on='title')#we have merged on the basis of title of the movie


# In[33]:


credits.shape


# In[34]:


movies.shape


# In[35]:


movies.head()


# In[36]:


#now we remove the columns that wont be a part of our analysis
# following will be important 
# genres
# id will be used in the end
# keyword
# overview (summary)
# title
# cast
# crew 


# In[37]:


movies['original_language'].value_counts()


# In[38]:


#mostly english hee hai so we wont keep this
movies['spoken_languages'].value_counts()


# In[39]:


movies.info()


# In[40]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[41]:


movies.head()


# In[42]:


#I will join the strings now like merging overview with keywords genre 1st 3 cast members and the director from the crew to get a tag for each movie
# we have to perform the pre processing steps


# In[43]:


movies.isnull().sum()


# In[44]:


movies.dropna(inplace=True)#will remove all with no overview as there are only 3 such


# In[45]:


movies.duplicated().sum()


# In[46]:


#genres keywords to be written in correct format
movies.iloc[0].genres


# In[47]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}
# we have to change this into:
#['Action','Adventure','SciFi']


# In[51]:


import ast


# In[52]:


# For this we will use a conversion function from the above to a list
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):# string to list
        L.append(i['name'])#and add the names of the dictionary to the list
    return L  
#   
    


# In[53]:


movies['genres']=movies['genres'].apply(convert)


# In[ ]:





# In[54]:


movies.head()


# In[55]:


movies['keywords']=movies['keywords'].apply(convert)


# In[56]:


movies.head()


# In[57]:


movies['cast'][0]


# In[58]:


# we need only the 1st 3 dictionaries and convert them into list
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

        


# In[59]:


movies['cast']=movies['cast'].apply(convert3)


# In[60]:


movies.head()


# In[61]:


movies['crew'][0]


# In[62]:


# we need the dictionary whose job is director and we have to extract the name from it
def fetchdirector(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L        


# In[63]:


movies['crew']=movies['crew'].apply(fetchdirector)


# In[64]:


movies.head()


# In[65]:


movies['overview'][0]


# In[66]:


movies['overview'].apply(lambda x:x.split())
# with this we will get a list for each row


# In[67]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[68]:


movies.head()


# In[69]:


# have to remove whitespaces .
# we do this bcoz if we have 2 names gervit chanda and gervit agarwal . gervit would have same tags for both and model would get confused 
# so we make a single word . we do this on
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])                
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])                


# In[70]:


movies.head()


# In[71]:


# now we will make a tag columns and concatenate the others
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[72]:


movies.head()


# In[73]:


new_df=movies[['movie_id','title','tags']]


# In[74]:


new_df.head()


# In[75]:


# we will convert the list in the tags to string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[76]:


new_df.head()


# In[77]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()


# In[78]:


get_ipython().system('pip install nltk')


# In[79]:


import nltk
nltk.download('stopwords')


# In[80]:


# stop word removal and stemming
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
stopword_list = stopwords.words(fileids = 'english')


# In[81]:


def stem(text):
    word_list = text.split()
    stemmed_text = " ".join([ps.stem(word) for word in word_list if word not in stopword_list])
    return stemmed_text


# In[82]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[83]:


new_df.head()


# ## Vectorization

# In[ ]:





# In[84]:


new_df.shape


# In[85]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[86]:


vectors=cv.fit_transform(new_df['tags']).toarray()#count vectorizer humme sparse matrix dega usse hum numpy array mei convert kr denge


# In[87]:


vectors.shape


# In[88]:


vectors#now we have each movie in a vector form


# In[89]:


vectors[0]


# In[90]:


list(cv.get_feature_names_out())# we use this to get the feature names 


# In[91]:


# we have siimilar words used in a different way so we dont wanna have them as 2 different words rather a single word 
# like action and actions should be similar
# we apply stemming 
# it would convert a list like ['love','loved','loving']
# to ['love','love','love'] and similarly for others


# In[92]:


# calculate distance between each movie
# we calculate cosine(angle between them)  distance not euclidean distance
# smaller the angle the similar they are 


# ## Cosine Similarity

# In[93]:


from sklearn.metrics.pairwise import cosine_similarity


# In[94]:


similarity=cosine_similarity(vectors)#distance between each movie 


# In[95]:


cosine_similarity(vectors).shape


# In[96]:


similarity[0]


# In[97]:


similarity[0].shape


# In[98]:


sorted(list(enumerate(similarity[0])) ,reverse=True,key=lambda x:x[1])[1:6]#0th  movie saath distance
#1 wale number se sorting nhi , second wale se sorting hori hai
#in this we get the index of movies as well in descending order
#the 1st 5 similar movies are taken


# In[99]:


def recommend(movie):
    # Convert the input movie title to lowercase
    movie = movie.lower()

    # Check if the movie exists in new_df (case-insensitive)
    if movie not in new_df['title'].str.lower().values:
        print(f"The movie '{movie}' is not in the database.")
        return

    # Get the index of the movie in new_df
    movie_index = new_df[new_df['title'].str.lower() == movie].index[0]

    # Calculate movie similarities and recommend similar movies
    distances = similarity[movie_index]
    movies_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]

    print(f"Recommendations for '{movie}':")
    for i in movies_list:
        recommended_movie = new_df.iloc[i[0]].title#to get title 
        print(recommended_movie)



# In[100]:


recommend('Batman Begins')


# In[101]:


import pickle


# In[102]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[103]:


new_df['title'].values


# In[104]:


new_df.to_dict()


# In[105]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[106]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




