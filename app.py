import streamlit as st
import pickle
import pandas as pd
import requests
def fetch_poster(movie_id):# will request for posters from the api
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=15a72db1be97275cbf691a462784938a&language=en-US'.format(movie_id))
    data=response.json()# this is the response we get from the whose poster path we use

   # return "https://image.tmdb.org/t/p/w500/ "+data['poster_path']#full path of image
    return "https://image.tmdb.org/t/p/original"+data['poster_path']#full path of image

def recommend(movie):#gives recommended movies with their posters
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []# for movies
    recommended_movies_posters=[]# for posters

    for i in movies_list:
        movie_id=movies.iloc[i[0]].movie_id


        recommended_movies.append(movies.iloc[i[0]].title)#adding names of the recommended movies
        #fetch poster from api
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies,recommended_movies_posters
similarity=pickle.load(open('similarity.pkl','rb'))#gives all similarities from jupyter
movies_dict=pickle.load(open('movies_dict.pkl','rb'))#gives movie names in form of a dictionary from jupyter
movies =pd.DataFrame(movies_dict)
st.title('Movie Recommender System')# title of our page
selected_movie_name =st.selectbox('Select Movie of ur chocie ',
                     movies['title'].values)# making a drop down box containing titles of movies
if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)
    col1,col2,col3,col4,col5=st.columns(5)# making 5 columns with names and poster of the movies
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])#could have done this in loop as well





