import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('white')

columns_names = ['user_id','item_id','rating','timestamp']

df = pd.read_csv('u.data',sep='\t',names=columns_names)

print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')
print(movie_titles.head())

df = pd.merge(df,movie_titles,on='item_id')
print(df.head())

#Sort by rating ascending 
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings)


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings)

ratings['num of ratings'].hist(bins=70)


ratings['rating'].hist(bins=70)
#plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
#plt.show()

#create a matrix 
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat)

print(ratings.sort_values('num of ratings',ascending=False).head(10))


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

print(starwars_user_ratings.head())


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
print(similar_to_starwars)

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
print(similar_to_liarliar)

#correlation 
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())


print(corr_starwars.sort_values('Correlation',ascending=False).head(10))

#filter riview with at least 100 reviews 
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars.head())

print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
print(corr_liarliar.dropna(inplace=True))

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head())