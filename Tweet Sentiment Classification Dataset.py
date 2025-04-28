#Step1: Unzip the dataset
path = r"D:\datasets\New To Work on 3\Tweet Sentiment Classification Dataset\Tweet Sentiment Classification Dataset.zip"
#First of all we need to unzip the dataset
import zipfile
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(r"D:\datasets\New To Work on 3\Tweet Sentiment Classification Dataset")

#Now we need to load the dataset
import pandas as pd
df = pd.read_csv(r"D:\datasets\New To Work on 3\Tweet Sentiment Classification Dataset\tweet_sentiment.csv")

#Step2: Understand the dataset
import matplotlib.pyplot as plt
import seaborn as sns
print(df.head(30))
print(df.sample(5))
print(df['sentiment'].value_counts()) # we can see it is normally distributed
print(df.info())
print(df.isna().sum())# we have no null values
#show number of tweets in each sentiment
print(sns.countplot(df['sentiment']))
plt.show()
#Step3: Clean and preprocess the dataset
#now we need to remove the stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
df['cleaned_text'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print(df.head())
#now we need to remove the special characters , URLS , HTML tags , @mentions 
import re
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', str(x)))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'<.*?>', '', str(x)))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: re.sub(r'@[A-Za-z0-9]+', '', str(x)))
print(df.head())
# now we need to remove the extra spaces
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x.split()))
print(df.head())
#now we need to lower the text
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.lower())
print(df.head())
#now we need to tokenize the text using Split
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.split())
print(df.head())
#Lets perform stemming and lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print(df.head())
#now we need to join the words back into a single string
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x))
print(df.head())

#Step4: Lets build our X and perform encoding on y and after that which class belongs to which number
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])
print(y.shape)
print(y[0:10])
X = df['cleaned_text']
print(X[0:10])

#Step5: lets apply vectorization to the text
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
print(X)

#Step6: lets split the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Step7: lets build our model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

#Step8: lets evaluate our model + using confusion matrix in heatmap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.show()


#Step9: lets save our model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))

#Step10: lets load our model
model = pickle.load(open('model.pkl', 'rb'))

#Step11: lets test our model with 3 random tweets
text1 = "I am happy"
text1 = vectorizer.transform([text1]).toarray()
print(model.predict(text1))
text2 = "I am sad"
text2 = vectorizer.transform([text2]).toarray()
print(model.predict(text2))
text3 = "I am angry"
text3 = vectorizer.transform([text3]).toarray()
print(model.predict(text3))

#Clearly my model has overfitting problem
#Step12: lets apply cross validation to our model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print(scores.mean())

#Step13: lets apply grid search to our model
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


#Step14: lets apply random forest to our model and see the performance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.show()
# lets test our model with some random tweets
text1 = "I am happy"
text1 = vectorizer.transform([text1]).toarray()
print(rf.predict(text1))
text2 = "I am sad"
text2 = vectorizer.transform([text2]).toarray()
print(rf.predict(text2))
text3 = "I am angry"
text3 = vectorizer.transform([text3]).toarray()
print(rf.predict(text3))
text4 = "I am not happy"
text4 = vectorizer.transform([text4]).toarray()
print(rf.predict(text4))
text5 = "I am not sad"
text5 = vectorizer.transform([text5]).toarray()
print(rf.predict(text5))

print(df.head(30))
text6 = " The event starts at 5 PM"
text6 = vectorizer.transform([text6]).toarray()
print(rf.predict(text6))
text7 = "today i want to go to the mall"
text7 = vectorizer.transform([text7]).toarray()
print(rf.predict(text7))

text7 = "setare is my sister"
text7 = vectorizer.transform([text7]).toarray()
print(rf.predict(text7))

