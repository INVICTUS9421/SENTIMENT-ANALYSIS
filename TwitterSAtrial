from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@user today's hot  @ home 😢 https://www.google.com/"
 #preprocessing tweet
 #create empty list
tweet_words = []

for word in tweet.split(' '):
    if word.startswith("@") and len(word) > 1:
        word = '@user'
    elif word.startswith("http") or word.startswith('https'):
        word = "http"
    tweet_words.append(word)
    
tweet_words = " ".join(tweet_words) #joining them 
print(tweet_words)

#load  model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
#downloading model 
model = AutoModelForSequenceClassification.from_pretrained(roberta)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(roberta)

#tokenizer.save_pretrained(model)
#model.save_pretrained(model)

labels = ['Negative','Neutral','Positive']

#sentiment Analysis
#convert into pytorch tensor
encoded_tweet = tokenizer(tweet_words, return_tensors='pt')
print(encoded_tweet)

output =  model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
print(output)
output =  model(**encoded_tweet)

#to detach numbers alone and compare the array with numpy
scores = output[0][0].detach().numpy()
print(scores)

#to convert into probability use numpy
scores = softmax(scores)
print(scores)

for i in range(len(scores)):
  l = labels[i]
  s = scores[i]
  print(l,s)
