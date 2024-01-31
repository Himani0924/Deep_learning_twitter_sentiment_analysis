 
**Abstract**

The purpose of the model is to analyze tweets and predict whether it expresses positive or negative sentiment. The model aims to implement and evaluate Naive Bayes model for classification and LSTM model for deep learning, and then select the better performing model for the client. A local client wants to implement a sentiment analysis model utilizing the tweets obtained about the product. This will help them detect angry customers or negative emotions before they escalate. The data was obtained from the Kaggle website. The dataset had 1.6 million tweets extracted using the twitter api.  Since the dataset was too huge to work on a regular machine, I trimmed the dataset to 1/4th of its original size. The data balance was maintained while trimming the data. The data was transformed and evaluated ysing Naive Bayes and LSTM model. LSTM mode was concluded to be the best model.

**Design**

The purpose of the model is to analyze tweets and predict whether it expresses positive or negative sentiment. The model aims to implement and evaluate Naive Bayes model for classification and LSTM model for deep learning, and then select the better performing model.

**Data**

The data was obtained from the kaggle website - https://www.kaggle.com/datasets/kazanova/sentiment140 . The dataset has 1.6 million tweets extracted using the twitter api.  Since the dataset was too huge to work on a regular machine, I trimmed the dataset to 1/4th of its original size. The data balance was maintained while trimming the data.

The data was cleaned and preprocessed using NLTK. I used TweetTokenizer to tokenize the data and split the text based on various criterions well suited for tweets. I used WordNetLemmatizer to lemmatize the text. WordCloud was used to show frequencies of positive and negative words.

**Algorithms**

•	Data preprocessing:

o	Tokenization (TweetTokenizer)

o	Lemmatization (WordNetLemmatizer)

o	Cleaning

o	Removal of stop words

o	Custom fine-tuning

o	Transforming into dictionary structure

o	Visualizing (WordCloud) 


•	Naive Bayes Model

o	Naive Bayes Classifier


•	LSTM Model:

o	Word Embeddings using GloVe

o	Data Padding

o	Sequential model 
	
  Embedding layer 
	
  Pair of Bidirectional LSTMs
	
  Sigmoid layer
	
  Binary cross-entropy loss function
	
  ‘Adam’ optimizer
	
  Accuracy metric

  
o	Training Model

	Batch size of 20 and epoch size of 20

o	Dropout

	LSTM model with a dropout rate of 40%

	Two rounds of training 

o	Inspecting unknown words and further data cleaning

o	Model built and trained on cleaned data


**Tools**

•	Pandas and Numpy: For cleaning data and preprocessing.

•	Keras, NLTK, scikit-learn: For processing data, Naive Bayes classification, GloVe embedding, LSTM

•	Matplotlib, Wordcloud: For visualizing the data.

**Communication**

The information is presented in the pdf – ‘Twitter Sentiment Analysis’.

