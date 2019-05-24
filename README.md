# FakeNews_Detection
A Fake News detector building a classifier by learning patterns of US news using "doc2vec" model

Fake News Detection

Overview 

The topic of fake news detection on social media has recently attracted tremendous attention. The basic countermeasure of comparing websites against a list of labeled fake news sources is inflexible, and so a machine learning approach is desirable. Our project aims to use Natural Language Processing to detect fake news directly, based on the text content of news articles.

Problem Definition 

Develop a machine learning program to identify when a news source may be producing fake news. We aim to use a corpus of labeled real and fake new articles to build a classifier that can make decisions about information based on the content from the corpus. The model will focus on identifying fake news sources, based on multiple articles originating from a source. Once a source is labeled as a producer of fake news, we can predict with high confidence that any future articles from that source will also be fake news. Focusing on sources widens our article misclassification tolerance, because we will have multiple data points coming from each source. The intended application of the project is for use in applying visibility weights in social media. Using weights produced by this model, social networks can make stories which are highly likely to be fake news less visible.

Dataset Description

<ul>
	<li>train.csv: A full training dataset with the following attributes:</li>
	<ul>
		<li>id: unique id for a news article</li>
		<li>title: the title of a news article </li>
		<li>author: author of the news article	</li>
		<li>text: the text of the article; could be incomplete </li>
	</ul>
	<li>label: a label that marks the article as potentially unreliable</li>
	<ul>
		<li>unreliable</li>
		<li>reliable</li>
	</ul>
	
	<li>test.csv: A testing training dataset with all the same attributes at train.csv without the label.</li>
</ul>
	 


Requirements:-
<ul>
	<li>*python 3.6+</li> 
	<li>*numpy </li>
	<li>*gensim </li>
	<li>*pandas </li>
	<li>*matplotlib </li>
	<li>*scikitplot </li>
	<li>*sklearn</li> 	
	<li>*nltk</li>  
</ul>
