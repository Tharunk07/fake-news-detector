import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class Predictor():
    def __init__(self) -> None:
        self.port_stem=PorterStemmer()
        self.vectorizer=pickle.load(open("vectorizer.pkl","rb"))
        self.predictor=pickle.load(open("model.pkl","rb"))
        
    def stemming(self,content):
        stemmed_content=re.sub("[^a-zA-Z]"," ",content)
        stemmed_content=stemmed_content.lower()
        stemmed_content=stemmed_content.split()
        stemmed_content=[self.port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
        return stemmed_content

    def joining(self,content):
        return " ".join(content)