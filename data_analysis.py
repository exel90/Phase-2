# Import aller benötigten Module
import numpy as np
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from functools import reduce
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def main():

    # Import der Daten
    df = pd.read_csv('reviews.csv')
    print("Import erste 5 Zeilen:")
    print("----------------------")
    print(df.head())
    print()

    #Alles Kleinbuchstaben
    lower_text = df['Review Text'].str.lower()

    # Entfernen von Sonderzeichen mit Hilfe von Regex
    pattern = r'[^\w\s]'
    removed_special = lower_text.replace(pattern, '', regex=True)

    # Konvertieren der Spalte zu einem String
    removed_special = removed_special.astype(str)

    #Tokensierung jedes Wortes
    tokenized = removed_special.apply(word_tokenize)
    print("tokenized erste 5 Zeilen:")
    print("----------------------")
    print(tokenized.head())
    print()

    #Entfernung von Stop-Wörtern
    stop_words = set(stopwords.words('english'))
    stop_words_removed = tokenized.apply(lambda x: [item for item in x if item not in stop_words])

    print("stop_words_removed erste 5 Zeilen:")
    print("----------------------")
    print(stop_words_removed.head())
    print()

    #Lemmatisierung
    lmtzr = WordNetLemmatizer()
    lemmatized = stop_words_removed.apply(lambda lz:[lmtzr.lemmatize(z) for z in lz])

    print("lemmatized erste 5 Zeilen:")
    print("----------------------")
    print(lemmatized.head)
    print()

    #Umwandlung in String
    lemmatized_str = lemmatized.astype(str)

    #Bag of Words
    vect = CountVectorizer()
    bag_of_words = vect.fit_transform(lemmatized_str)
    bag_of_words = pd.DataFrame(bag_of_words.toarray(),columns=vect.get_feature_names_out())
    print("Bag of Words erste 5 Zeilen:")
    print("----------------------------")
    print(bag_of_words.head())
    print()

    #TF-IDF
    vect = TfidfVectorizer(min_df=1)
    model = vect.fit_transform(lemmatized_str)
    tf_idf=pd.DataFrame(model.toarray(),columns=vect.get_feature_names_out())
    print("TF IDF erste 5 Zeilen:")
    print("----------------------")
    print(tf_idf.head())
    print()


    #LSA
    LSA_model = TruncatedSVD(algorithm='randomized',n_components=5)
    lsa = LSA_model.fit_transform(model)

    # Ausgabe der 5 fünf wichtigsten Wörter für die 5 Topics
    terms = vect.get_feature_names_out()
    print("LSA:")
    print("----")
    for i, comp in enumerate(LSA_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        for t in sorted_terms:
            print(t[0]," ")
        print("")
    print()

    #LDA
    lda_model=LatentDirichletAllocation(n_components=5,learning_method='online',random_state=42)
    lda_top=lda_model.fit_transform(model)

    # Ausgabe der 5 fünf wichtigsten Wörter für die 5 Topics
    vocab = vect.get_feature_names_out()

    print("LDA:")
    print("----")
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
        print("Topic "+str(i)+": ")
        for t in sorted_words:
            print(t[0],end=" ")
        print("\n")

    # Coherence Score
    # Dictionary erstellen

    word2id = Dictionary(lemmatized)
    print("Dictionary:")
    print("-----------")
    print(word2id)
    print()

    # Get the top words for each topic from the components_ attribute
    top_words = []
    feature_names = [word2id[i] for i in range(len(word2id))]
    n_top_words = 20

    for topic in lda_model.components_:
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])


    cm = CoherenceModel(topics=top_words, 
                        texts=lemmatized,
                        coherence='c_v',  
                        dictionary=word2id)
    score = cm.get_coherence()
    print("Coherence Score:")
    print("----------------")
    print(score)
    print()

if __name__ == "__main__":
    main()