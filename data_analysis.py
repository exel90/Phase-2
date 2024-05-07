# Import aller benötigten Module
import numpy as np
import nltk.corpus
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

# Import der Daten
df = pd.read_csv('reviews.csv')

# In diesem Schritt wird das Beispielkorpus definiert und der Text anschließend mit den
# Methoden aus den im ersten Schritt erwähnten Python-Modulen bereinigt. Nach der Tokenisierung
# der Sätze wird mit Hilfe der Funktionen w.lower() und w.isalpha() der Text
# in Kleinbuchstaben umgewandelt und geprüft, ob die Zeichenkette einen Text darstellt, so
# dass die Interpunktion entfernt werden kann. Das Vokabular, das sich aus den im Korpus
# vorkommenden Einzelwörtern zusammensetzt, wird erstellt und gedruckt. (Script S.42)

#Umwandlung in Kleinbuchstaben
df['Review Text'] = df['Review Text'].str.lower()

#Entfernen von Sonderzeichen
# Define the regular expression pattern
pattern = r'[^\w\s]'

# Entfernen von Sonderzeichen mit Hilfe von Regex
df['Review Text'] = df['Review Text'].replace(pattern, '', regex=True)

# Konvertieren der Spalte zu einem String
df['Review Text'] = df['Review Text'].astype(str)

#Anwenden der `word_tokenize()` Funktion
#df = df['Review Text'].apply(word_tokenize)

# create the transform
vect = CountVectorizer()
# tokenize and build vocab
bag_of_words = vect.fit_transform(df['Review Text'])
#encode document
bag_of_words = pd.DataFrame(bag_of_words.toarray(),columns=vect.get_feature_names_out())
print("Bag of Words erste 5 Zeilen:")
print("----------------------------")
print(bag_of_words.head())

#TF-IDF
vect = TfidfVectorizer(min_df=1)
model = vect.fit_transform(df['Review Text'])
tf_idf=pd.DataFrame(model.toarray(),columns=vect.get_feature_names_out())
print("TF IDF erste 5 Zeilen:")
print("----------------------")
print(tf_idf.head())
# print("TF-IDF Shape")
# print(tf_idf.shape)

#LSA
LSA_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=10)
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

#LDA
lda_model=LatentDirichletAllocation(n_components=5,learning_method='online',random_state=42,max_iter=1)
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


'''
1. Module importieren
2. CSV Dateien importieren
3. alles in Kleinbuchstaben
4. Sonderzeichen entfernen
5. Konvertierung in numerische Vektorren mit Bag of Words (Pandas und Sklearn) und TF-IDF (scikit-learn)
6. Extraktion, der am häufigsten vorkommenden Themen werden die Latente Semantische Analyse (LSA) und die Latent Dirichlet Allocation (LDA) Methode (Sklearn-Pakets)
'''

'''
Beschreibe dein Datenset (welche Feature werden verwendet)
Ergänze deine Pre-Processing-Pipeline wie Stopwortfilter, N-Grams, Min/Max von Termen
Schaue dir ebenfalls im Course Feed das Thema Coherence Score an.
'''