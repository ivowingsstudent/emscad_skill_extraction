from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from statistics import mean
import pickle
import numpy as np

from tqdm import tqdm

class Thesis_Helper():

    def __init__(self, spacy_model='en_core_web_lg'):
        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        self.nlp = spacy.load(spacy_model)
        self.seed = 456
        #Set self.processor_mode to -1 to use all processors
        self.processor_mode = None
        self.iter = 10000000000000000000000

        self.LR = LogisticRegression(solver='lbfgs', max_iter=self.iter,random_state=self.seed, n_jobs=self.processor_mode)

        self.GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=self.seed)

        self.SGD = SGDClassifier(loss="hinge", penalty="l2", random_state=self.seed, n_jobs=self.processor_mode)

        self.RF = RandomForestClassifier(random_state=self.seed, n_jobs=self.processor_mode)

        self.SVM = svm.SVC(decision_function_shape='ovo',random_state=self.seed)

        self.MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=self.seed,max_iter=self.iter)

    def store_object(self,name,variable):
        pickle_out = open((name+".pickle"),"wb")
        pickle.dump(variable, pickle_out)
        pickle_out.close()

    def load_object(self,name):
        pickle_in = open((name+".pickle"),"rb")
        pickle_object = pickle.load(pickle_in)
        return pickle_object

      #Function to retrieve the dependencies of the nouns
    def dep_tagger(self,text):
        doc = self.nlp (str(text))
        text = ' '.join([token.dep_ for token in doc])
        return text

    #Function to retrieve the part of speech
    def pos_tagger(self,text):
        doc = self.nlp (str(text))
        text = ' '.join([token.pos_ for token in doc])
        return text

    #Function to remove stopwords
    def stop_word_remover(self,text):
        doc = self.nlp (str(text))
        text = ' '.join([token.text for token in doc if token.is_stop == False])
        return text

    #Function to lemmatize
    def lemmatizer(self,text):
        doc = self.nlp (str(text))
        text = ' '.join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])
        return text

    def sequence_counter(self,text):
        from collections import Counter
        doc = self.nlp(str(text))
        tags = []
        for tag in doc:
            tags.append(tag.text)
        return Counter(tags)



    # #Function which loads a ngram with context df and transforms the 3 columns into one
    def load_context_df(self,path,sep):
        df = pd.read_csv(path,sep)
        df['allgrams'] = df['left_context'].str.lower() +' ' +df['recovered_gram'].str.lower() +' ' +df['right_context'].str.lower()
        df = df[['allgrams', 'label']]
        return df

    def performance(self,model,x,y):
        print('Precision ',precision_score(y,model.predict(x)))
        print('Recall ',recall_score(y,model.predict(x)))
        print('F1_Score ',f1_score(y,model.predict(x)))

    #Functions which cross validates 6 classifiers and prints the average results
    def model_performance(self,x,y):

        #Setting the datatype of x to np.float 64 in order to remove warnings
        x = x.astype(np.float64)

        #Creating a dictionary which stores the classifiers
        classifiers = {'LR': self.LR,
                       'GBC':self.GBC,
                       'SGD': self.SGD,
                       'RF': self.RF,
                       'SVM':self.SVM,
                       "MLP":self.MLP}

        #Creating empty list to store the values
        model_name = []
        precision_scores=[]
        recall_scores=[]
        f1_scores=[]

        print("Starting model evaluation")
        for model in tqdm(classifiers):

            performance_measures = ['precision_macro', 'recall_macro','f1_macro']
            scores = cross_validate(classifiers[model], x, y, scoring=performance_measures,cv=10,verbose=1, n_jobs=-1)

            average_precision = mean(scores['test_precision_macro'])
            average_recall = mean(scores['test_recall_macro'])
            average_f1 = mean(scores['test_f1_macro'])

            model_name.append(model)
            precision_scores.append(average_precision)
            recall_scores.append(average_recall)
            f1_scores.append(average_f1)

        results = pd.DataFrame(zip(model_name,precision_scores,recall_scores,f1_scores),
        columns=['Classifier','Precision', 'Recall', 'F1'])

        #results = results.sort_values(by='F1',ascending=False)

        return results

    def confusion_matrix(self,model,x,y):

        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=self.seed)

        plot_confusion_matrix(model.fit(x_train,y_train), x_test, y_test,display_labels=['not skill', 'soft skill', 'hard skill'])
        plt.show()


    def training_graph(self,model,x,y,x_axis_size=500):

        precision_scores=[]
        recall_scores=[]
        f1_scores=[]
        training_size=[]

        for training_amount in range(x_axis_size,(len(x)+x_axis_size),x_axis_size):

            X = x[0:training_amount]
            Y = y[0:training_amount]

            x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2,random_state=seed)

            model.fit(x_train, y_train)

            precision_scores.append(model.precision_score(y_test, model.predict(x_test), average='macro'))
            recall_scores.append(recall_score(y_test, model.predict(x_test), average='macro'))
            f1_scores.append(f1_score(y_test, model.predict(x_test), average='macro'))
            training_size.append(training_amount)


        plt.plot(training_size,precision_scores)
        plt.plot(training_size,recall_scores)
        plt.plot(training_size,f1_scores)

        plt.ylabel('Score')
        plt.xlabel('Training size')
        plt.title('Training graph')

        plt.legend(['Precision', 'Recall', 'F1'])
        plt.show()

    def sentence_creator(self,text):
        sentences = []

        doc = self.nlp(text)

        for sentence in tqdm(doc.sents):
            sentences.append(str(sentence))

        df = pd.DataFrame(zip(sentences),columns=['sentence'])


        df['sentence'] = df['sentence'].str.lower()
        df['sentence'] = df['sentence'].str.replace(r'\n\n', ' ',regex=False)
        df['sentence'] = df['sentence'].str.replace(r'\n', '. ',regex=False)
        df['sentence'] = df['sentence'].str.replace(r'[^A-Za-z0-9 ]+', ' ',regex=True)
        df['sentence'] = df['sentence'].str.replace(r'  ', ' ',regex=False)
        df['sentence'] = df['sentence'].str.strip()

        return df

    def ngram_creator(self,dataframecolumn,gramsize):
        from nltk.util import ngrams
        from nltk.tokenize import RegexpTokenizer

        #Creating all the possible grams for each sentence
        allgrams = []

        #This tokenizer immediately removes punctuation and special characters from the sentence
        tokenizer = RegexpTokenizer(r'\w+')

        for sentence in tqdm(dataframecolumn):
            tokenizedsentence = tokenizer.tokenize(str(sentence))

            #getting up to tri grams for each sentence
            for n in range(1,gramsize+1):
                grams = ngrams(tokenizedsentence,n)
                for gram in grams:
                    allgrams.append(str(gram))

        ngrams = pd.DataFrame(allgrams,columns=['ngrams'])

        ngrams['ngrams'] = ngrams['ngrams'].astype(str)
        ngrams['ngrams'] = ngrams['ngrams'].str.replace(r"[(),.']", '',regex=True)
        ngrams['ngrams'] = ngrams['ngrams'].str.strip()

        ngrams = ngrams.drop_duplicates()

        return ngrams

    def print(self,something):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(something)

    def folder_reader(self,path):
        import pandas as pd
        import glob
        all_files = glob.glob(path + "/*.csv")
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=True)
        return frame
