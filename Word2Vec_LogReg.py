from gensim.models.word2vec import Word2Vec
import Data_Scraper
import snap
import Snap_Analytics
import pickle
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier

Data_Scraper.load_data()

nDim = 300

def getDataSource(forceReload=False):
    dataSourceFileName = "scoring_logReg.pkl"
    import os.path
    
    if forceReload or not os.path.isfile(dataSourceFileName) :
        print('before mapping')
        Data_Scraper.load_data()
        mapping = snap.TStrIntSH()
        G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
        
        rankedCommentsPos = []
        rankedCommentsNeg = []
        combinedNeg = []
        combinedPos = []
        
        #1000000
        commentsToTake = 15000
        
        for i in range(1, 5):
            rankedCommentData = Snap_Analytics.sort_comments(60, mapping,i)
            rankedCommentsPos.append(rankedCommentData[:commentsToTake])
            rankedCommentsNeg.append(rankedCommentData[len(rankedCommentData)-commentsToTake:len(rankedCommentData)])
            #combinedPos.extend(rankedCommentData[:commentsToTake/4])
            #combinedNeg.extend(rankedCommentData[len(rankedCommentData)-(commentsToTake*4)/4:len(rankedCommentData)])
        
        #rankedCommentsPos.append(combinedPos)
        #rankedCommentsNeg.append(combinedNeg)
        print(len(rankedCommentsPos))
        
        f = open(dataSourceFileName, 'wb')
        pickle.dump((rankedCommentsPos, rankedCommentsNeg), f)
        f.close()
        return rankedCommentsPos, rankedCommentsNeg
        
    else:
        print("loading from file")
        f = open(dataSourceFileName, 'rb')
        classifier = pickle.load(f)
        print(len(classifier[0]))
        f.close()
        return classifier[0], classifier[1]



def cleanText(corpus):
    corpus = [z[0].content.decode("utf-8").lower().replace('\n','').split() for z in corpus]
    return corpus


def train(oneClassifier=False, forceReload=False):
    posComments, negComments = getDataSource(forceReload)
    classifiers = []
    for pos, neg in zip(posComments, negComments):
        posC = cleanText(pos)
        negC = cleanText(neg)
        y = np.concatenate((np.ones(len(posC)), np.zeros(len(negC))))
        xTrain = np.concatenate((posC, negC))
        #Initialize model and build vocab
        #imdb_w2v = Word2Vec(xTrain)
        
        
        imdb_w2v = Word2Vec(size=nDim, min_count = 10)
        print("Build vocab")
        imdb_w2v.build_vocab(xTrain)
        #Train the model over train_reviews (this may take several minutes)
        print("Training")
        imdb_w2v.train(xTrain)
        #imdb_w2v = Word2Vec(xTrain, size=nDim)
        trainVecs = scaleData(xTrain, nDim, imdb_w2v)
        
        
        classifiers.append((trainVecs, y, imdb_w2v))
        if oneClassifier:
            return classifiers
    return classifiers
        
#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= float(count)
    return vec

def scaleData(data, n_dim, model):
    print("Scaling data")
    #print(data[0])
    vecs = np.concatenate([buildWordVector(z, n_dim, model) for z in data])
    #print(vecs[:1])
    vecs = scale(vecs)
    return vecs
    
def predict(train_vecs, y_train, model, nDim, partial=None):
    print("Predicting ")
    Data_Scraper.load_data()
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    allData = []
    if partial == None:
        print("running all")
        allData = [i.content.split() for i in Data_Scraper.all_comments]
    else:
        allData = [i.content.split() for i in Data_Scraper.all_comments[:partial]]
    model.train(allData)
    model.init_sims(replace=True)
    allDataScaled = scaleData(allData, nDim, model)
    model.clear_sims()
    return lr.predict(allDataScaled)
    
def getAllPosComments(result):
    Data_Scraper.load_data()
    return [Data_Scraper.all_comments[i] for i,v in enumerate(result) if v > 0]
    
def getAllNegComments(result):
    Data_Scraper.load_data()
    return [Data_Scraper.all_comments[i] for i,v in enumerate(result) if v < 1]
    
def trainAndPredict(partial=None, oneClassifier=False, forceReload=False):
    datas = train(oneClassifier, forceReload)
    predictions = []
    for data in datas:
        predictions.append(predict(data[0], data[1], data[2], nDim, partial))
    return predictions
        