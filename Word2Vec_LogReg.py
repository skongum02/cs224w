from gensim.models.word2vec import Word2Vec
import Data_Scraper
import snap
import Snap_Analytics
import pickle
import numpy as np

def getDataSource(forceReload=False):
    dataSourceFileName = "scoring_logReg.pkl"
    import os.path
    
    if forceReload or not os.path.isfile(dataSourceFileName) :
        Data_Scraper.load_data()
        print('before mapping')
        mapping = snap.TStrIntSH()
        G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
        
        rankedCommentsPos = []
        rankedCommentsNeg = []
        combinedNeg = []
        combinedPos = []
        
        #1000000
        commentsToTake = 5000
        
        for i in range(1, 5):
            rankedCommentData = Snap_Analytics.sort_comments(60, mapping,i)
            rankedCommentsPos.append(rankedCommentData[:commentsToTake])
            rankedCommentsNeg.append(rankedCommentData[len(rankedCommentData)-commentsToTake*4:len(rankedCommentData)])
            #combinedPos.extend(rankedCommentData[:commentsToTake/4])
            #combinedNeg.extend(rankedCommentData[len(rankedCommentData)-(commentsToTake*4)/4:len(rankedCommentData)])
        
        rankedCommentsPos.append(combinedPos)
        rankedCommentsNeg.append(combinedNeg)
        
        f = open(dataSourceFileName, 'wb')
        pickle.dump((rankedCommentsPos, rankedCommentsNeg), f)
        f.close()
        return (rankedCommentsPos, rankedCommentsNeg)
        
    else:
        print("loading from file")
        f = open(dataSourceFileName, 'rb')
        classifier = pickle.load(f)
        f.close()
        return (classifier[0], classifier[1])



def train():
    posComments, negComments = getDataSource()
    classifiers = []
    for pos, neg in zip(posComments, negComments):
        posC = [i[0].content.split() for i in pos]
        negC = [i[0].content.split() for i in neg]
        y = np.concatenate((np.ones(len(posC)), np.zeros(len(negC))))
        xTrain = np.concatenate((posC, negC))
        #Initialize model and build vocab
        imdb_w2v = Word2Vec(xTrain)
        print("Build vocab")
        imdb_w2v.build_vocab(xTrain)

        #Train the model over train_reviews (this may take several minutes)
        print("Training")
        imdb_w2v.train(xTrain)
        return imdb_w2v
        