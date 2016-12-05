from gensim.models.word2vec import Word2Vec
import Data_Scraper
import snap
import Snap_Analytics

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
    y = np.concatenate((np.ones(len(posComments)), np.zeros(len(negComments))))
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)
        