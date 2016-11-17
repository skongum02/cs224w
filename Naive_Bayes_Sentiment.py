from nltk.classify import NaiveBayesClassifier
import csv
import os
import Snap_Analytics
import pickle
import Data_Scraper
import snap


namesOfScoreSystems = ["degree", "maxdepth", "treesize", "upvotes", "combined"]

def loadClassifier():
    f = open('nbClassifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

def createDictionary(sent):
    bagOfWords = sent.strip("\n").split(" ")
    return dict([(word, True) for word in bagOfWords])

# Structure is a tuple of dictionary and label
def loadTrainingTweet():
    bOW = []
    with open('training.txt', 'r') as f:
        for line in f:
            splittedLine = line.split('\t')
            dict = createDictionary(splittedLine[1])
            bOW.append((dict, splittedLine[0]))
    return bOW
    
def loadTwitterData():
    bOW = []
    with open('training.twitter.csv', 'rb') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='"')
        for line in csvreader:
            dict = createDictionary(line[5])
            label = line[0]
            if label == '4':
                label = '1'
            bOW.append((dict, label))
    return bOW
    
def loadMovieReviewData():
    baseDir = './aclimdb/train/'
    posDir = baseDir + 'pos/'
    negDir = baseDir + 'neg/'
    bow = []
    for filename in os.listdir(posDir):
        with open(posDir + filename, 'r') as f:
            data = f.read().replace('\n', '')
            dict = createDictionary(data)
            bow.append((dict, '1'))
    for filename in os.listdir(negDir):
        with open(negDir + filename, 'r') as f:
            data = f.read().replace('\n', '')
            dict = createDictionary(data)
            bow.append((dict, '0'))
    return bow
    
    
def convertCommentsListToTuple(data, cat):
    return [(createDictionary(i.content), cat ) for i in data]
    
    
def createClassifier():
    Data_Scraper.load_data()
    print('before mapping')
    mapping = snap.TStrIntSH()
    G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
    
    rankedCommentsPos = []
    rankedCommentsNeg = []
    combinedNeg = []
    combinedPos = []
    
    commentsToTake = 20000
    
    for i in range(4):
        rankedCommentData = Snap_Analytics.sort_comments(200, mapping,i)
        rankedCommentsPos.append(rankedCommentData[:commentsToTake])
        rankedCommentsNeg.append(rankedCommentData[len(rankedCommentData)-commentsToTake:len(rankedCommentData)])
        combinedPos.extend(rankedCommentData[:commentsToTake/4])
        combinedNeg.extend(rankedCommentData[len(rankedCommentData)-commentsToTake/4:len(rankedCommentData)])
    
    rankedCommentsPos.append(combinedPos)
    rankedCommentsNeg.append(combinedNeg)
    
    for i,n in enumerate(namesOfScoreSystems):
        posData = convertCommentsListToTuple(rankedCommentsPos[i], '1')
        negData = convertCommentsListToTuple(rankedCommentsNeg[i], '0')
        classifier =  NaiveBayesClassifier.train(posData + negData)
        f = open(n + '_nbClassifier.pkl', 'wb')
        pickle.dump(classifier, f)
        f.close()

def loadClassifiers():
    classifiers = []
    for n in namesOfScoreSystems:
        f = open(n + '_nbClassifier.pkl', 'rb')
        classifier = pickle.load(f)
        f.close()
        classifiers.append(classifier)
    return classifiers


#"<EOS> hey ted go shit and piss in your pants for 7 days because you were drafted for the vietnam war <EOS> you know the war where guns are used that your are a big supporter of <EOS> you giant hulking piece of shit and primary example of why selective breeding should be in place <EOS>"

#"<EOS> i know my republican family members are actually praising him for this whole thing because it shows he s no-nonsense <EOS> yeah the mental gymnastics people put themselves through to unconditionally support their party can be shocking <EOS>"

#"<EOS> something which contributes to the waste of billions of taxpayer dollars throws hundreds of thousands of non-violent innocents in jail scarring them with a record and breeds gang and cartel violence and unscrupulous black markets moving millions of dollars a day <EOS> yeah real insignificant <EOS>"



