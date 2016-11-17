from nltk.classify import NaiveBayesClassifier


def createDictionary(sent):
    bagOfWords = sent.strip("\n").split(" ")
    return dict([(word, True) for word in bagOfWords])

# Structure is a tuple of dictionary and label
def loadFileIntoStructure():
    bOW = []
    with open('training.txt', 'r') as f:
        for line in f:
            splittedLine = line.split('\t')
            dict = createDictionary(splittedLine[1])
            bOW.append((dict, splittedLine[0]))
    return bOW
    
def createClassifier():
    currentData = loadFileIntoStructure()
    classifier =  NaiveBayesClassifier.train(currentData)
    import pickle
    f = open('nbClassifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
