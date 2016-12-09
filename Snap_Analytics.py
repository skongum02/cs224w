import Data_Scraper
import snap
import numpy
import matplotlib.pyplot as plt
import math
import copy
import pickle
from collections import deque
import re
import random

### READ ME ###
#--------------------------------#
# Data_Scraper structures that are useful for project:
# Data_Scraper.all_comments		Array of all comment objects
# Data_Scraper.root_comments	Array of all comment objects that replied directly to threads
# Data_Scraper.thread_ids		Set of all Thread ID strings
# Data_Scraper.comment_edges	Array of tuples where val1 is the parent comment ID and
#								val2 is the child comment ID
# Data_Scraper.comment_id_lookup	Map object that links comment ID strings to their objects

# The comment_object is a struct that is modeled as so:
#	time_stamp: An int that represents the time this comment was posted
#	comment_id: A unique string for this comment
#	parent_comment_id: The unique string for this comment's parent
#	author_name: The string username of the comment's author
#	score: An int representing the total score of  comment from up and downvotes
#	thread_id: A string for this comment's entire thread
#	content: The string that represents the actual text of the comment
#--------------------------------#

#returns a vector that contains the normalized degree, tree size, and depth of each comment in the dataset

def get_comment_from_nid(nid, mapping):
	comment_name = mapping.GetKey(nid)
	comment_obj = Data_Scraper.comment_id_lookup[str(comment_name)]
	return comment_obj



#sorts statistics of comments by degree (pv=1), maxdepth (pv=2), treesize (pv=3), or upvotes (pv=4)
#comment_stats2[i] = [node ID, normalized degree, normalized maxdepth, normalized treesize, normalized upvotes, threadsize]
def sort_comments(cutoff, mapping, pv):
	print("before open pickle")
	pkl_file = open('comment_stats.pkl', 'rb')
	print("after open pickle")
	comment_stats = pickle.load(pkl_file)
	pkl_file.close()
	print("after load pickle")
	comment_stats = numpy.array(comment_stats)
	comment_stats2 = []
	for i in range(len(comment_stats)):
		#print comment_stats[i][4]
		if(comment_stats[i][5] >= cutoff):
			#numpy.delete(comment_stats,i,0)
			comment_stats2.append(comment_stats[i])
	#comment_stats2 = []
	#for c in comment_stats:
	#	comment_stats2.append(c)
	#print("before sort")
	#print comment_stats2[0]
	comment_stats2 = sorted(comment_stats2, key=lambda x: x[pv], reverse=True)
	#print comment_stats2[2]
	comment_name = mapping.GetKey(int(comment_stats2[2][0]))
	comment_obj = Data_Scraper.comment_id_lookup[comment_name]
	#print(comment_obj.content)
	#print('root comments in large threads = ' + str(len(comment_stats2)))
	return [(Data_Scraper.comment_id_lookup[mapping.GetKey(int(comment_stats2[i][0]))], comment_stats2[i][pv] )for i in range(len(comment_stats2))]



def comment_statistics(mapping, g):
	stats_vec = []
	root = getRootNode(mapping, g)
	for thread in root.GetOutEdges():
		#print(thread)
		threadsize = _findDepth(g.GetNI(thread), g)
		for n in g.GetNI(thread).GetOutEdges():
			deg = (g.GetNI(n).GetDeg()-1) /float(threadsize)
			maxdepth = getMaxDepth(g,n)/float(threadsize)
			treesize = _findDepth(g.GetNI(n),g)/float(threadsize)
			upvotes = get_comment_from_nid(n, mapping).score/float(threadsize)
			stats_vec.append([n,deg,maxdepth,treesize,upvotes,threadsize])
	return stats_vec


# Helper function, not to be called directly
def maxDepthRecur(G, nid):
	depthvec = [0]
	for reply in G.GetNI(nid).GetOutEdges():
		depthvec.append(maxDepthRecur(G,reply))
	return max(depthvec)+1

def getMaxDepth(G, nid):
	return maxDepthRecur(G, nid)-1


# Helper function that counts the number of words (not unique) in a string
def count_words(content):
	content = content.split()
	content = filter(lambda x: "." not in x and "\n" not in x, content)
	return len(content)

# Creates histogram of comment lengths without periods
def measure_comment_lengths():
	print "Measuring comment lengths..."
	comment_length_count = {}


	for comment in Data_Scraper.all_comments:
		content = comment.content
		content = content.split()
		content = filter(lambda x: "." not in x and "\n" not in x, content)
		if len(content) in comment_length_count:
			comment_length_count[len(content)] += 1
		else:
			comment_length_count[len(content)] = 1

	# length_array = []
	# for key,val in comment_length_count.iteritems():
	# 	for i in xrange(val):
	# 		length_array.append(key)
	# print length_array
	# print "Comment Length standard deviation: " + str(numpy.std(length_array))
	# print "Comment Length mean: " + str(numpy.mean(length_array))

	# return

	plt.figure(0)
	# plt.hist(length_histogram)
	X = [key for key,value in comment_length_count.iteritems()]
	Y = [value for key,value in comment_length_count.iteritems()]




	plt.loglog(X,Y,"bo")
	plt.title("Comment Lengths")
	plt.ylabel("Frequency")
	plt.xlabel("Comment Length")
	plt.show()

	# return comment_length_count
def getFKScores():

	# READ FROM PICKLE FILE
	f = open("FK_reading_ease_scores.pkl", "rb")
	FK_scores = pickle.load(f)
	f.close()
	return FK_scores

def get_FK_histogram():
	# READ FROM PICKLE FILE
	f = open("FK_reading_ease_scores.pkl", "rb")
	FK_scores = pickle.load(f)
	f.close()
	return FK_scores

def FK_histogram():
	from textstat.textstat import textstat
	print "Measuring Flesch-Kincaid scores..."
	FK_scores = {}
	counter = 0
	f = open("FK_reading_ease_scores.pkl", "wb")
	for comment in Data_Scraper.all_comments:
		counter += 1
		if counter %100000 == 0:
			print str(counter) + " comments done"
		content = comment.content
		content = content[3:].replace(" .", ".")
		if not re.search('[a-zA-Z]', content):
			continue

	# 	""" DO THIS TOO: grade_level = textstat.text_standard(test_data) """
		fk_score = math.floor(textstat.flesch_reading_ease(content))
		FK_scores[comment.comment_id] = fk_score
		# f.write(comment.comment_id + "\t" + str(fk_score) + "\n")
	pickle.dump(FK_scores, f)
	f.close()
	print "File written"

	return

	# 	if fk_score in FK_scores:
	# 		FK_scores[fk_score] += 1
	# 	else:
	# 		FK_scores[fk_score] = 1

	# f = open("FK_reading_ease_scores.txt", "w")
	# for key,value in FK_scores.iteritems():
	# 	f.write(str(key) + "\t" + str(value) + "\n")
	# f.close()
	# print "File written"



	# READING THE PAGE BACK IN
	FK_scores = {}

	f = open("FK_reading_ease_histogram.txt", "r")
	lines = f.readlines()
	for line in lines:
		tup = line.split('\t')
		print tup
		# To account for the '\n' at the end of each line
		x = math.floor(float(tup[0]))
		if x not in FK_scores:
			FK_scores[x] = 0
		FK_scores[x] += int(tup[1].rstrip('\n'))
	f.close()
	print FK_scores

	# FK_array = []
	# for key,val in FK_scores.iteritems():
	# 	if key in range(0, 150):
	# 		for i in xrange(val):
	# 			FK_array.append(key)

	# print "FK standard deviation: " + str(numpy.std(FK_array))
	# print "FK mean: " + str(numpy.mean(FK_array))
	# return

	plt.figure(1)
	# plt.hist(length_histogram)
	X = [key for key,value in FK_scores.iteritems()]
	Y = [value for key,value in FK_scores.iteritems()]
	plt.bar(X,Y)
	plt.xlim((0, 200))
	plt.title("Flesch-Kincaid Reading Ease of Comments")
	plt.ylabel("Frequency")
	plt.xlabel("Flesch-Kincaid Reading Ease Score")
	plt.show()
	# return FK_scores



def getRootNode(mapping, g):
	rootId = mapping.GetKeyId("root")
	root = g.GetNI(rootId)
	print("The number of threads: {0}".format(root.GetOutDeg()))
	return root
	
def getCommentHistogram(firstLevelNodes, g):
	hist = []
	for node in firstLevelNodes:
		commentsInThread = _findDepth(node, g)
		hist.append(commentsInThread)
	print("Number of comments of total comments is {0}".format(len(hist)))
	print("mean {0}, stddev {1}".format(numpy.mean(hist), numpy.std(hist)))
	#plt.hist(hist, log=True)
	#plt.show()
	return hist

#BFS
def _findDepth(node, g):
	nodes = [node]
	totalNodes = []
	while len(nodes) > 0:
		currentNode = nodes[0]
		totalNodes.append(currentNode)
		nodes.pop(0)
		children = [g.GetNI(i) for i in currentNode.GetOutEdges()]
		nodes.extend(children)
	return len(totalNodes)
		

# newRoot should be the actual node ID in g.
# newRoot = mapping.GetKeyId(orig_id)


def makeSubGraph(g, newRoot):

	newRoot_NI = g.GetNI(newRoot)

	newRoot_graph = snap.TNGraph.New()
	newRoot_graph.AddNode(newRoot)

	queue = deque([newRoot_NI])

	while len(queue) > 0:
		NI = queue.popleft()
		for nid in NI.GetOutEdges():
			newRoot_graph.AddNode(nid)
			queue.append(g.GetNI(nid))
			newRoot_graph.AddEdge(NI.GetId(), nid)

	return newRoot_graph


def sample_random_comment(g, mapping):
	index = random.randint(0, len(Data_Scraper.all_comments)-1)
	comment = Data_Scraper.all_comments[index]
	content = comment.content
	fk_score = math.floor(textstat.flesch_reading_ease(content))
	if fk_score in range(50, 90):
		sample_random_comment(g, mapping)
		return
	comment_words = filter(lambda x: "." not in x and "\n" not in x, content.split())
	print # blank line
	print "FK Ease of Reading: " + str(fk_score)
	print "Number of words: " + str(len(comment_words))
	print content


def main():
	print('begin main')
	Data_Scraper.load_data()
	#measure_comment_lengths()
	
	# measure_comment_lengths()
	# FK_histogram()

	
	print('before mapping')
	
	#for i in xrange(25):
	#	sample_random_comment(G, mapping)

	root = getRootNode(mapping, G)

	# FK_histogram()

	FK_scores = getFKScores()

	# """ TRYING TO CREATE A SMALL GRAPH SAMPLE FOR VISUALIZATION """
	# f = open("Mini_graph.csv", "w")
	# thread_id = random.sample(Data_Scraper.thread_ids, 1)
	# newRoot = G.GetNI(mapping.GetKeyId(thread_id)).GetId()
	# # newRoot = mapping.GetKeyId(thread_id)
	# f.write(str(root.GetId()) + "," + str(newRoot) + "\n")
	# sg = makeSubGraph(G, mapping)
	# for edge in sg.Edges():
	# 	tup = edge.GetId()
	# 	f.write(str(tup[0]) + "," + str(tup[1]) + "\n")
	# f.close()
	# print "File written"
	
	# for key,value in FK_scores.iteritems():
	# 	f.write(str(key) + "\t" + str(value) + "\n")
	# f.close()
	# print "File written"



	#stats_vec = comment_statistics(mapping, G)
	#print(stats_vec[4])
	#getCommentHistogram([G.GetNI(n) for n in root.GetOutEdges()], G)
	#output = open('comment_stats.pkl', 'wb')
	#pickle.dump(stats_vec,output)
	#getCommentHistogram([G.GetNI(n) for n in root.GetOutEdges()], G)
	#output = open('comment_stats2.pkl', 'wb')

	#pickle.dump(stats_vec,output)
	#output.close()
	# sort_comments(200, mapping, 1)

	# pickle.dump(stats_vec,output)
	#output.close()

	# print "Nodes: " + str(Data_Scraper.all_comments)
	# print "Threads: " + str(len(Data_Scraper.thread_ids))
	# print "Root comments: " + str(len(Data_Scraper.root_comments))



	print G.GetNodes()
	print G.GetEdges()


def getDegree(G, mapping, nodeId):
	node = G.GetNI(mapping.GetKeyId(nodeId))
	return node.GetOutDeg()

def getNormalizedDegree(G, mapping, nodeId):
	node = G.GetNI(mapping.GetKeyId(nodeId))
	deg = node.GetOutDeg()
	c_object = Data_Scraper.comment_id_lookup[nodeId]
	threadsize = _findDepth(G.GetNI(mapping.GetKeyId(c_object.thread_id)), G)
	return 1.0*deg/threadsize

	
# TODO: mean and standard deviation of FK - scores and word lengths

if __name__ == "__main__":
	main()

mapping = snap.TStrIntSH()
G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
root = getRootNode(mapping, G)
thread_sizes = {}
for thread in root.GetOutEdges():
		threadsize = _findDepth(g.GetNI(thread), g)
		thread_sizes[mapping.GetKey(thread)] = threadsize





