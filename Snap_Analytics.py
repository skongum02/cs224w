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

def estimated_delay(nid, mapping, g):
	#comment = get_comment_from_nid(int(nid), mapping)			
	comment_name = mapping.GetKey(int(nid))
	comment = Data_Scraper.comment_id_lookup[comment_name]
	time = comment.time_stamp
	thread = comment.thread_id

	root = getRootNode(mapping, g)
	for thread in root.GetOutEdges():
		#print(thread)

		time_vec = []
		for n in g.GetNI(thread).GetOutEdges():
			c_name = mapping.GetKey(int(n))
			c = Data_Scraper.comment_id_lookup[c_name]
			t = c.time_stamp
			thread_id = c.thread_id
			if(thread_id == thread):
				time_vec.append(t)
		time_vec = numpy.array(time_vec)
		return time - min(time_vec)


def add_attr():
	for comment in Data_Scraper.all_comments:
		comment["test_attr"] = "this is a test"

def record_comment_lengths():
	length_dict = {}
	for comment in Data_Scraper.all_comments:
		content = comment.content
		content = content.split()
		content = filter(lambda x: "." not in x and "\n" not in x, content)
		length_dict[comment.comment_id] = len(content)
	#print(len(length_dict))
	return length_dict



def sequence_corr(sequences):
	corr_list = []
	for i in xrange(10):
		true_neg = 0
		true_pos = 0
		false_neg = 0
		false_pos = 0
		for seq in sequences:
			ancestors = seq[1]
			if(len(ancestors) <= i):
				continue
			if(ancestors[i] == 0 and seq[0] == 0):
				true_neg = true_neg+1
			if(ancestors[i] == 0 and seq[0] == 1):
				false_neg = false_neg+1
			if(ancestors[i] == 1 and seq[0] == 0):
				false_pos = false_pos+1
			if(ancestors[i] == 1 and seq[0] == 1):
				true_pos = true_pos+1
		corr_list.append([true_neg,true_pos,false_neg,false_pos])

	index = 0
	for j in corr_list:
		print index
		index = index+1
		print j
		print  'total corr ' + str( (float(j[0])+j[1])/(j[0]+j[1]+j[2]+j[3]) )
		print 'neg corr ' + str( float(j[0])/(j[0]+j[2]))
		print 'pos corr ' + str(float(j[1])/(j[1]+j[3]))


def get_percentage(arr):
	numpos = arr[1]
	numneg = arr[0]
	return 1.0*numpos/(numpos+numneg)

def sequence_corr_dict(sequences):
	corr_dict = {}
	corr_dict['0'] = [0,0]
	corr_dict['1'] = [0,0]
	corr_dict['00'] = [0,0]
	corr_dict['10'] = [0,0]
	corr_dict['01'] = [0,0]
	corr_dict['11'] = [0,0]
	corr_dict['000'] = [0,0]
	corr_dict['100'] = [0,0]
	corr_dict['110'] = [0,0]
	corr_dict['101'] = [0,0]
	corr_dict['010'] = [0,0]
	corr_dict['011'] = [0,0]
	corr_dict['001'] = [0,0]
	corr_dict['111'] = [0,0]
	for seq in sequences:
		anc_str = '0'
		if(len(seq[1]) ==  1):
			anc_str = str(seq[1][0])
		elif(len(seq[1]) == 2):
			anc_str = str(seq[1][0])+str(seq[1][1])
		else:
			ancestors = seq[1][:3]
			anc_str = str(ancestors[0])+str(ancestors[1])+str(ancestors[2])
		print(anc_str)
		if(seq[0]==1):
			corr_dict[anc_str][1]= corr_dict[anc_str][1]+1
		else:
			corr_dict[anc_str][0] = corr_dict[anc_str][0]+1

	print('next phase')
	# print corr_dict

	# COMPARE THINGS CORRECTLY USING THE GET_PERCENTAGE FUNCTION #
	ggp1 = get_percentage(corr_dict['001']) - get_percentage(corr_dict['000'])
	ggp2 = get_percentage(corr_dict['011']) - get_percentage(corr_dict['010'])
	ggp3 = get_percentage(corr_dict['101']) - get_percentage(corr_dict['100'])
	ggp4 = get_percentage(corr_dict['111']) - get_percentage(corr_dict['110'])
	ggp_avg = 1.0*(ggp1+ggp2+ggp3+ggp4)/4

	gp1 = get_percentage(corr_dict['01']) - get_percentage(corr_dict['00'])
	gp2 = get_percentage(corr_dict['11']) - get_percentage(corr_dict['11'])
	gp_avg = 1.0*(gp1+gp2)/2

	p_avg = get_percentage(corr_dict['1']) - get_percentage(corr_dict['0'])

	print "Influence of great-grandparents: " + str(ggp_avg)
	print "Influence of grandparents: " + str(gp_avg)
	print "Influence of parents: " + str(p_avg)
	return



	for config in corr_dict.keys():
		print(' ')
		print(config)
		print(float(corr_dict[config][1])+corr_dict[config][0])
		print (float(corr_dict[config][1])/(corr_dict[config][1]+corr_dict[config][0]))




def assemble_sequence_dict(mapping):
	sequences = []
	has_attr = {}
	# print(len(Data_Scraper.root_comments))
	for comment in Data_Scraper.all_comments:

		"""# This is the part where we put in different metrics #"""
		if(get_length(comment) > 120):
			has_attr[comment.comment_id] = True
		else:
			has_attr[comment.comment_id] = False

	if('cedt5zw' in has_attr.keys()):
		# print('ceefsvz is in')
	else:
		# print('ceefsvz is out')
	for comment in Data_Scraper.all_comments:
		#print('before calc_attr')
		#print comment
		attr_tup = calculate_attr(comment, has_attr, mapping)
		if(len(attr_tup[1]) != 0):

			# print(attr_tup)
			#print comment.comment_id
			sequences.append(attr_tup)

	sequence_corr_dict(sequences)
	

def get_length(comment):
	content = comment.content
	content = content.split()
	content = filter(lambda x: "." not in x and "\n" not in x, content)
	return len(content)


def calculate_attr(comment, has_attr, mapping):
	attr_tup = [0,[]]
	attr_vec = []
	if(has_attr[comment.comment_id] == True):
		attr_tup[0]=1
	else:
		attr_tup[0]=0
	while(True):
		#print('location 1')
		#print(Data_Scraper.comment_id_lookup[comment.comment_id])


		parent_id = comment.parent_comment_id
		#print parent_id
		#parent = Data_Scraper.comment_id_lookup[parent_id]
		if(parent_id not in Data_Scraper.comment_id_lookup or parent_id in Data_Scraper.thread_ids):
			break

		if(parent_id in has_attr and has_attr[parent_id] == True):
			attr_vec.append(1)
		if(parent_id in has_attr and has_attr[parent_id] == False):
			attr_vec.append(0)
		#print('location 2')
		#print commentID

		comment = Data_Scraper.comment_id_lookup[parent_id]
		#if(mapping.GetKeyId(Data_Scraper.comment_id_lookup[parent]) in Data_Scraper.root_comments):
			#break
	#attr_vec = list(reversed(attr_vec))
	attr_tup[1] = attr_vec
	return attr_tup


'''
def calculate_attr_children(nid, mapping, g, attr, attr_mean, attr_std):
	
	comment =  Data_Scraper.comment_id_lookup[mapping.GetKey(int(nid))]

	attr_vec = []
	for child in comment.GetOutEdges():
		attr_vec.append(comment.attr())

	return np.mean(np.array(attr_vec))

	parent_lens = []
	child_lens = []
	for n in g.Nodes():
		comment_name = mapping.GetKey(int(n.GetId()))
		comment = Data_Scraper.comment_id_lookup[comment_name]
		content = comment.content
		content = content.split()
		content = filter(lambda x: "." not in x and "\n" not in x, content)
		parent_len = len(content)
		attr_vec = []
		for child in n.GetOutEdges():
			reply = Data_Scraper.comment_id_lookup[mapping.GetKey(int(child))]
			reply = comment.content
			reply = content.split()
			reply = filter(lambda x: "." not in x and "\n" not in x, content)
			attr_vec.append(len(reply))
		child_lens.append(attr_vec.mean())
'''




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
	plt.hist(hist, log=True)
	plt.show()
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

	mapping = snap.TStrIntSH()
	G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)

	root = getRootNode(mapping, G)

	# FK_histogram()

	# FK_scores = getFKScores()

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

	#output.close()


	#sort_comments(200, mapping, 1)



	# pickle.dump(stats_vec,output)
	#output.close()

	# print "Nodes: " + str(Data_Scraper.all_comments)
	# print "Threads: " + str(len(Data_Scraper.thread_ids))
	# print "Root comments: " + str(len(Data_Scraper.root_comments))


	#print "delay: " + estimated_delay(144834, mapping, G)
	
	#record_comment_lengths()
	
	assemble_sequence_dict(mapping)
	print G.GetNodes()
	print G.GetEdges()


def getDegree(G, mapping, nodeId):
	node = G.GetNI(mapping.GetKeyId(nodeId))
	return node.GetOutDeg()

def getNormalizedDegree(G, mapping, nodeId):
	node = G.GetNI(mapping.GetKeyId(nodeId))
	deg = node.GetOutDeg()
	c_object = Data_Scraper.comment_id_lookup[nodeId]
	threadsize = thread_sizes[c_object.thread_id]
	return 1.0*deg/threadsize
	
	
def getJson():
	import json
	nodes = []
	print("looping thru nodes")
	for i in G.Nodes():
		node = {}
		node["id"] = mapping.GetKey(i.GetId())
		node["group"] = 1
		nodes.append(node)
	links = []
	print("looping thru edges")
	for i in G.Edges():
		edge = {}
		edge["source"] = mapping.GetKey(i.GetSrcNId())
		edge["target"] = mapping.GetKey(i.GetDstNId())
		edge["value"] = 1
		links.append(edge)
	f = open("graph.json", "w")
	f.write(json.dumps({"nodes": nodes, "links": links}))
	f.close()

	
# TODO: mean and standard deviation of FK - scores and word lengths

if __name__ == "__main__":
	main()

mapping = snap.TStrIntSH()
G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
root = getRootNode(mapping, G)
thread_sizes = {}
for thread in root.GetOutEdges():
        threadsize = _findDepth(G.GetNI(thread), G)
        thread_sizes[mapping.GetKey(thread)] = threadsize



