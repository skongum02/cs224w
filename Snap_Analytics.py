import Data_Scraper
import snap
import numpy
import matplotlib.pyplot as plt
import copy
import pickle
from collections import deque


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

def get_comment_from_nid(nid):
	comment_name = mapping.GetKey(comment_stats2[0])
	comment_obj = comment_id_lookup[comment_name]
	return comment_obj



#sorts statistics of comments by degree (pv=1), maxdepth (pv=2), or treesize (pv=3)
def sort_comments(cutoff, mapping, pv):
	print("before open pickle")
	pkl_file = open('comment_stats.pkl', 'rb')
	print("after open pickle")
	comment_stats = pickle.load(pkl_file)
	print("after load pickle")
	comment_stats = numpy.array(comment_stats)
	comment_stats2 = []
	for i in range(len(comment_stats)):
		#print comment_stats[i][4]
		if(comment_stats[i][4] >= cutoff):
			#numpy.delete(comment_stats,i,0)
			comment_stats2.append(comment_stats[i])
	#comment_stats2 = []
	#for c in comment_stats:
	#	comment_stats2.append(c)
	print("before sort")
	print comment_stats2[0]
	comment_stats2 = sorted(comment_stats2, key=lambda x: x[pv], reverse=True)
	print comment_stats2[0]
	comment_name = mapping.GetKey(int(comment_stats2[0][0]))
	comment_obj = Data_Scraper.comment_id_lookup[comment_name]
	print(comment_obj.content)
	return comment_stats2



def comment_statistics(mapping, g):
	stats_vec = []
	root = getRootNode(mapping, g)
	for thread in root.GetOutEdges():
		#print(thread)
		threadsize = _findDepth(g.GetNI(thread), g)
		for n in g.GetNI(thread).GetOutEdges():
			deg = (g.GetNI(n).GetDeg()-1)/float(threadsize)
			maxdepth = getMaxDepth(g,n)/float(threadsize)
			treesize = _findDepth(g.GetNI(n),g)/float(threadsize)
			stats_vec.append([n,deg,maxdepth,treesize,threadsize])
	return stats_vec


# Helper function, not to be called directly
def maxDepthRecur(G, nid):
	depthvec = [0]
	for reply in G.GetNI(nid).GetOutEdges():
		depthvec.append(maxDepthRecur(G,reply))
	return max(depthvec)+1

def getMaxDepth(G, nid):
	return maxDepthRecur(G, nid)-1

"""### NOT DONE YET ###"""
def measure_comment_lengths():
	comment_length_count = {}

	for comment in Data_Scraper.all_comments:
		content = comment.content
		print content


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


def main():
	print('begin main')
	Data_Scraper.load_data()
	#measure_comment_lengths()
	
	print('before mapping')
	mapping = snap.TStrIntSH()
	G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
	
	root = getRootNode(mapping, G)
	#stats_vec = comment_statistics(mapping, G)
	#print(stats_vec[4])
	print('before comment histogram')
	#getCommentHistogram([G.GetNI(n) for n in root.GetOutEdges()], G)
	#output = open('comment_stats.pkl', 'wb')
	#pickle.dump(stats_vec,output)
	sort_comments_by_degree(200, mapping)
	print G.GetNodes()
	print G.GetEdges()




if __name__ == "__main__":
	main()





