import Data_Scraper
import snap
import numpy
import matplotlib.pyplot as plt
import copy
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
	Data_Scraper.load_data()

	measure_comment_lengths()
	
	mapping = snap.TStrIntSH()
	G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
	# # convert input string to node id
	# NodeId = mapping.GetKeyId("1065")
	# # convert node id to input string
	# NodeName = mapping.GetKey(NodeId)

	root = getRootNode(mapping, G)
	
	getCommentHistogram([G.GetNI(n) for n in root.GetOutEdges()], G)

	print G.GetNodes()
	print G.GetEdges()


if __name__ == "__main__":
	main()





