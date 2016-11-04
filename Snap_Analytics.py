import Data_Scraper
import snap

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

def main():
	Data_Scraper.load_data()
	
	mapping = snap.TStrIntSH()
	G = snap.LoadEdgeListStr(snap.PNGraph, "politics_edge_list.txt", 0, 1, mapping)
	
	numberOfThreads(mapping, G)

	print G.GetNodes()
	print G.GetEdges()

def numberOfThreads(mapping, g):
	rootId = mapping.GetKeyId("root")
	root = g.GetNI(rootId)
	print("The number of threads: {0}".format(root.GetDeg()))
	
	

if __name__ == "__main__":
	main()