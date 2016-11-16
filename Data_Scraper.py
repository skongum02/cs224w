# Tom Kremer
# October 23, 2016

# Adding an edit comment


#BEN KRAUSZ IS ADDING A TEST COMMENT HERE!!!!
### READ ME ###
#--------------------------------#
# This class reads in the /r/politics file and scrapes it. To do so, just
# import the class and invoke the load_data() method.

# The three variables that can be used publicly are:
# all_comments: A Python list of all the comment objects
# root_comments: A Python list of all the root comments
# thread_ids: A Python set (!) of all the thread IDs. These can be made into a list 

# The comment_object is a struct that is modeled as so:
#	time_stamp: An int that represents the time this comment was posted
#	comment_id: A unique string for this comment
#	parent_comment_id: The unique string for this comment's parent
#	author_name: The string username of the comment's author
#	score: An int representing the total score of  comment from up and downvotes
#	thread_id: A string for this comment's entire thread
#	content: The string that represents the actual text of the comment
#--------------------------------#



from sets import Set
from Comment import Comment
import re

# <subreddit_name>	-> 0 
# <time_stamp>  	 	-> 1
# <subreddit_id>  	-> 2
# <comment_id>  		-> 3
# <parent_comment_id> -> 4
# <author_name> 		-> 5
# <score> 			-> 6
# <???> 				-> 7
# <thread/link_id> 	-> 8
# <text>				-> 9

# In Python, Sets require hashable objects, which a dictionary entry isn't
# So we'll try using arrays, though it might be too cumbersome down the line
all_comments = []
root_comments = []
thread_ids = Set([])	# This is a Set of all the root nodes, since they  only
						# have edges coming out, aka thread_ids. To be read in 
						# along with the comment IDs into which ever program we 
						# use to make the model so that the threads are 
						# represented as a node with only roots and an id.

comment_edges = []		# Array of tuples of form: (parent, child), possibly 
						# more useful than inverse tree model. Could be an issue
						# since node IDs are strings, not sure how snap.py handles that.
comment_id_lookup = {}

# Specifically tailored for the /r/politics dataset
def load_data():
	f = open("politics.tsv", "r")
	lines = f.readlines()

	comment_object = {}

	for line in lines:
		arr = line.split('\t')
		comment_object["time_stamp"] = int(arr[1])
		comment_object["comment_id"] = arr[3]
		comment_object["parent_comment_id"] = arr[4]
		
		comment_edges.append((arr[4], arr[3]))
		
		comment_object["author_name"] = arr[5]
		comment_object["score"] = int(arr[6])
		comment_object["thread_id"] = arr[8]
		thread_ids.add(arr[8])

		content = arr[9]
		content = re.sub(' ?<EOS> ?', '', content)
		comment_object["content"] = content


		this_comment = Comment(comment_object)
		if comment_object["parent_comment_id"] == comment_object["thread_id"]:
			# This is a root node for a comment tree
			root_comments.append(this_comment)
		all_comments.append(this_comment)

		comment_id_lookup[comment_object["parent_comment_id"]] = comment_object

	f.close()


# Writes the edge list file. Thread nodes have central root node as parent.
def write_edge_list():
	f = open('politics_edge_list.txt','w')
	for (parent, child) in comment_edges:
		f.write(parent + "\t" + child + "\n")
	for thread_id in thread_ids:
		f.write("root\t" + thread_id + "\n")
	f.close()

def main():
	load_data()

	# write_edge_list

	return


if __name__ == "__main__":
	main()
	# load_data()

