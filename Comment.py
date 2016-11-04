# Tom Kremer
# CS 224W r/politics analytics
# November 3, 2016

class Comment(object):

# The comment_object is a class with the following attributes:
#	time_stamp: An int that represents the time this comment was posted
#	comment_id: A unique string for this comment
#	parent_comment_id: The unique string for this comment's parent
#	author_name: The string username of the comment's author
#	score: An int representing the total score of  comment from up and downvotes
#	thread_id: A string for this comment's entire thread
#	content: The string that represents the actual text of the comment

	# c_object is the comment_object used in Data_Scraper.py
	def __init__(self, c_object):
		"""Return a Comment object with all the traits from the original Reddit 
		comment scrape."""
		self.time_stamp = c_object["time_stamp"]
		self.comment_id = c_object["comment_id"]
		self.parent_comment_id = c_object["parent_comment_id"]
		self.author_name = c_object["author_name"]
		self.score = c_object["score"]
		self.thread_id = c_object["thread_id"]
		self.content = c_object["content"]
