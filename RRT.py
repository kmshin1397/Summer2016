from Tkinter import *
import math
import random
import numpy as np
from collections import defaultdict
import json
from scipy.spatial import cKDTree
import Queue

from ..RealWorld.Point import Point
from ..RealWorld.Surface import Surface

import time

# from shapely.geometry import Point, Polygon


class RRT(object):

	'''
		@param list of Surface instances that belong to the world
	'''
	def __init__(self, surfaces):

		#--------------------------------------------------------------------
		# Parameters

		# Closest the tree can get to a surface edge before being obstructed
		self.max_dist = 0.1
		self.distance_allowed = 0.36

		# Number whose inverse gives the probability that the algorithm will
		# attempt to connect the tree to the goal at any given iteration.
		self.goal_probability = 750

		self.iterations = 750

		# Maximum length an edge in the tree can be. Points connected with
		# distances greater than this will be split into many edges, with 
		# vertices placed along the distance. Decreasing this increases accuracy
		# of the function finding the nearest point on the current tree to a randomly
		# selected point to be added, but will increase computation time.
		self.max_length = 0.15

		#--------------------------------------------------------------------

		self.current = (0, 0)
		self.surfaces = surfaces
		self.obstacles = []

		for eachSurface in surfaces: 
			self.obstacles.append(eachSurface.points)


		# Bounds for the negative space
		x_max = np.NINF
		x_min = np.inf
		y_max = np.NINF
		y_min = np.inf

		# Find min x & min z(aka y)
		for eachSurface in self.obstacles:
			for eachPoint in eachSurface:
				if (eachPoint[0] < x_min):
					x_min = eachPoint[0]
				elif (eachPoint[0] > x_max):
					x_max = eachPoint[0]
				if (eachPoint[2] < y_min):
					y_min = eachPoint[2]
				elif (eachPoint[2] > y_max):
					y_max = eachPoint[2]

		# Dict obj to hold RRT
		self.graph = defaultdict(list)

		# Populate world space with points for sampling
		space = []
		x_values = np.linspace(x_min - 0.7 , x_max + 0.7, 500)
		y_values = np.linspace(y_min - 0.1, y_max + 0.3, 500)
		for x in x_values:
			for y in y_values:
				space.append((x, y)) # Points stored as 2-D tuple coordinates

		self.free_space = space

		self.total_dist = 0
		self.total_soln = []

	# Calculates the linear distance between two points
	def distance (self, point1, point2):
		dx = point1[0] - point2[0]
		dy = point1[1] - point2[1]

		return math.sqrt(dx ** 2 + dy ** 2)

	# Returns the midpoint of a line segment with ends pt1 and pt2
	def midpoint(self, pt1, pt2):
		x = (pt1[0] + pt2[0])/2
		y = (pt1[1] + pt2[1])/2
		return (x, y)

	# Given a line with coordinates 'start' and 'end' and the coordinates of a point 
	# 'pt' returns the shortest distance from pt to the line and the coordinates of 
	# the nearest point on the line.
	#
	# The angle made by the vectors from the point to the start and the start to
	# the end of the segment, and the angle made by the segment vector to the 
	# vector from the end point to the point are used to compute the nearest point.
	# If the angle (pt, start, end) is obtuse, then the closest point is the start
	# point. If angle (start, end, pt) is acute, then the closest point is the end.
	# Otherwise, the nearest point is the normal to the segment which interects the
	# point.
	#
	# Algorithm modified from 
	# http://geomalgorithms.com/a02-_lines.html#distance-to-Infinite-Line
	# Copyright 2001 softSurfer, 2012 Dan Sunday
	  
	def pt2line(self, pt, start, end):
		# Convert start point to array to allow vector addition later
		start_arr = np.array([start[0], start[1], 0], float)

		seg_vec = np.array([end[0] - start[0], end[1] - start[1], 0], float)
		pt_vec = np.array([pt[0] - start[0], pt[1] - start[1], 0], float)
		
		a = np.dot(pt_vec, seg_vec)
		if (a <= 0): # If pt is to the left of start
			return (self.distance(pt, start), start)

		b = np.dot(seg_vec, seg_vec)
		if (b <= a): # If pt is to the right of end
			return (self.distance(pt, end), end)

		c = a / b    # a / b = (Projection of pt_vec onto seg_vec) / seg_vec
		nearest = start_arr + c * seg_vec
		dist = self.distance(nearest, pt)

		# Convert to tuple from np.array
		nearest = (nearest[0], nearest[1])
		return (dist, nearest)

	# Get the minimum distance betweem two line segments (pt1, pt2) and (pt3, pt4)
	# This is accomplished by minimizing W(s, t) = S1(s) - S2(t), where S1 and S2
	# are vectors describing the line segments. W is thus a vector between points
	# on the two segments.
	#
	# If S1(s) = pt1 + s(pt2 - pt1) and S2(t) = pt3 + t(pt4 - pt3),
	# then the line segments have bounds s , t = [0, 1]
	#
	# First the closest points will be found for the lines extended from the 
	# line segments, S1(s) and S2(t). If the closest points with the minimum 
	# distance are outside the bounds [0, 1] X [0,1], then the closest points will 
	# will occur at a boundary edge of the [0, 1] X [0,1] region on the (s, t) plane,
	# as we are essentially trying to minimized (w(t))^2 , which is a quadratic in
	# the (s, t) plane. In other words, we find the point (sc, tc) which give the
	# points on the lines closest to each other. This point will be the vertex of a
	# concave up parabola in the (s, t) plane, and the range of our line segments 
	# will be the unit square on the (s, t) plane described by [0, 1] X [0,1]. If 
	# (sc, tc) lies outside these bounds for the line segments, the edge of the 
	# unit square closest to the closest point can be found, followed by the nearest
	# point on that edge to the closest point - thus minimizing (w(t))^2.
	#
	# Algorithm modified from http://geomalgorithms.com/a07-_distance.html
	# Copyright 2001 softSurfer, 2012 Dan Sunday 

	def segment2segment(self, pt1, pt2, pt3, pt4):
		u = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1], 0], float)
		v = np.array([pt4[0] - pt3[0], pt4[1] - pt3[1], 0], float)
		w = np.array([pt1[0] - pt3[0], pt1[1] - pt3[1], 0], float)
		a = np.dot(u, u) # Always >= 0
		b = np.dot(u, v)
		c = np.dot(v, v) # Always >= 0
		d = np.dot(u, w)
		e = np.dot(v, w)
		D = a * c - b * b # Always >= 0
		sc = 0  # Value of s where closest point occurs  -   sN / sD
		sN = 0  # sN is the numerator to the s variable
		sD = D  # The denominator of s
		tc = 0  # Value of t where closest point occurs  -   tN / tD
		tN = 0  # Numerator of t
		tD = D  # Denominator of t

		# Compute the line parameters of the two closest points
		if (D < 0.00000001): # The lines are almost parallel
			sN = 0.0         # WLOG use s = 0 point, since the lines are parallel
			sD = 1.0
			
			# tc = tN / tD = dot(v, w) / dot(v, v) = (Projection of w onto v) / v
			tN = e            
			tD = c
		else:
			sN = (b * e - c * d)
			tN = (a * e - b * d)

			if (sN < 0.0):  # sc < 0 => the s = 0 edge of the square is closest
				sN = 0.0
				tN = e
				tD = c
			elif (sN > sD): # sc > 1 -> the s = 1 edge of the square is closest
				sN = sD
				tN = e + b
				tD = c

		if (tN < 0.0):  # tc < 0 => the t = 0 edge of the square is closest
			tN = 0.0
			# recompute sc for this edge
			if (-d < 0.0):
				sN = 0.0
			elif (-d > a):
				sN = sD
			else:
				sN = -d
				sD = a
		elif (tN > tD): # tc > 1 => the t = 1 edge of the square is closest
			tN = tD
			# recompute sc for this edge
			if ((-d + b) < 0.0):
				sN = 0
			elif ((-d + b) > a):
				sN = sD
			else:
				sN = -d + b
				sD = a
		# Finally divide to get sc and tc
		sc = 0.0 if abs(sN) < 0.00000001 else sN / sD
		tc = 0.0 if abs(tN) < 0.00000001 else tN / tD

		# Get the difference of the two closest points
		dP = w + (sc * u) - (tc * v)
		return np.linalg.norm(dP)

	# Return coefficients for velocity vector along line segment (pt1, pt2)
	def parametric_velo(self, pt1, pt2):
		i = pt2[0] - pt1[0]
		j = pt2[1] - pt1[1]
		return (i, j)

	# Shortest path through RRT found with AStar. 
	# Returns a tuple of the distance to the goal and a list of edges in the 
	# solution path.
	def shortestPath(self, graph, initial, goal):
		openset = Queue.PriorityQueue()
		openset.put((0, initial))
		# Costs for vertices in graph
		distances = dict.fromkeys(graph.keys(), np.inf)
		distances[initial] = 0

		# Dict of previous vertex for each node in solution
		prev =  dict.fromkeys(graph.keys(), ())

		while not openset.empty():
			current = openset.get()[1]

			if current == goal:
				break

			for eachNeighbor in graph[current]:
				new_dist = distances[current] + self.distance(current, eachNeighbor)
				if new_dist < distances[eachNeighbor]:
					distances[eachNeighbor] = new_dist
					priority = new_dist + self.distance(goal, eachNeighbor)
					openset.put((priority, eachNeighbor))
					prev[eachNeighbor] = current

		# Retrace
		soln = []
		while prev[current] != ():
			soln.append((current, prev[current]))
			current = prev[current]
		return (distances[goal], soln)

	# Return the nearest vertex in current tree to randomly sampled point
	def nearest(self, rand_pt):

		data = self.graph.keys()
		tree = cKDTree(data)
		pt = np.array(rand_pt)
		nearest = tree.query(pt)
		
		return data[nearest[1]]

	# Check if path to random point has obstacle and adjust accordingly.
	# Returns nearest point in current swath if it is already too close to the
	# obstacle.
	# Returns a new point in the direction of the random point but outside the
	# obstacle if the path to the random point is blocked.
	# Returns the random point if the path is not obstructed.
	def stopping_config(self, near_pt, rand_pt):
		limit = self.max_dist
		blocked = False
		intersect_edge = 0
		dist = np.inf
		near = (0, 0)   # The point on the surface boundary closest to near_pt

		# Check if new edge segment intersects a surface edge or gets too close
	
		for eachSurface in self.obstacles:
			for i, vertex in enumerate(eachSurface): 
				vertex1 = (vertex[0], vertex[2]) # X and Z coordinates
				vertex2 = (eachSurface[(i + 1) % len(eachSurface)][0], 
					eachSurface[(i + 1) % len(eachSurface)][2])

				# If the minimum distance between the new edge and a surface 
				# edge is below a limit, the path is blocked.
				# Keep track of the surface edge that is closest in distance
				# to the new edge, in the case that the path is blocked by 
				# multiple surface edges.
				dis_sgmnts = self.segment2segment(near_pt, rand_pt, vertex1, vertex2)
				if (dis_sgmnts <= limit):
					blocked = True
					limit = self.segment2segment(near_pt, rand_pt, vertex1, vertex2)
					dist, near = self.pt2line(near_pt, vertex1, vertex2)

		if (blocked):
			
			dist_from = 0 # Keeps track of distance of stopping point from 
						  # Obstacle edge

			# Find point in direction of rand_pt but outside obstacle by
			# traveling in that direction until limit is reached.

			# Get velocity vector in direction of rand_pt from near_pt
			i, j = self.parametric_velo(near_pt, rand_pt) 

			t = 0 # Time variable
			while (dist_from < self.max_dist):
				# Travel along line towards rand_pt
				new_point = (near_pt[0] + i * t, 
					near_pt[1] + j * t)
				dist_from = self.distance(near, new_point)

				t += 0.000001 # Increment time by a small number
				
			return new_point

		return rand_pt

	# Functions for updating the RRT graph. 
	# The graph is a dictionary in which the keys are the vertices of the graph
	# and their values are the vertices with which they are connected by an edge
	def add_vertex(self, pt):
		if (not(pt in self.graph)):
			self.graph[pt] = []

	def add_edge(self, pt1, pt2):
		if (not(pt2 in self.graph[pt1])):
			self.graph[pt1].append(pt2)
		if (not(pt1 in self.graph[pt2])):
			self.graph[pt2].append(pt1)

	# Splits up a long edge into smaller segments, so that intermediate vertices
	# can be used to approximate nearest vertex for future iterations
	def add_intermediates(self, pt1, pt2):

		# Base case
		if (self.distance(pt1, pt2) <= self.max_length):
			self.add_vertex(pt2)
			self.add_edge(pt1, pt2)
			return True
		
		elif (self.distance(pt1, pt2) > self.max_length):
			self.add_vertex(self.midpoint(pt1, pt2))
			self.add_vertex(pt2)
			if(self.add_intermediates(pt1, self.midpoint(pt1, pt2)) and 
				self.add_intermediates(self.midpoint(pt1, pt2), pt2)):
				return True

	# Function returns a random point in the negative space which has not been random
	# explored, and occasionally returns the goal to see if it can be connected
	# to the tree.
	def randPoint(self, goal):
		weight = random.randrange(self.goal_probability)
		rand_pt = goal
		if (weight != 1):
			rand_pt = self.free_space[random.randrange(len(self.free_space))]
			while rand_pt in self.graph:
				rand_pt = self.free_space[random.randrange(len(self.free_space))]
		return rand_pt

	# Smooths the solution path after it is found with Djikstra's
	def smoothCurve(self, soln):
		result = []
		base = soln[0][0]
		previous = base
		blocked = False
		for eachEdge in soln:
			current = eachEdge[1]
			for eachSurface in self.obstacles:
				for i, vertex in enumerate(eachSurface): 
					vertex1 = (vertex[0], vertex[2]) # X and Z coordinates
					vertex2 = (eachSurface[(i + 1) % len(eachSurface)][0], 
						eachSurface[(i + 1) % len(eachSurface)][2])

					# If the minimum distance between the new edge and a surface 
					# edge is below a limit, the path is blocked.
					dis_sgmnts = self.segment2segment(base, current, vertex1, vertex2)
					if (dis_sgmnts <= self.max_dist):
						blocked = True

			if (blocked):
				result.append((base, previous))
				base = (previous[0], previous[1])
				blocked = False
			if (eachEdge == soln[-1]):
				result.append((base, current))
			previous = (current[0], current[1])
		return result

	# Ended up not using
	# # Returns True if a given point is inside an obstacle, or is too near one.
	# def PointnearObstacle(self, point):
	# 	result = False
	# 	surface_num = 0

	# 	point = (point[0],0.0,point[1])

	# 	# Use this for loop to check if point is inside a obstacle. 
	# 	for index, eachSurface in enumerate(self.surfaces):
	# 		if eachSurface.surfaceContainsPoint(point): 
	# 			return True
	# 		else: 
	# 			distance = Point(eachSurface.findClosestPointOnEdge(point)).getEdgeLength(Point(point))
	# 			if distance <= self.max_dist:
	# 				return True
		
	# 	return False


	#---------------------------------------------------------------------------
	#   Useful functions
	#
	# Creates an RRT to traverse the space and reach the goal point, or close
	# to it if the goal is within/near an obstacle. The edges of a path to the
	# goal point will be added to a total solution list, and the total distance
	# will be updated. The distance and solution edges from this update only will
	# also be returned by the function.

	# Make large RRT - 1000 iterations?
	# Find nearest vertices to checkpoints
	# Shortest path/smooth to each of those nearest vertices.


	'''
		@param: goal - (x,z)
	'''
	def updateTo(self, goal):
		t = 0.0
		# Reset RRT tree
		#self.graph = defaultdict(list)
		#self.graph[self.current] = []

		# If first time updating, make an RRT
		if (self.graph == {}):
			self.graph[self.current] = []
			#t = time.clock()
			#print "Creating rrt... " + str(t)

		# goal_in_obstacle = self.PointnearObstacle(goal)

		# if (goal_in_obstacle[0]):
			# print "Goal point located in/near surface #" + str(goal_in_obstacle[1])
			# print "Goal point: " + str(goal)

			for i in range(self.iterations):
				# Random point to be used for expanding tree.
				rand_pt = self.randPoint(goal)
				
				# Vertex in current tree closest to the random point
				near_pt = self.nearest(rand_pt)
				
				# Point in direction of random point appropriate distance away from
				# obstacles.
				stop_pt = self.stopping_config(near_pt, rand_pt)

				if (near_pt != stop_pt):

					# If an edge is too long, split it into multiple edges
					if (self.distance(near_pt, stop_pt) > self.max_length): 
						self.add_intermediates(near_pt, stop_pt)
				
					else:
						self.add_vertex(stop_pt)
						self.add_edge(near_pt, stop_pt)

			# Since all our goal points are inside the obstacles, this first if condition will never be true. 
			# Keeping code for future ref. 
			# if (goal in self.graph):
			# 	dist, soln = self.shortestPath(self.graph, initial, goal)
			# 	soln = self.smoothCurve(soln)
			# 	dist = 0
			# 	for eachEdge in soln:
			# 		dist += self.distance(eachEdge[0], eachEdge[1])
			# 	soln.reverse()
			# 	for i, eachEdge in enumerate(soln):
			# 		soln[i] = eachEdge[::-1]
			# 	self.total_dist += dist
			# 	if (soln != []):
			# 		total_soln.extend(soln)
			# 		self.current = soln[-1][1] # Start next path from end of this one
			# 	return (dist, soln)
			# if not goal_in_obstacle:

			# If new vertex is close enough to the goal
			#if (self.distance(stop_pt, goal) <= self.max_dist + self.distance_allowed):

		#t = time.clock()
		#print "RRT done! Took: " + str(t)

		goal_pt = self.nearest(goal) # Find nearest vertex in RRT to goal

		dist, soln = self.shortestPath(self.graph, self.current, goal_pt)

		# If current point is already close enough
		if (soln == []):
			return (1, [])

		soln = self.smoothCurve(soln)

		dist = 0
		for eachEdge in soln:
			dist += self.distance(eachEdge[0], eachEdge[1])
		
		soln.reverse()
		for i, eachEdge in enumerate(soln):
			soln[i] = eachEdge[::-1]

		self.total_dist += dist

		if (soln != []):
			self.total_soln.extend(soln)
			self.current = soln[-1][1] # Start next path from end of this one

		return (dist, soln)

	def getCurrent(self):
		return self.current


	'''
		Find closest points of each eurface to the one before, to give to program to visit
		Using the findClosestPointOnEdge method in the Surface class
		@param start (x,y,z) point of the current estimated human location
	'''
	def findCheckpoints(self, start, surface_indexes_to_visit):
		self.current = (start[0],start[2])
		previous = start
		checkpoints = []
		path = surface_indexes_to_visit    # Use these obstacle indices, in that order
		

		# Closest point in surface
		for index in surface_indexes_to_visit:
			surfaceObj = self.surfaces[index]
			previous = surfaceObj.findClosestPointOnEdge(previous)
			checkpoints.append((previous[0],previous[2]))

		return checkpoints

	# Graphics commands

	def displayWorld(self, canvas, checkpoints):

		colors = ["green", "blue", "cyan", "yellow", "magenta", "red", "black", "orange"]
		j = 0

		for eachSurface in self.obstacles:
			for i in range(len(eachSurface)):
				canvas.create_line(int((eachSurface[i][0] + 1.3) * 150), int((eachSurface[i][2] + 4.2) * 150), 
					int(((eachSurface[(i + 1) % len(eachSurface)][0]) + 1.3) * 150), 
					int(((eachSurface[(i + 1) % len(eachSurface)][2]) + 4.2) * 150), 
					fill='blue', width=1)
			j += 1

		# j = 0
		
		for j, eachPoint in enumerate(checkpoints):
			point = (int((eachPoint[0] + 1.3) * 150), int((eachPoint[1] + 4.2) * 150))
			self.create_circle(canvas, colors[j], 10, point)


	def create_circle(self, canvas, color, size, point):
		'''
			The create_circle function takes a canvas, a color written as a string, 
			a diamter length, and a point and creates a circle at that location
			with the given specifications, also returning the handle of the circle as 
			well.
		'''
		handle = canvas.create_oval(point[0] - size/2, point[1] - size/2, point[0] + 
									size/2, point[1] + size/2, fill = color, 
									outline = color)
		return handle


	# Sasha: Can be used to draw the route selected by heuristics. Do not delete yet.
	def displayRRT(self, canvas):
			
		for eachEdge in self.total_soln:
			if (eachEdge != []):
				canvas.create_line(int((eachEdge[0][0] + 1.3) * 150), int((eachEdge[0][1] + 4.2) * 150), 
					int((eachEdge[1][0] + 1.3) * 150), int((eachEdge[1][1] + 4.2) * 150), 
					fill='red', width=2)     

		print "Distance so far: " + str(self.total_dist)


# Script for testing path-finding, calls on graphics functions to display results.

##Event handlers.
#def key_handler(event):
#	global checkpoints
#	global checkpoints_i
#	'''Handle key presses.'''
#	key = event.keysym
#	if key == 'q':
#		quit()
#	# Press r to advance the path by one step
#	elif key == 'r':
#		world.updateTo(checkpoints[checkpoints_i])
#		world.displayRRT(self, canvas)
#		checkpoints_i += 1
#	# Press a to compute the entire path
#	elif key == 'a':
#		for eachPoint in checkpoints:
#			world.updateTo(eachPoint)
#			world.displayRRT(canvas)#

#if __name__ == '__main__':
#	root = Tk()
#	root.geometry('600x600')
#	canvas = Canvas(root, width=600, height=600)
#	canvas.pack()
#	JSON_FILE_NAME = sys.argv[-1]
#	# Parse json file - for testing
#	json_file = open(JSON_FILE_NAME, "r")
#	json_data = json.load(json_file)
#	obstacles = []
#	for eachSurface in json_data['surfaces']:
#	    for each in eachSurface:
#	        if ('points' in each):
#	            vertices = each['points']
#	            if (abs(vertices[0][0]) != 50):
#	                obstacles.append(vertices)
#	world = RRT(obstacles)
#	start = (-0.41, -1.3)
#	surfaces_to_visit = [3,0,1,0]
#	checkpoints = world.findCheckpoints(start, surfaces_to_visit)
#	world.setStart(start)
#	world.displayWorld(canvas, checkpoints)
#	checkpoints_i = 0
#	# Bind events to handlers.
#	root.bind('<Key>', key_handler)
#	# Start it up.
#	root.mainloop()



