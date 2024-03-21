import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the dimensions of the blank image
image_width = 800
image_height = 600

# Create a blank image
img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Define the start node (0) and its position
start_pos = (image_width // 2, 100)

# Define the depth of the tree (total_depth) and the depth of the current node (depth)
total_depth = 5
depth = 0


# Define a function to draw a node
def draw_node(img, point, size, color):
    x, y = point
    cv2.circle(img, (x, y), size, color, -1)


# Define a function to draw an edge
def draw_edge(img, point1, point2, color):
    x1, y1 = point1
    x2, y2 = point2
    cv2.line(img, (x1, y1), (x2, y2), color)


# Add the start position as the first node
draw_node(img, start_pos, size=10, color=(0, 255, 0))


# Generate nodes and edges
def draw_tree(img, start_pos, depth, total_depth, G):
    if depth > total_depth:
        return

    # Calculate end positions
    angle = 30
    length = 100
    radian = np.radians(angle)
    dx = int(length * np.cos(radian))
    dy = int(length * np.sin(radian))

    end_pos1 = (start_pos[0] + dx, start_pos[1] + dy)
    end_pos2 = (start_pos[0] - dx, start_pos[1] + dy)

    # Draw lines and circles
    thickness = max(1, 2 - depth + total_depth // 3)
    radius = 10 + (total_depth - depth) * 3
    cv2.line(img, start_pos, end_pos1, (255, 255, 255), thickness=thickness)
    cv2.line(img, start_pos, end_pos2, (255, 255, 255), thickness=thickness)
    cv2.circle(img, start_pos, radius, (0, 255, 0), -1)

    # Add edges to graph
    G.add_edge(start_pos, end_pos1)
    G.add_edge(start_pos, end_pos2)

    # Recursive drawing for next depth
    draw_tree(img, end_pos1, depth + 1, total_depth, G)
    draw_tree(img, end_pos2, depth + 1, total_depth, G)


# Create an empty graph
G = nx.Graph()

# Build the tree and add edges to the graph
draw_tree(img, start_pos, depth, total_depth, G)

# Calculate the Minimum Spanning Tree using Kruskal's algorithm
mst = nx.minimum_spanning_tree(G)

# Draw MST edges on the image
for edge in mst.edges():
    draw_edge(img, edge[0], edge[1], (0, 0, 255))

# Display the image with MST edges
cv2.imshow("Tree with Minimum Spanning Tree", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the graph with MST using NetworkX
nx.draw(mst, with_labels=True)
plt.show()
