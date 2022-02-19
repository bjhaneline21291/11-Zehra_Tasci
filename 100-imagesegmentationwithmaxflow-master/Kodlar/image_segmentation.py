#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:48:01 2019

@author: fschwarz
"""
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np



#input: array of the image
#output: the network G, pos: a dictionnary that gives the position in 2D to nodes
def getNetwork(imgArray):
    G = nx.DiGraph()
    pos = {}
    SIZE = len(imgArray)    
    for x in range(SIZE):
        for y in range(SIZE):
            G.add_edge('s', str(x) + str(y), capacity=imgArray[y,x][1])
            
    for x in range(SIZE):
        for y in range(SIZE):
            G.add_edge(str(x) + str(y), 't', capacity=255-imgArray[y,x][1])
            
            
    CAPACITY_LOCAL = 64
    for x in range(SIZE-1):
        for y in range(SIZE):
            G.add_edge(str(x) + str(y), str(x+1) + str(y), capacity=CAPACITY_LOCAL)
            G.add_edge(str(x+1) + str(y), str(x) + str(y), capacity=CAPACITY_LOCAL)
    
    for x in range(SIZE):
        for y in range(SIZE-1):
            G.add_edge(str(x) + str(y), str(x) + str(y+1), capacity=CAPACITY_LOCAL)
            G.add_edge(str(x) + str(y+1), str(x) + str(y), capacity=CAPACITY_LOCAL)
            
            
    pos['s'] = [-1, -SIZE/2]
    pos['t'] = [SIZE , -SIZE/2]
    
    for x in range(SIZE):
        for y in range(SIZE):
            pos[str(x) + str(y)] = [x, -y]
            
    return G, pos
    

#input: rgba_color color array [r, g, b] where numbers are between 0 and 255
#output: the hexadeciam lcode
def rgbarray2hex(rgba_color):
    red = int(rgba_color[0])
    green = int(rgba_color[1])
    blue = int(rgba_color[2])
    return '#%02x%02x%02x' % (red, green, blue)

#input: an integer i between 0 and 255
#output: a gray color in hexa that represents that integer
def int2grayhex(i):
    j = 192-2*i//3
    return rgbarray2hex([j, j, j])

#load the file image.png and returns the array of the colors
def getImgArray():
    im = Image.open("image.png")
    return np.array(im)

#draw the network
def draw(G, pos, imgArray, cut, filename):    
    #pos = nx.spring_layout(G, iterations=500)
    plt.figure(figsize=(8, 8))
    #nx.draw(G, pos) #pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False
    
    NODE_SIZE = 300
    nx.draw_networkx_nodes(G,pos,
                           nodelist=['s', 't'],
                           node_color='black',
                           node_size=NODE_SIZE)
    
    
    SIZE = len(imgArray)
    for x in range(SIZE):
        for y in range(SIZE):
            if(str(x) + str(y) in cut[0]):
                flag = 'blue'
                node_shape='v'
            else:
                flag = 'black'
                node_shape = 'o'
                
            nx.draw_networkx_nodes(G,pos,
                           nodelist=[str(x) + str(y)],
                           node_color=rgbarray2hex(imgArray[y,x]),
                           edgecolors = flag,
                           node_shape=node_shape,
                           node_size=NODE_SIZE)
            
    for e in G.edges:
        nx.draw_networkx_edges(G,pos,
                           edgelist=[e],
                           width=1,alpha=1,edge_color=int2grayhex(G.get_edge_data(e[0], e[1])['capacity']))   
    
    plt.axis('equal')
    
    plt.savefig(filename)
    plt.show()
        
    
    
    

imgArray = getImgArray()
G, pos = getNetwork(imgArray)
cutvvalue, cut = nx.minimum_cut(G, 's', 't')

draw(G, pos, imgArray, [[], []], "maxflow_imagesegmentation_network.png")
draw(G, pos, imgArray, cut, "maxflow_imagesegmentation_result.png")



