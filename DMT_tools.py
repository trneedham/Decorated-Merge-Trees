import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
import persim
import networkx as nx
import random
import copy

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.optimize import linear_sum_assignment

from sklearn.metrics.pairwise import pairwise_distances

import gudhi as gd

import ot


from bisect import bisect_left
from hopcroftkarp import HopcroftKarp

"""
A collection of functions for manipulating, visualizing and comparing merge trees
and merge trees decorated with persistence data.

Code associated to the paper "Decorated Merge Trees for Persistent Topology", by
Justin Curry, Haibin Hang, Washington Mio, Tom Needham and Osman Okutan

Code by Tom Needham
"""

"""
Merge Tree class
"""

"""
Merge trees are represented as classes with basic attributes and visualization methods.
A merge tree class can be fit from:
- directly inputing the tree/node height
- a point cloud, assumed to be Euclidean, created using Vietoris-Rips
- a network, with node filtration specified
"""

"""
Helper Functions
"""

def invert_label_dict(label):

    # Take dict {key:item}, return dict {item:key}

    inverted_label = dict()

    for key in label.keys():

        if type(label[key]) == list:
            for l in label[key]:
                inverted_label[l] = key
        else:
            inverted_label[label[key]] = key

    return inverted_label

def get_key(dictionary, val):

    # Given a dictionary and a value, returns list of keys with that value
    res = []
    for key, value in dictionary.items():
         if val == value:
            res.append(key)

    return res

def matrix_ell_infinity_distance(M1,M2):
    # Inputs: two numpy arrays of the same size
    # Output: \ell_\infty distance between the matrices

    M = np.abs(M1 - M2)
    dist = np.max(M)

    return dist

def remove_short_bars(dgm, thresh=0.01):

    dgm_thresh = dgm[dgm[:,1]-dgm[:,0] > thresh]

    return dgm_thresh

def linkage_to_merge_tree(L,X):

    nodeIDs = np.unique(L[:,:2]).astype(int)
    num_leaves = X.shape[0]

    edge_list = []
    height = dict()

    for j in range(num_leaves):
        height[j] = 0

    for j in range(L.shape[0]):
        edge_list.append((int(L[j,0]),num_leaves+j))
        edge_list.append((int(L[j,1]),num_leaves+j))
        height[num_leaves+ j] = L[j,2]

    T = nx.Graph()
    T.add_nodes_from(nodeIDs)
    T.add_edges_from(edge_list)

    return T, height

"""
Network to Merge Tree Tools

Extracting a merge tree from a network requires more work.
"""

def perturb_function(f, perturbation = 1e-10):

    if len(np.unique(list(f.values()))) < len(list(f.values())):
        f = {j:f[j]+perturbation*np.random.rand() for j in f.keys()}

    return f

def get_merge_tree_from_network(G,f):

    # The algorithm needs unique entries to work. Check here and perturb if
    # that's not the case.

    if len(np.unique(list(f.values()))) < len(list(f.values())):
        f = perturb_function(f,perturbation = 1e-6)

    sorted_f = np.sort(np.unique(list(f.values())))

    T = nx.Graph()
    merge_tree_heights = {}

    height = sorted_f[0]
    subgraph_nodes = [j for j in list(f.keys()) if f[j] <= height]
    H = G.subgraph(subgraph_nodes)

    conn_comp_list = [list(c) for c in nx.connected_components(H)]
    conn_comp_dict = {}

    for c in conn_comp_list:
        c_heights = [f[n] for n in c]
        c_max_height = max(c_heights)
        c_max_height_node = get_key(f,c_max_height)[-1]
        T.add_node(c_max_height_node)
        conn_comp_dict[c_max_height_node] = c
        merge_tree_heights[c_max_height_node] = c_max_height

    for k in range(1,len(sorted_f)):

        plt.show()

        conn_comp_dict_prev = conn_comp_dict

        height = sorted_f[k]
        subgraph_nodes = [j for j in list(f.keys()) if f[j] <= height]
        H = G.subgraph(subgraph_nodes)

        conn_comp_list = [list(c) for c in nx.connected_components(H)]
        conn_comp_dict = {}

        # Add vertices
        for c in conn_comp_list:
            c_heights = [f[n] for n in c]
            c_max_height = max(c_heights)
            c_max_height_node = get_key(f,c_max_height)[-1]
            T.add_node(c_max_height_node)
            conn_comp_dict[c_max_height_node] = c
            merge_tree_heights[c_max_height_node] = c_max_height

        # Add edges from previous level
        for child_node in conn_comp_dict_prev.keys():
            for parent_node in conn_comp_dict.keys():
                if child_node in conn_comp_dict[parent_node] and child_node != parent_node:
                    T.add_edge(child_node,parent_node)

    return T, merge_tree_heights

def get_barcodes_from_filtered_network(G,f,infinity = None):

    # Initialize with an empty simplex tree
    spCpx = gd.SimplexTree()

    # Add edges from the adjacency graph
    for edge in G.edges:
        spCpx.insert(list(edge))

    # Insert a single 2-dimensional simplex
    spCpx.insert([0,1,2])

    # Add filtration values to vertices
    zero_skeleton = spCpx.get_skeleton(0)
    for (j,spx) in enumerate(zero_skeleton):
        spCpx.assign_filtration(spx[0], filtration=f[j])

    # Extend to filtration of the whole complex
    spCpx.make_filtration_non_decreasing()

    # Compute persistence and extract barcodes
    BarCodes = spCpx.persistence()

    dgm0 = spCpx.persistence_intervals_in_dimension(0)
    dgm1 = spCpx.persistence_intervals_in_dimension(1)

    # Truncate infinite deg-1 bars to end at the maximum filtration value
    # OR the predefined value of infinity, which may be data-driven

    dgm1_fixed = []

    if infinity == None:
        max_val = np.max(list(f.values()))
    else:
        max_val = infinity

    for bar in dgm1:
        if bar[1] == np.inf:
            new_bar = [bar[0],max_val]
        else:
            new_bar = bar

        dgm1_fixed.append(new_bar)

    dgm1 = np.array(dgm1_fixed)

    return spCpx, dgm0, dgm1

def find_nearest(heights, value):
    array = np.array(list(heights.values()))
    idx = (np.abs(array - value)).argmin()
    best_val = array[idx]
    node_idx = get_key(heights,best_val)[0]
    return node_idx

def decorate_merge_tree_networks(T,heights,dgm):

    """
    Inputs: merge tree (T,height), degree-1 persistence diagram as a list of lists of birth-death pairs
    Output: dictionary of barcodes associated to each leaf of the merge tree
    """

    root = get_key(heights,max(list(heights.values())))[0]

    leaf_barcodes = {n:[] for n in T.nodes() if T.degree(n) == 1 and n != root}

    for bar in dgm:

        birth = bar[0]
        cycle_ind = find_nearest(heights,birth)

        descendent_leaves = get_descendent_leaves(T,heights,cycle_ind)
        non_descendent_leaves = [n for n in T.nodes() if T.degree(n)==1 and n not in descendent_leaves and n != root]


        non_descendent_LCAs = dict()

        for n in non_descendent_leaves:
            LCA_idx_tmp, LCA_height_tmp = least_common_ancestor(T,heights,n,cycle_ind)
            non_descendent_LCAs[n] = LCA_idx_tmp[0]

        for leaf in descendent_leaves:
            leaf_barcodes[leaf] = leaf_barcodes[leaf]+[list(bar)]

        for leaf in non_descendent_leaves:
            ancestor = non_descendent_LCAs[leaf]
            truncated_bar = truncate_bar(bar,heights[ancestor])
            if type(truncated_bar) == list:
                leaf_barcodes[leaf] = leaf_barcodes[leaf] + [list(truncated_bar)]

    return leaf_barcodes

"""
Diffusion Frechet Functions - for filtering networks by density
"""

# Find eigenvalues and vectors for graph Laplacian
def laplacian_eig(G):
    # Input: Networkx graph
    # Output: eigenvalues and eigenvectors of graph laplacian
    L = nx.laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)

    return lam, phi

# Create heat kernel matrix from precomputed eigenvalues/tangent_vectors
def heat_kernel(lam,phi,t):
    # Input: eigenvalues and eigenvectors for normalized Laplacian, time parameter t
    # Output: heat kernel matrix

    u = np.matmul(phi,np.matmul(np.diag(np.exp(-t*lam)),phi.T))

    return u

def diffusion_distance_matrix(lam,phi,t):

    dist = np.zeros([len(lam),len(lam)])

    HK = heat_kernel(lam,phi,t)

    for i in range(len(lam)):
        v1 = HK[:,i]
        for j in range(i+1,len(lam)):
            v2 = HK[:,j]
            dist[i,j] = np.linalg.norm(v1 - v2)

    dist = dist + dist.T

    return dist

def diffusion_frechet_function(dist,mu):

    f = {j:np.dot(dist[j,:]**2,mu) for j in range(dist.shape[0])}

    return f

def get_diffusion_frechet_function(G,t,mu = None):

    # Reorder the node labels to be safe
    Adj = nx.to_numpy_array(G)
    G = nx.from_numpy_array(Adj)

    # Get spectral and distributional information
    lam, phi = laplacian_eig(G)
    dist = diffusion_distance_matrix(lam,phi,t)

    if mu is None:
        mu = ot.unif(len(G))

    # Get diffusion frechet
    f = diffusion_frechet_function(dist,mu)

    return f


"""
Distance to a Node Function
"""

def distance_to_a_node(G,node_id):

    D = nx.floyd_warshall_numpy(G)

    f = {n:D[node_id,n]  for n in G.nodes()}
    f = {n:-f[n] for n in range(len(f))}
    mm = min(list(f.values()))
    f = {n:f[n] - mm for n in range(len(f))}

    return f

"""
Manipulating Merge Trees
"""

def threshold_merge_tree(T,height,thresh):

    """
    Takes a merge tree and truncates it at the given threshold level.
    Makes a cut at threshold height, removes all lower vertices.
    """

    subdiv_heights = [thresh]

    T_sub, height_sub = subdivide_edges(T,height,subdiv_heights)

    height_array = np.array(list(set(height_sub.values())))
    height_array_thresh = height_array[height_array >= thresh]

    kept_nodes = []

    for j in range(len(height_array_thresh)):
        kept_nodes += get_key(height_sub,height_array_thresh[j])

    T_thresh = T_sub.subgraph(kept_nodes).copy()
    height_thresh = {n:height_sub[n] for n in kept_nodes}

    root = get_key(height_thresh,max(list(height_thresh.values())))[0]
    T_thresh_leaves = [n for n in T_thresh.nodes() if T_thresh.degree(n) == 1 and n != root]

    for n in T_thresh_leaves:
        descendents = get_descendent_leaves(T_sub,height_sub,n)
        descendent_node_rep = list(set(get_key(height_sub,min([height[node] for node in descendents]))).intersection(set(descendents)))[0]
        T_thresh.add_edge(n,descendent_node_rep)
        height_thresh[descendent_node_rep] = height_sub[descendent_node_rep]

    return T_thresh, height_thresh

"""
Main class definition
"""

class MergeTree:

    """
    Creates a merge tree from (exactly) one of the three types of inputs:
    - T and height: the merge tree is defined directly from a nx.Graph() T (which must be
                    a tree!) and a height function on the nodes (which must satisfy
                    certain conditions). The height function should be a dictionary whose keys
                    are node labels of T.
    - pointCloud: a Euclidean point cloud of shape (num points) x (dimension). The merge tree
                    is generated from the Vietoris-Rips filtration of the point cloud
    - network and network filtration: the merge tree is created from connectivity information
                    corrsponding to the input network and filtration function on its nodes.
                    network should be a nx.Graph() object.
                    network_filtration has the following options:
                     - a dictionary whose keys are node labels from the network and values are positive numbers
                     - network_filtration = 'Diffusion', in which case the diffusion filtration function is
                    used, with a variable t_diffusion parameter
                    - network_filtration = 'Distance', in which case the distance to a specified node is used,
                    with the node chosen by the user. The filtration is *superlevel set* for this function.

    Merge trees can be 'decorated' with higher-dimensional homological data by the 'fit_barcode' method.
    The result is a 'decorated merge tree'.
    """

    def __init__(self,
                tree = None,
                height = None,
                pointCloud = None,
                network = None,
                network_filtration = None,
                t_diffusion = 0.1,
                node_distance = 0,
                simplify = True):

        self.T = tree
        self.pointCloud = pointCloud
        self.network = pointCloud
        self.network_filtration = network_filtration
        self.t_diffusion = t_diffusion
        self.node_distance = node_distance
        self.leaf_barcode = None
        self.ultramatrix = None
        self.label = None
        self.inverted_label = None

        # Define merge tree from tree/height data
        if tree is not None:

            if pointCloud is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')
            elif network is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')

            if nx.is_tree(tree):
                self.tree = tree
            else:
                raise Exception('Input Graph must be a tree')

            if set(height.keys()) == set(tree.nodes):
                self.height = height
            else:
                raise Exception('height keys must match node keys')

        # Define merge tree from point cloud data
        elif pointCloud is not None:

            if tree is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')
            elif network is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')

            L = linkage(pointCloud)
            T, height = linkage_to_merge_tree(L,pointCloud)

            self.tree = T
            self.height = height

        # Define merge tree from network data
        elif network is not None:

            if tree is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')
            elif pointCloud is not None:
                raise Exception('Only enter tree data OR point cloud data OR network data')

            if network_filtration is None:
                raise Exception('Network merge tree requires filtration---see documentation')
            elif network_filtration == 'Diffusion':
                f = get_diffusion_frechet_function(network,t_diffusion)
                T, height = get_merge_tree_from_network(network,f)
                self.filtration = f
                self.tree = T
                self.height = height

            elif network_filtration == 'Distance':
                if node_distance in list(network.nodes()):
                    f = distance_to_a_node(network,node_distance)
                    T, height = get_merge_tree_from_network(network,f)
                    self.filtration = f
                    self.tree = T
                    self.height = height
                else:
                    raise Exception('Must enter a valid node to use Distance filtration')

            elif type(network_filtration) == dict:
                if set(network_filtration.keys()) == set(network.nodes()):
                    T, height = get_merge_tree_from_network(network,network_filtration)
                    self.filtration = network_filtration
                    self.tree = T
                    self.height = height
                else:
                    raise Exception('Filtration keys must match node keys')
            else:
                raise Exception('network_filtration is not valid')


        # Cleans up the merge tree by removing degree-2 nodes
        if simplify:
            TNew, heightNew = simplify_merge_tree(self.tree,self.height)
            self.tree = TNew
            self.height = heightNew

    """
    Creating a Decorated Merge Tree
    """

    def fit_barcode(self,
                    degree = 1,
                    leaf_barcode = None):

        if leaf_barcode is not None:

            self.leaf_barcode = leaf_barcode

        else:
            if self.T is not None:
                raise Exception('fit_barcode for directly defined merge tree requires leaf_barcode input')

            if self.pointCloud is not None:
                dgm = ripser(self.pointCloud,maxdim = degree)['dgms'][-1]
                leaf_barcode_init = decorate_merge_tree(self.tree, self.height, self.pointCloud, dgm)
                leaf_barcode = {key: [bar for bar in leaf_barcode_init[key] if bar[1]-bar[0] > 0] for key in leaf_barcode_init.keys()}
                self.barcode = dgm
                self.leaf_barcode = leaf_barcode

    """
    Getting the ultramatrix from a labeling of the merge tree
    """

    def fit_ultramatrix(self,label = None):

        if label is None:
            label = {n:j for (j,n) in enumerate(self.tree.nodes())}

        ultramatrix, inverted_label = get_ultramatrix(self.tree,self.height,label)

        self.ultramatrix = ultramatrix
        self.label = label
        self.inverted_label = inverted_label

    """
    Merge tree manipulation
    """

    def threshold(self,threshold):

        if self.leaf_barcode is None:
            T_thresh, height_thresh = threshold_merge_tree(self.tree,self.height,threshold)
            self.tree = T_thresh
            self.height = height_thresh
            self.ultramatrix = None
            self.label = None
            self.inverted_label = None

        else:
            T_thresh, height_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(self.tree,self.height,self.leaf_barcode,threshold)
            self.tree = T_thresh
            self.height = height_thresh
            self.leaf_barcode = leaf_barcode_thresh
            self.ultramatrix = None
            self.label = None
            self.inverted_label = None

    def copy(self):

        return copy.copy(self)

    """
    Visualization Tools
    """

    # For general merge trees
    def draw(self, axes = False):

        draw_merge_tree(self.tree,self.height,axes = axes)

    # For merge trees coming from a network
    def draw_network(self):

        draw_network_and_function(network,self.filtration)

    def draw_network_with_merge_tree(self):

        draw_network_and_merge_tree(network,self.filtration)

    def draw_with_labels(self,label):

        draw_labeled_merge_tree(self.tree,self.height,label)


    def draw_decorated(self,tree_thresh,barcode_thresh):

        if self.pointCloud is not None:
            _, _, _, _, _ = visualize_DMT_pointcloud(self.tree,
                                                     self.height,
                                                     self.barcode,
                                                     self.pointCloud,
                                                     tree_thresh,
                                                     barcode_thresh)



"""
Plotting Functions
"""

def mergeTree_pos(G, height, root=None, width=1.0, xcenter = 0.5):

    '''
    Adapted from Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    height: dictionary {node:height} of heights for the vertices of G.
            Must satisfy merge tree conditions, but this is not checked in this version of the function.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    height_vals = list(height.values())
    max_height = max(height_vals)

    root = get_key(height,max_height)[0]
    # The root for the tree is the vertex with maximum height value

    vert_loc = max_height

    def _hierarchy_pos(G, root, vert_loc, width=1., xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                vert_loc = height[child]
                pos = _hierarchy_pos(G, child, vert_loc, width = dx, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, vert_loc, width, xcenter)

def draw_merge_tree(G,height,axes=False):
    # Input: merge tree as G, height
    # Output: draws the merge tree with correct node heights
    pos = mergeTree_pos(G,height)
    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, with_labels=True)
    if axes:
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    return

def draw_labeled_merge_tree(T,height,label,axes = False):

    # Input: merge tree as T, height. Label dictionary label with labels for certain nodes
    # Output: draws the merge tree with labels over the labeled nodes

    pos = mergeTree_pos(T,height)

    draw_labels = dict()

    for key in label.keys():
        draw_labels[key] = str(label[key])

    nx.draw_networkx(T, pos = pos, labels = draw_labels, node_color = 'r', font_weight = 'bold', font_size = 16)
    if axes:
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    return

"""
Merge Tree Processing
"""

def least_common_ancestor(G,height,vertex1,vertex2):

    height_vals = list(height.values())
    max_height = max(height_vals)
    root = get_key(height,max_height)[0]

    shortest_path1 = nx.shortest_path(G, source = vertex1, target = root)
    shortest_path2 = nx.shortest_path(G, source = vertex2, target = root)

    common_vertices = list(set(shortest_path1).intersection(set(shortest_path2)))
    LCA_height = min([height[n] for n in common_vertices])
    LCA_idx = get_key(height,LCA_height)

    return LCA_idx, LCA_height

def get_ultramatrix(T,height,label,return_inverted_label = True):

    """
    Gets an ultramatrix from a labeled merge tree.

    Input: T, height are data from a merge tree (tree structure and height function dictionary),
    label is a dictionary of node labels of the form {node:label}, where labels are given by a function
    {0,1,\ldots,N} --> T, which is surjective onto the set of leaves of T.

    Output: matrix with (i,j) entry the height of the least common ancestor of nodes labeled i and j. Optionally
    returns the inverted label dictionary {label:node}, which is useful downstream.
    """

    inverted_label = invert_label_dict(label)
    ultramatrix = np.zeros([len(label),len(label)])

    for j in range(len(label)):
        ultramatrix[j,j] = height[inverted_label[j]]

    sorted_heights = np.sort(np.unique(list(height.values())))[::-1]

    old_node = get_key(height,sorted_heights[0])[0]

    for h in sorted_heights:

        node_list = get_key(height,h)

        for node in node_list:
            T_with_node_removed = T.copy()
            T_with_node_removed.remove_node(node)
            conn_comp_list = [list(c) for c in nx.connected_components(T_with_node_removed)]
            descendent_conn_comp_list = [c for c in conn_comp_list if old_node not in c]

            for c in descendent_conn_comp_list:

                ultramatrix[label[node],[label[i] for i in c]] = h

            for j in range(len(descendent_conn_comp_list)-1):

                c = descendent_conn_comp_list[j]

                for k in range(j+1,len(descendent_conn_comp_list)):

                    cc = descendent_conn_comp_list[k]

                    for i in c:

                        ultramatrix[label[i],[label[i] for i in cc]] = h

            old_node = node

    ultramatrix = np.maximum(ultramatrix,ultramatrix.T)

    return ultramatrix, inverted_label

"""
Matching merge trees and estimating interleaving distance
"""

def get_heights(height1,height2,mesh):

    initial_heights = list(set(list(height1.values()) + list(height2.values())))
    M = max(initial_heights)
    m = min(initial_heights)
    num_samples = int(np.floor((M-m)/mesh))

    all_heights = np.linspace(m,M,num_samples+1)

    return all_heights


def subdivide_edges_single_height(G,height,subdiv_height):

    G_sub = G.copy()
    height_sub = height.copy()

    node_idx = max(G.nodes()) + 1

    for edge in G.edges():

        if (height[edge[0]] < subdiv_height and subdiv_height < height[edge[1]]) or (height[edge[1]] < subdiv_height) and (subdiv_height < height[edge[0]]):
            G_sub.add_node(node_idx)
            G_sub.add_edge(edge[0],node_idx)
            G_sub.add_edge(node_idx,edge[1])
            G_sub.remove_edge(edge[0],edge[1])
            height_sub[node_idx] = subdiv_height
            node_idx += 1

    return G_sub, height_sub

def subdivide_edges(G,height,subdiv_heights):

    for h in subdiv_heights:
        G, height = subdivide_edges_single_height(G,height,h)

    return G, height

def get_heights_and_subdivide_edges(G,height1,height2,mesh):

    all_heights = get_heights(height1,height2,mesh)

    return subdivide_edges(G,height1,all_heights)

def interleaving_subdivided_trees(T1_sub,height1_sub,T2_sub,height2_sub, verbose = True):

    # Input: data from two merge trees
    # Output: dictionary of matching data

    ####
    #Get initial data
    ####

    # Get cost matrices and dictionaries
    label1 = {n:j for (j,n) in enumerate(T1_sub.nodes())}
    label2 = {n:j for (j,n) in enumerate(T2_sub.nodes())}
    C1, idx_dict1 = get_ultramatrix(T1_sub,height1_sub,label1)
    C2, idx_dict2 = get_ultramatrix(T2_sub,height2_sub,label2)

    # Get leaf node labels
    leaf_nodes1 = [n for n in T1_sub.nodes() if T1_sub.degree(n) == 1 and n != get_key(height1_sub,max(list(height1_sub.values())))[0]]
    leaf_nodes2 = [n for n in T2_sub.nodes() if T2_sub.degree(n) == 1 and n != get_key(height2_sub,max(list(height2_sub.values())))[0]]

    # Compute coupling
    p1 = ot.unif(C1.shape[0])
    p2 = ot.unif(C2.shape[0])

    loss_fun = 'square_loss'
    d, log = ot.gromov.gromov_wasserstein2(C1,C2,p1,p2,loss_fun)
    coup = log['T']

    ####
    #Create list of matched points Pi
    ####

    Pi = []

    for leaf in leaf_nodes1:

        leaf_node = get_key(idx_dict1,leaf)[0]

        # Find where the leaf is matched
        matched_node_coup_idx = np.argmax(coup[leaf_node,:])

        # Add ordered pair to Pi
        Pi.append((leaf,idx_dict2[matched_node_coup_idx]))

    for leaf in leaf_nodes2:

        leaf_node = get_key(idx_dict2,leaf)[0]

        # Find where the leaf is matched
        matched_node_coup_idx = np.argmax(coup[:,leaf_node])

        # Add ordered pair to Pi
        Pi.append((idx_dict1[matched_node_coup_idx],leaf))

    Pi = list(set(Pi))

    ####
    # Create new ultramatrices and compute interleaving distance
    ####

    indices_1 = [label1[pair[0]] for pair in Pi]
    indices_2 = [label2[pair[1]] for pair in Pi]
    C1New = C1[indices_1,:][:,indices_1]
    C2New = C2[indices_2,:][:,indices_2]

    dist = matrix_ell_infinity_distance(C1New,C2New)
    dist_l2 = np.sqrt(np.sum((C1New - C2New)**2))

    ####
    # Collect results for output
    ####

    if verbose:
        res = dict()
        res['coupling'] = coup

        labels1New = dict()
        labels2New = dict()

        for j, pair in enumerate(Pi):

            if pair[0] in labels1New.keys():
                labels1New[pair[0]].append(j)
            else:
                labels1New[pair[0]] = [j]

            if pair[1] in labels2New.keys():
                labels2New[pair[1]].append(j)
            else:
                labels2New[pair[1]] = [j]
        res['label1'] = labels1New
        res['label2'] = labels2New

        res['ultra1'] = C1New
        res['ultra2'] = C2New
        res['dist'] = dist
        res['dist_l2'] = dist_l2
        res['dist_gw'] = d
        res['gw_log'] = log
    else:
        res = dist

    return res


def merge_tree_interleaving_distance(MT1, MT2, mesh, verbose = True, return_subdivided = False):

    T1 = MT1.tree
    height1 = MT1.height
    T2 = MT2.tree
    height2 = MT2.height

    T1_sub, height1_sub = get_heights_and_subdivide_edges(T1,height1,height2,mesh)
    T2_sub, height2_sub = get_heights_and_subdivide_edges(T2,height2,height1,mesh)

    res = interleaving_subdivided_trees(T1_sub,height1_sub,T2_sub,height2_sub,verbose = verbose)

    if return_subdivided:

        MT1_sub = MergeTree(tree = T1_sub, height = height1_sub, simplify = False)
        MT2_sub = MergeTree(tree = T2_sub, height = height2_sub, simplify = False)

        return MT1_sub, MT2_sub, res

    else:

        return res

"""
Matching decorated merge trees and estimating interleaving distance
"""

def linkage_to_merge_tree(L,X):

    nodeIDs = np.unique(L[:,:2]).astype(int)
    num_leaves = X.shape[0]

    edge_list = []
    height = dict()

    for j in range(num_leaves):
        height[j] = 0

    for j in range(L.shape[0]):
        edge_list.append((int(L[j,0]),num_leaves+j))
        edge_list.append((int(L[j,1]),num_leaves+j))
        height[num_leaves+ j] = L[j,2]

    T = nx.Graph()
    T.add_nodes_from(nodeIDs)
    T.add_edges_from(edge_list)

    return T, height

def get_descendent_leaves(T,height,vertex):

    root = get_key(height,max(list(height.values())))[0]
    leaves = [n for n in T.nodes() if T.degree(n)==1 and n != root]

    descendent_leaves = []

    for leaf in leaves:

        shortest_path = nx.shortest_path(T, source = leaf, target = root)

        if vertex in shortest_path:
            descendent_leaves.append(leaf)

    return descendent_leaves

def truncate_bar(bar,height):

    if height <= bar[0]:
        truncated_bar = bar
    elif bar[0] < height and height < bar[1]:
        truncated_bar = [height,bar[1]]
    else:
        truncated_bar = [0,0]

    return truncated_bar

def decorate_merge_tree(T, height, data, dgm):

    D = pairwise_distances(data)

    leaf_barcodes = {n:[] for n in T.nodes() if T.degree(n) == 1  and n != get_key(height,max(list(height.values())))[0]}

    for bar in dgm:

        birth = bar[0]
        cycle_inds = np.argwhere(D == D.flat[np.argmin(np.abs(D-birth))])[0]

        LCA_idx, LCA_height = least_common_ancestor(T,height,cycle_inds[0],cycle_inds[1])
        descendent_leaves = get_descendent_leaves(T,height,LCA_idx[0])
        non_descendent_leaves = [n for n in T.nodes() if T.degree(n)==1 and n not in descendent_leaves]

        non_descendent_LCAs = dict()

        for n in non_descendent_leaves:
            LCA_idx_tmp, LCA_height_tmp = least_common_ancestor(T,height,n,LCA_idx[0])
            non_descendent_LCAs[n] = LCA_idx_tmp

        for leaf in descendent_leaves:
            leaf_barcodes[leaf] = leaf_barcodes[leaf]+[list(bar)]

        for leaf in non_descendent_leaves:
            ancestor = non_descendent_LCAs[leaf][0]
            truncated_bar = truncate_bar(bar,height[ancestor])
            if type(truncated_bar) == list:
                leaf_barcodes[leaf] = leaf_barcodes[leaf] + [list(truncated_bar)]

    return leaf_barcodes

def propagate_leaf_barcodes(T,height,leaf_barcode):

    node_barcodes = {n:[] for n in T.nodes()}

    for n in T.nodes():
        descendent_leaves = get_descendent_leaves(T,height,n)
        descendent_leaf = descendent_leaves[0]
        dgm = leaf_barcode[descendent_leaf]

        node_dgm = []

        for bar in dgm:
            truncated_bar = truncate_bar(bar,height[n])
            if type(truncated_bar) == list:
                node_barcodes[n] = node_barcodes[n] + [list(truncated_bar)]

    return node_barcodes

def get_barcode_matching_matrix(node_barcode1,node_barcode2,label1,label2):

    matrix_size1 = len(list(label1.keys()))
    matrix_size2 = len(list(label2.keys()))

    M = np.zeros([matrix_size1,matrix_size2])

    for i in range(matrix_size1):
        ind1 = label1[i]
        dgm11 = node_barcode1[ind1]
        for j in range(matrix_size2):
            ind2 = label2[j]
            dgm12 = node_barcode2[ind2]
            M[i,j] = bottleneck(dgm11,dgm12)

    return M

def fusedGW_interleaving_decorated_trees(T1_sub,height1_sub,node_barcode1,
                                        T2_sub,height2_sub,node_barcode2,
                                        thresh = 1.0, alpha = 1/2, armijo = True,
                                        degree_weight = True, verbose = True):

    ####
    # Get initial data
    ####

    # Get ultramatrix cost matrices and dictionaries
    label1 = {n:i for (i,n) in enumerate(T1_sub.nodes())}
    label2 = {n:i for (i,n) in enumerate(T2_sub.nodes())}
    C1, idx_dict1 = get_ultramatrix(T1_sub,height1_sub,label1)
    C2, idx_dict2 = get_ultramatrix(T2_sub,height2_sub,label2)

    # Get persistence cost matrix
    M = get_barcode_matching_matrix(node_barcode1, node_barcode2, idx_dict1, idx_dict2)

    # Get GW data
    if degree_weight:
        p1 = np.array([1/T1_sub.degree(idx_dict1[j]) for j in list(idx_dict1.keys())])
        p1 = p1/sum(p1)
        p2 = np.array([1/T2_sub.degree(idx_dict2[j]) for j in list(idx_dict2.keys())])
        p2 = p2/sum(p2)
    else:
        p1 = ot.unif(C1.shape[0])
        p2 = ot.unif(C2.shape[0])


    # Compute FGW coupling
    dist, log = ot.gromov.fused_gromov_wasserstein2(M,C1,C2,p1,p2,alpha = alpha, armijo = armijo)
    coup = log['T']

    ####
    # Construct List of Labeled Pairs
    ####

    leaf_nodes1 = [n for n in T1_sub.nodes() if T1_sub.degree(n) == 1 and n != get_key(height1_sub,max(list(height1_sub.values())))[0]]
    leaf_nodes2 = [n for n in T2_sub.nodes() if T2_sub.degree(n) == 1 and n != get_key(height2_sub,max(list(height2_sub.values())))[0]]

    Pi = []
    for leaf_node in leaf_nodes1:

        leaf_node_idx = get_key(idx_dict1,leaf_node)[0]

        # Find where the leaf is matched
        matched_node_coup_idx = np.argmax(coup[leaf_node_idx,:])

        # Add ordered pair to Pi
        Pi.append((leaf_node,idx_dict2[matched_node_coup_idx]))

    for leaf_node in leaf_nodes2:

        leaf_node_idx = get_key(idx_dict2,leaf_node)[0]

        # Find where the leaf is matched
        matched_node_coup_idx = np.argmax(coup[:,leaf_node_idx])

        # Add ordered pair to Pi
        Pi.append((idx_dict1[matched_node_coup_idx],leaf_node))

    Pi = list(set(Pi))


    ####
    # Compute Distances
    ####

    indices_1 = [label1[pair[0]] for pair in Pi]
    indices_2 = [label2[pair[1]] for pair in Pi]
    C1New = C1[indices_1,:][:,indices_1]
    C2New = C2[indices_2,:][:,indices_2]

    distMerge = matrix_ell_infinity_distance(C1New,C2New)
    dist_l2 = np.sqrt(np.sum((C1New - C2New)**2))

    # Compute barcode matching distance
    distDgm = np.max([bottleneck(node_barcode1[pair[0]],node_barcode2[pair[1]]) for pair in Pi])
    distMax = max([distMerge,distDgm])

    ####
    # Collect results for output
    ####

    if verbose:
        res = dict()
        res['coupling'] = coup

        labels1New = dict()
        labels2New = dict()

        for j, pair in enumerate(Pi):

            if pair[0] in labels1New.keys():
                labels1New[pair[0]].append(j)
            else:
                labels1New[pair[0]] = [j]

            if pair[1] in labels2New.keys():
                labels2New[pair[1]].append(j)
            else:
                labels2New[pair[1]] = [j]
        res['label1'] = labels1New
        res['label2'] = labels2New

        res['ultra1'] = C1New
        res['ultra2'] = C2New
        res['dist'] = distMax
        res['dist_l2'] = dist_l2
        res['dist_gw'] = dist
        res['distMerge'] = distMerge
        res['distDgm'] = distDgm
        res['gw_log'] = log
    else:
        res = distMax

    return res

def DMT_interleaving_distance(MT1,MT2, mesh,
                              thresh = 1e-5, alpha = 1/2,
                              armijo = True, degree_weight = True,
                              verbose = True):

    T1 = MT1.tree
    height1 = MT1.height
    barcodes1 = MT1.leaf_barcode

    if barcodes1 is None:
        raise Exception('Merge tree must be decorated with a barcode')

    T2 = MT2.tree
    height2 = MT2.height
    barcodes2 = MT2.leaf_barcode

    if barcodes2 is None:
        raise Exception('Merge tree must be decorated with a barcode')

    T1_sub, height1_sub = get_heights_and_subdivide_edges(T1,height1,height2,mesh)
    T2_sub, height2_sub = get_heights_and_subdivide_edges(T2,height2,height1,mesh)

    node_barcode1 = propagate_leaf_barcodes(T1_sub,height1_sub,barcodes1)
    node_barcode2 = propagate_leaf_barcodes(T2_sub,height2_sub,barcodes2)

    res = fusedGW_interleaving_decorated_trees(T1_sub,height1_sub,node_barcode1,
                                        T2_sub,height2_sub,node_barcode2,
                                        thresh = thresh, alpha = alpha, armijo = armijo,
                                        degree_weight = degree_weight, verbose = verbose)

    return res


"""
The following function is from the `persim` package, with some light edits.
"""

def bottleneck(dgm1, dgm2, matching=False):
    """
    Perform the Bottleneck distance matching between persistence diagrams.
    Assumes first two columns of S and T are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).
    See the `distances` notebook for an example of how to use this.
    Parameters
    -----------
    dgm1: Mx(>=2)
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2)
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching infromation and cross-similarity matrix
    Returns
    --------
    d: float
        bottleneck distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)
    """

    return_matching = matching

    S = np.array(dgm1)
    M = min(S.shape[0], S.size)
    if S.size > 0:
        S = S[np.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            M = S.shape[0]
    T = np.array(dgm2)
    N = min(T.shape[0], T.size)
    if T.size > 0:
        T = T[np.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            N = T.shape[0]

    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1

    # Step 1: Compute CSM between S and T, including points on diagonal
    # L Infinity distance
    Sb, Sd = S[:, 0], S[:, 1]
    Tb, Td = T[:, 0], T[:, 1]
    D1 = np.abs(Sb[:, None] - Tb[None, :])
    D2 = np.abs(Sd[:, None] - Td[None, :])
    DUL = np.maximum(D1, D2)

    # Put diagonal elements into the matrix, being mindful that Linfinity
    # balls meet the diagonal line at a diamond vertex
    D = np.zeros((M + N, M + N))
    D[0:M, 0:N] = DUL
    UR = np.max(D) * np.ones((M, M))
    np.fill_diagonal(UR, 0.5 * (S[:, 1] - S[:, 0]))
    D[0:M, N::] = UR
    UL = np.max(D) * np.ones((N, N))
    np.fill_diagonal(UL, 0.5 * (T[:, 1] - T[:, 0]))
    D[M::, 0:N] = UL

    # Step 2: Perform a binary search + Hopcroft Karp to find the
    # bottleneck distance
    M = D.shape[0]
    ds = np.sort(np.unique(D.flatten()))
    bdist = ds[-1]
    matching = {}
    while len(ds) >= 1:
        idx = 0
        if len(ds) > 1:
            idx = bisect_left(range(ds.size), int(ds.size / 2))
        d = ds[idx]
        graph = {}
        for i in range(M):
            graph["%s" % i] = {j for j in range(M) if D[i, j] <= d}
        res = HopcroftKarp(graph).maximum_matching()
        if len(res) == 2 * M and d <= bdist:
            bdist = d
            matching = res
            ds = ds[0:idx]
        else:
            ds = ds[idx + 1::]

    if return_matching:
        matchidx = [(i, matching["%i" % i]) for i in range(M)]
        return bdist, (matchidx, D)
    else:
        return bdist

"""
Decorated Merge Tree Processing
"""

def threshold_decorated_merge_tree(T,height,leaf_barcode,thresh):

    """
    Takes a decorated merge tree and truncates it at the given threshold level.
    Makes a cut at threshold height, removes all lower vertices, truncates barcodes.
    """

    subdiv_heights = [thresh]

    T_sub, height_sub = subdivide_edges(T,height,subdiv_heights)
    node_barcode_sub = propagate_leaf_barcodes(T_sub,height_sub,leaf_barcode)

    height_array = np.array(list(set(height_sub.values())))
    height_array_thresh = height_array[height_array >= thresh]

    kept_nodes = []

    for j in range(len(height_array_thresh)):
        kept_nodes += get_key(height_sub,height_array_thresh[j])

    T_thresh = T_sub.subgraph(kept_nodes)
    height_thresh = {n:height_sub[n] for n in kept_nodes}
    node_barcode_thresh = {n:node_barcode_sub[n] for n in kept_nodes}

    leaf_nodes = [n for n in kept_nodes if T_thresh.degree(n) == 1 and n != get_key(height_thresh,max(list(height_thresh.values())))[0]]
    leaf_barcode_thresh = {n:node_barcode_sub[n] for n in leaf_nodes}

    return T_thresh, height_thresh, node_barcode_thresh, leaf_barcode_thresh

def simplify_decorated_merge_tree(T,height,leaf_barcode,thresh):

    """
    Simplifies a decorated merge tree as follows. Makes a cut at the threshold height.
    Below each vertex at the cut, all nodes are merged to a single leaf at the lowest
    height for that branch. The barcode for that leaf is kept.
    """

    subdiv_heights = [thresh]

    T_sub, height_sub = subdivide_edges(T,height,subdiv_heights)

    height_array = np.array(list(set(height_sub.values())))
    height_array_thresh = height_array[height_array >= thresh]

    kept_nodes = []

    for j in range(len(height_array_thresh)):
        kept_nodes += get_key(height_sub,height_array_thresh[j])

    T_thresh = T_sub.subgraph(kept_nodes).copy()
    height_thresh = {n:height_sub[n] for n in kept_nodes}

    root = get_key(height_thresh,max(list(height_thresh.values())))[0]
    T_thresh_leaves = [n for n in T_thresh.nodes() if T_thresh.degree(n) == 1 and n != root]

    for n in T_thresh_leaves:
        descendents = get_descendent_leaves(T_sub,height_sub,n)
        descendent_node_rep = list(set(get_key(height_sub,min([height[node] for node in descendents]))).intersection(set(descendents)))[0]
        T_thresh.add_edge(n,descendent_node_rep)
        height_thresh[descendent_node_rep] = height_sub[descendent_node_rep]

    leaf_nodes = [n for n in T_thresh.nodes() if T_thresh.degree(n) == 1 and n != root]
    leaf_barcode_thresh = {n:leaf_barcode[n] for n in leaf_nodes}

    return T_thresh, height_thresh, leaf_barcode_thresh

# def simplify_decorated_merge_tree(T,height,leaf_barcode,thresh):
#
#     """
#     Simplifies a decorated merge tree as follows. Makes a cut at the threshold height.
#     Below each vertex at the cut, all nodes are merged to a single leaf at the lowest
#     height for that branch. The barcode for that leaf is kept.
#     """
#
#     subdiv_heights = [thresh]
#
#     T_sub, height_sub = subdivide_edges(T,height,subdiv_heights)
#     node_barcode_sub = propagate_leaf_barcodes(T_sub,height_sub,leaf_barcode)
#
#     height_array = np.array(list(set(height_sub.values())))
#     height_array_thresh = height_array[height_array >= thresh]
#
#     kept_nodes = []
#
#     for j in range(len(height_array_thresh)):
#         kept_nodes += get_key(height_sub,height_array_thresh[j])
#
#     T_thresh = T_sub.subgraph(kept_nodes).copy()
#     height_thresh = {n:height_sub[n] for n in kept_nodes}
#
#     root = get_key(height_thresh,max(list(height_thresh.values())))[0]
#     T_thresh_leaves = [n for n in T_thresh.nodes() if T_thresh.degree(n) == 1 and n != root]
#
#     for n in T_thresh_leaves:
#         descendents = get_descendent_leaves(T_sub,height_sub,n)
#         descendent_node_rep = list(set(get_key(height_sub,min([height[node] for node in descendents]))).intersection(set(descendents)))[0]
#         T_thresh.add_edge(n,descendent_node_rep)
#         height_thresh[descendent_node_rep] = height_sub[descendent_node_rep]
#
#     node_barcode_thresh = {n:node_barcode_sub[n] for n in T_thresh.nodes()}
#
#     leaf_nodes = [n for n in T_thresh.nodes() if T_thresh.degree(n) == 1 and n != root]
#     leaf_barcode_thresh = {n:node_barcode_sub[n] for n in leaf_nodes}
#
#     return T_thresh, height_thresh, node_barcode_thresh, leaf_barcode_thresh

"""
Decorated Merge Trees for Networks
"""

def get_merge_tree_from_network(G,f):

    # The algorithm needs unique entries to work. Check here and perturb if
    # that's not the case.

    if len(np.unique(list(f.values()))) < len(list(f.values())):
        f = perturb_function(f,perturbation = 1e-6)

    sorted_f = np.sort(np.unique(list(f.values())))

    T = nx.Graph()
    merge_tree_heights = {}

    height = sorted_f[0]
    subgraph_nodes = [j for j in list(f.keys()) if f[j] <= height]
    H = G.subgraph(subgraph_nodes)

    conn_comp_list = [list(c) for c in nx.connected_components(H)]
    conn_comp_dict = {}

    for c in conn_comp_list:
        c_heights = [f[n] for n in c]
        c_max_height = max(c_heights)
        c_max_height_node = get_key(f,c_max_height)[-1]
        T.add_node(c_max_height_node)
        conn_comp_dict[c_max_height_node] = c
        merge_tree_heights[c_max_height_node] = c_max_height

    for k in range(1,len(sorted_f)):

        plt.show()

        conn_comp_dict_prev = conn_comp_dict

        height = sorted_f[k]
        subgraph_nodes = [j for j in list(f.keys()) if f[j] <= height]
        H = G.subgraph(subgraph_nodes)

        conn_comp_list = [list(c) for c in nx.connected_components(H)]
        conn_comp_dict = {}

        # Add vertices
        for c in conn_comp_list:
            c_heights = [f[n] for n in c]
            c_max_height = max(c_heights)
            c_max_height_node = get_key(f,c_max_height)[-1]
            T.add_node(c_max_height_node)
            conn_comp_dict[c_max_height_node] = c
            merge_tree_heights[c_max_height_node] = c_max_height

        # Add edges from previous level
        for child_node in conn_comp_dict_prev.keys():
            for parent_node in conn_comp_dict.keys():
                if child_node in conn_comp_dict[parent_node] and child_node != parent_node:
                    T.add_edge(child_node,parent_node)

    return T, merge_tree_heights

# def reorder_nodes(G,f):
#
#     GG = G.copy()
#     sorted_f = np.sort(list(f.values()))
#     sorted_indices = np.argsort(list(f.values()))
#     mapping = {sorted_indices[j]:n for (j,n) in enumerate(G.nodes())}
#
#     G_new = nx.relabel_nodes(GG, mapping)
#     f_new = {j:sorted_f[j] for j in range(len(sorted_f))}
#
#     return G_new, f_new

def perturb_function(f, perturbation = 1e-10):

    if len(np.unique(list(f.values()))) < len(list(f.values())):
        f = {j:f[j]+perturbation*np.random.rand() for j in f.keys()}

    return f

def filtered_network_to_merge_tree(G,f,perturbation = 1e-10):

    G, f = reorder_nodes(G,f)

    # Check for unique heights. If not, add a perturbation
    if len(np.unique(list(f.values()))) < len(G):
        f = {j:f[j]+perturbation*np.random.rand() for j in range(len(f))}


    # Initialize a tree
    T = nx.Graph()
    node_list = []
    heights = np.sort(list(f.values()))

    # Add the first vertex to the tree
    height = heights[0]
    node = 0
    T.add_node(node)

    node_list.append(node)
    H = G.subgraph(node_list)
    conn_comp_list = [list(c) for c in nx.connected_components(H)]

    # Add remaining nodes and edges to the tree iteratively
    for j in range(1,len(heights)):
        height = heights[j]
        node = j
        T.add_node(node)

        node_list.append(node)
        H = G.subgraph(node_list)

        node_component = nx.node_connected_component(H,node)


        for c in conn_comp_list:
            if set(c).issubset(node_component):
                T.add_edge(node,max(c))

        conn_comp_list = [list(c) for c in nx.connected_components(H)]

    f_new = f

    return T, f_new, heights

def get_barcodes_from_filtered_network(G,f,infinity = None):

    # Initialize with an empty simplex tree
    spCpx = gd.SimplexTree()

    # Add edges from the adjacency graph
    for edge in G.edges:
        spCpx.insert(list(edge))

    # Insert a single 2-dimensional simplex
    spCpx.insert([0,1,2])

    # Add filtration values to vertices
    zero_skeleton = spCpx.get_skeleton(0)
    for (j,spx) in enumerate(zero_skeleton):
        spCpx.assign_filtration(spx[0], filtration=f[j])

    # Extend to filtration of the whole complex
    spCpx.make_filtration_non_decreasing()

    # Compute persistence and extract barcodes
    BarCodes = spCpx.persistence()

    dgm0 = spCpx.persistence_intervals_in_dimension(0)
    dgm1 = spCpx.persistence_intervals_in_dimension(1)

    # Truncate infinite deg-1 bars to end at the maximum filtration value
    # OR the predefined value of infinity, which may be data-driven

    dgm1_fixed = []

    if infinity == None:
        max_val = np.max(list(f.values()))
    else:
        max_val = infinity

    for bar in dgm1:
        if bar[1] == np.inf:
            new_bar = [bar[0],max_val]
        else:
            new_bar = bar

        dgm1_fixed.append(new_bar)

    dgm1 = np.array(dgm1_fixed)

    return spCpx, dgm0, dgm1

def find_nearest(heights, value):
    array = np.array(list(heights.values()))
    idx = (np.abs(array - value)).argmin()
    best_val = array[idx]
    node_idx = get_key(heights,best_val)[0]
    return node_idx

def decorate_merge_tree_networks(T,heights,dgm):

    """
    Inputs: merge tree (T,height), degree-1 persistence diagram as a list of lists of birth-death pairs
    Output: dictionary of barcodes associated to each leaf of the merge tree
    """

    root = get_key(heights,max(list(heights.values())))[0]

    leaf_barcodes = {n:[] for n in T.nodes() if T.degree(n) == 1 and n != root}

    for bar in dgm:

        birth = bar[0]
        cycle_ind = find_nearest(heights,birth)

        descendent_leaves = get_descendent_leaves(T,heights,cycle_ind)
        non_descendent_leaves = [n for n in T.nodes() if T.degree(n)==1 and n not in descendent_leaves and n != root]


        non_descendent_LCAs = dict()

        for n in non_descendent_leaves:
            LCA_idx_tmp, LCA_height_tmp = least_common_ancestor(T,heights,n,cycle_ind)
            non_descendent_LCAs[n] = LCA_idx_tmp[0]

        for leaf in descendent_leaves:
            leaf_barcodes[leaf] = leaf_barcodes[leaf]+[list(bar)]

        for leaf in non_descendent_leaves:
            ancestor = non_descendent_LCAs[leaf]
            truncated_bar = truncate_bar(bar,heights[ancestor])
            if type(truncated_bar) == list:
                leaf_barcodes[leaf] = leaf_barcodes[leaf] + [list(truncated_bar)]

    return leaf_barcodes


"""
Visualization
"""

def simplify_merge_tree(T,heights):

    root = get_key(heights,max(list(heights.values())))[0]

    TNew = nx.Graph()
    leaves = [n for n in T.nodes() if T.degree(n) == 1 and n != root]
    splits = [n for n in T.nodes() if T.degree(n) > 2 and n != root]
    new_nodes = leaves + splits + [root]

    TNew.add_nodes_from(new_nodes)

    new_edges = []

    for node in new_nodes:
        shortest_path = nx.shortest_path(T, source = node, target = root)
        new_path = list(set(shortest_path).intersection(set(new_nodes)))
        new_path_dict = {n:heights[n] for n in new_path}
        if len(new_path) > 1:
            attaching_node = get_key(new_path_dict,np.sort(list(new_path_dict.values()))[1])[0]
            new_edges.append((node,attaching_node))

    TNew.add_edges_from(new_edges)

    heightsNew = {n:heights[n] for n in new_nodes}

    return TNew, heightsNew


def draw_simplified_merge_tree(T,heights,title = None, figsize = (10,10), title_fontsize = 15, y_fontsize = 12):

    TNew, heightsNew = simplify_merge_tree(T,heights)

    pos = mergeTree_pos(TNew,heightsNew)

    fig = plt.figure(figsize = figsize)
    ax = plt.subplot(111)
    nx.draw_networkx(TNew,pos = pos, with_labels = False)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.yticks(np.round(list(heightsNew.values()),2), fontsize = y_fontsize)

    if title == None:
        plt.show()
    else:
        plt.title(title, fontsize = title_fontsize)
        plt.show()

    return

def draw_network_and_function(G,f,figsize = (15,10), title = None, title_fontsize = 15):

    plt.figure(figsize = figsize)

    colors = list(f.values())
    pos=nx.kamada_kawai_layout(G)
    cmap=plt.cm.Blues
    vmin = min(list(f.values()))
    vmax = max(list(f.values()))
    nx.draw_networkx(G, pos,  node_color=colors, cmap=cmap, with_labels=False, vmin = vmin, vmax = vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.axis('off')

    if title == None:
        plt.show()
    else:
        plt.title(title,fontsize = title_fontsize)
        plt.show()

    return

def draw_network_and_merge_tree(G,f,figsize = (25,10), title = None, title_fontsize = 15, y_fontsize = 12, style = None):

    T, heights = get_merge_tree_from_network(G, f)
    TNew, heightsNew = simplify_merge_tree(T,heights)

    fig = plt.figure(figsize = figsize)

    ax1 = fig.add_subplot(1,2,1)
    colors = list(f.values())
    if style is None:
        pos=nx.kamada_kawai_layout(G)
    elif style=='kamada_kawai':
        pos=nx.kamada_kawai_layout(G)
    elif style=='networkx':
        pos=None

    cmap=plt.cm.Blues
    vmin = min(list(f.values()))
    vmax = max(list(f.values()))
    nx.draw_networkx(G, pos,  node_color=colors, cmap=cmap, with_labels=False, vmin = vmin, vmax = vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)

    if title != None:
        plt.title(title, fontsize = title_fontsize)

    plt.axis('off')

    # Add a node at infinity if necessary
    root = get_key(heightsNew,max(list(heightsNew.values())))[0]
    lowest_node = get_key(heightsNew,min(list(heightsNew.values())))[0]
    max_node = max(TNew.nodes())

    TNew.add_edges_from([(root,max_node+1)])
    inf_height = heightsNew[root]+0.1*(heightsNew[root]-heightsNew[lowest_node])
    heightsNew[max_node+1] = inf_height

    pos = mergeTree_pos(TNew,heightsNew)

    ax = fig.add_subplot(1,2,2)
    nx.draw_networkx(TNew,pos = pos, with_labels = False)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plotHeights = [np.round(min(list(heightsNew.values())),2)]

    for n in heightsNew.keys():
        height = heightsNew[n]
        if height > plotHeights[-1]+.01:
            plotHeights.append(np.round(height,2))

    plotHeightLabels = plotHeights.copy()
    plotHeightLabels[-1] = 'inf'
    plotHeightLabels = tuple(plotHeightLabels)

    plt.yticks(plotHeights, plotHeightLabels, fontsize = y_fontsize)

    if title != None:
        plt.title('Associated Merge Tree', fontsize = title_fontsize)

    plt.show()

    return

"""
With 2-Simplices
"""

def get_barcodes_from_filtered_network_with_2_simplices(G,f,dist = None, infinity = None):

    # Get distance matrix
    if dist is None:
        dist = np.array(nx.floyd_warshall_numpy(G))

    # Initialize with an empty simplex tree
    spCpx = gd.SimplexTree()

    # Add edges from the adjacency graph
    for edge in G.edges:
        spCpx.insert(list(edge))

    # Add filtration values to vertices
    zero_skeleton = spCpx.get_skeleton(0)
    for (j,spx) in enumerate(zero_skeleton):
        spCpx.assign_filtration(spx[0], filtration=f[j])

    # Extend to filtration of the whole complex
        spCpx.make_filtration_non_decreasing()

    # Insert 2-dimensional simplices
    one_skeleton = gd.RipsComplex(distance_matrix=np.array(dist))
    simplex_tree = one_skeleton.create_simplex_tree(max_dimension=2)
    skeleton_list = simplex_tree.get_skeleton(2)
    for spx in skeleton_list:
        if len(spx[0]) == 3:
            spCpx.insert(spx[0], filtration=spx[1])

    # Compute persistence and extract barcodes
    BarCodes = spCpx.persistence()

    dgm0 = spCpx.persistence_intervals_in_dimension(0)
    dgm1 = spCpx.persistence_intervals_in_dimension(1)

    # Truncate infinite deg-1 bars to end at the maximum filtration value
    # OR the predefined value of infinity, which may be data-driven

    dgm1_fixed = []

    if infinity == None:
        max_val = np.max(list(f.values()))
    else:
        max_val = infinity

    for bar in dgm1:
        if bar[1] == np.inf:
            new_bar = [bar[0],max_val]
        else:
            new_bar = bar

        dgm1_fixed.append(new_bar)

    dgm1 = np.array(dgm1_fixed)

    return spCpx, dgm0, dgm1

"""
Visualizing DMTs
"""

def visualize_DMT_pointcloud(T,height,dgm1,data,tree_thresh,barcode_thresh,offset = 0.01,draw = True,verbose = False):

    """
    In:
    - T, height define a merge tree. T is a networkx graph, height is a dictionary giving node heights.
    - dgm1 is a persistence diagram
    - tree_thresh is a user-defined threshold level. Branches of the tree below the threshold level are
        merged for improved visualization
    - barcode_thresh is a user-defined threshold level. Bars in the barcode dgm1 with length below the
        threshold level are omitted from the visualization
    - offset controls how far dgm1 bars are pushed off the merge tree. This should be adjusted to get
        the best visualization
    - draw is an option to automatically produce a plot of the DMT

    Out:
    - T_DMT is a networkx graph which simultaneously visualizes the merge tree and its barcode (as an offset
        network in red).
    - pos, edges, colors, weights are parameters used for visualizing a networkx graph. See the final code block
        here for example usage.
    """

    # Make a copy of T, height
    T_tmp = T.copy()
    height_tmp = height.copy()

    # Generate leaf barcode
    if verbose:
        print('Generating Leaf Barcode...')

    leaf_barcode = decorate_merge_tree(T, height, data, dgm1)

    # Threshold the barcode and leaf barcode according to the user-defined threshold level
    """
    TODO: find a more elegant solution to the following problem of thresholding bars
    This still gives unsatisfactory results when a bar is born lower than the tree threshold
    """
    # dgm1_thresh = []
    # for bar in dgm1:
    #     if bar[1]-bar[0] > barcode_thresh:
    #         birth = max([bar[0],tree_thresh])
    #         death = max([bar[1],tree_thresh])
    #         dgm1_thresh.append([birth,death])

    dgm1_thresh = [bar for bar in dgm1 if bar[1]-bar[0] > barcode_thresh and bar[0] > tree_thresh]

    # Add a node at infinity for display
    height_vals = list(height.values())
    max_height = max(height_vals)
    root = get_key(height,max_height)[0]

    node_inf = max(T_tmp.nodes())+1000
    T_tmp.add_node(node_inf)
    T_tmp.add_edge(root,node_inf)
    max_death = max([bar[1] for bar in dgm1_thresh])
    infinity_y_val = max([0.25*(max_height - min(list(height.values()))) + max_height,max_death])
    height_tmp[node_inf] = infinity_y_val

    # Subdivide tree at birth and death times
    new_heights = [bar[0] for bar in dgm1_thresh] + [bar[1] for bar in dgm1_thresh]
    T_sub, height_sub = subdivide_edges(T_tmp,height_tmp,new_heights)

    # Get node positions
    pos = mergeTree_pos(T_sub,height_sub)

    ### Create new graph object containing offset bars ###

    T_offsets = nx.Graph()
    pos_offsets = {}


    node_bar_counts = {n:0 for n in T_sub.nodes()}
    bar_counter = 1

    if verbose:
        print('Adding Bars...')

    for bar in dgm1_thresh:

        for leaf, barcode in leaf_barcode.items():

            if list(bar) in barcode:
                bar_leaf = leaf
                break

        birth = bar[0]
        birth_node_candidates = get_key(height_sub,birth)
        for candidate in birth_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                birth_node = candidate
                break

        death = bar[1]
        death_node_candidates = get_key(height_sub,death)
        for candidate in death_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                death_node = candidate
                break

        bar_path = nx.shortest_path(T_sub, source = birth_node, target = death_node)

        node_bar_list = [node_bar_counts[n] for n in bar_path]

        x_offset = (max(node_bar_list)+1)*offset

        for n in bar_path:
            node_bar_counts[n] += 1

        for j in range(len(bar_path)):
            node = (bar_path[j],'B'+str(bar_counter))
            T_offsets.add_node(node)
            pos_offsets[node] = x_offset

        for j in range(1,len(bar_path)):
            T_offsets.add_edge((bar_path[j],'B'+str(bar_counter)),(bar_path[j-1],'B'+str(bar_counter)),color='r',weight=2)

        bar_counter += 1

    ### Create thresholded merge tree ###

    if verbose:
        print('Creating Decorated Merge Tree...')

    # T_thresh, height_thresh, node_barcode_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(T_sub,height_sub,leaf_barcode,tree_thresh)
    T_thresh, height_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(T_sub,height_sub,leaf_barcode,tree_thresh)

    ### Create overall node positions dictionary ###
    pos_DMT = mergeTree_pos(T_thresh,height_thresh)
    for node in T_offsets.nodes():
        merge_tree_node = node[0]
        x_offset = pos_offsets[node]
        pos_DMT[node] = (pos_DMT[merge_tree_node][0] + x_offset, pos_DMT[merge_tree_node][1])

    ### Combine the two graph objects to get DMT ###
    T_DMT = nx.Graph()
    T_DMT.add_nodes_from(list(T_thresh.nodes()))
    T_DMT.add_nodes_from(list(T_offsets.nodes()))

    for edge in T_thresh.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'black', weight = 1)

    for edge in T_offsets.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'r', weight = 2)

    # Collect some display parameters for output
    edges = T_DMT.edges()
    colors = [T_DMT[u][v]['color'] for u,v in edges]
    weights = [T_DMT[u][v]['weight'] for u,v in edges]

    if draw:
        if verbose:
            print('Creating Figure...')

        plt.figure(figsize = (7,7))
        nx.draw_networkx(T_DMT, pos = pos_DMT, edge_color=colors, width=weights,node_size = 0,with_labels = False)
        ax = plt.gca()
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    return T_DMT, pos_DMT, edges, colors, weights

"""
Image Networks Tools
"""

def visualize_DMT_network(T,height,dgm1,tree_thresh,barcode_thresh,offset = 0.01,draw = True):

    """
    In:
    - T, height define a merge tree. T is a networkx graph, height is a dictionary giving node heights.
    - dgm1 is a persistence diagram
    - tree_thresh is a user-defined threshold level. Branches of the tree below the threshold level are
        merged for improved visualization
    - barcode_thresh is a user-defined threshold level. Bars in the barcode dgm1 with length below the
        threshold level are omitted from the visualization
    - offset controls how far dgm1 bars are pushed off the merge tree. This should be adjusted to get
        the best visualization
    - draw is an option to automatically produce a plot of the DMT

    Out:
    - T_DMT is a networkx graph which simultaneously visualizes the merge tree and its barcode (as an offset
        network in red).
    - pos, edges, colors, weights are parameters used for visualizing a networkx graph. See the final code block
        here for example usage.
    """

    # Make a copy of T, height
    T_tmp = T.copy()
    height_tmp = height.copy()

    # Generate leaf barcode
    leaf_barcode = decorate_merge_tree_networks(T,height,dgm1)

    # Threshold the barcode and leaf barcode according to the user-defined threshold level
    # will be changed later
    dgm1_thresh_tmp = [bar for bar in dgm1 if bar[1]-bar[0] > barcode_thresh]

    # Add a node at infinity for display
    height_vals = list(height.values())
    max_height = max(height_vals)
    root = get_key(height,max_height)[0]

    node_inf = max(T_tmp.nodes())+1000
    T_tmp.add_node(node_inf)
    T_tmp.add_edge(root,node_inf)
    max_birth = max([bar[0] for bar in dgm1_thresh_tmp])
    infinity_y_val = max([0.25*(max_height - min(list(height.values()))) + max_height,1.1*max_birth])
    height_tmp[node_inf] = infinity_y_val

    # Fix thresholded diagrams
    dgm1_thresh = [[bar[0],min(bar[1],infinity_y_val)] for bar in dgm1_thresh_tmp]

    # Subdivide tree at birth and death times
    new_heights = [bar[0] for bar in dgm1_thresh] + [bar[1] for bar in dgm1_thresh]
    T_sub, height_sub = subdivide_edges(T_tmp,height_tmp,new_heights)

    # Get node positions
    pos = mergeTree_pos(T_sub,height_sub)

    ### Create new graph object containing offset bars ###

    T_offsets = nx.Graph()
    pos_offsets = {}


    node_bar_counts = {n:0 for n in T_sub.nodes()}
    bar_counter = 1

    for j,bar in enumerate(dgm1_thresh):

        for leaf, barcode in leaf_barcode.items():

            if list(dgm1_thresh_tmp[j]) in barcode:
                bar_leaf = leaf
                break

        birth = bar[0]
        birth_node_candidates = get_key(height_sub,birth)
        for candidate in birth_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                birth_node = candidate
                break

        death = bar[1]
        death_node_candidates = get_key(height_sub,death)
        for candidate in death_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                death_node = candidate
                break

        bar_path = nx.shortest_path(T_sub, source = birth_node, target = death_node)

        node_bar_list = [node_bar_counts[n] for n in bar_path]

        x_offset = (max(node_bar_list)+1)*offset

        for n in bar_path:
            node_bar_counts[n] += 1

        for j in range(len(bar_path)):
            node = (bar_path[j],'B'+str(bar_counter))
            T_offsets.add_node(node)
            pos_offsets[node] = x_offset

        for j in range(1,len(bar_path)):
            T_offsets.add_edge((bar_path[j],'B'+str(bar_counter)),(bar_path[j-1],'B'+str(bar_counter)),color='r',weight=2)

        bar_counter += 1

    ### Create thresholded merge tree ###

    T_thresh, height_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(T_sub,height_sub,leaf_barcode,tree_thresh)

    """
    Not sure why the next line is necessary. Need to debug.
    """
    T_thresh.remove_edges_from(nx.selfloop_edges(T_thresh))

    ### Create overall node positions dictionary ###
    pos_DMT = mergeTree_pos(T_thresh,height_thresh)
    for node in T_offsets.nodes():
        merge_tree_node = node[0]
        x_offset = pos_offsets[node]
        pos_DMT[node] = (pos_DMT[merge_tree_node][0] + x_offset, pos_DMT[merge_tree_node][1])

    ### Combine the two graph objects to get DMT ###
    T_DMT = nx.Graph()
    T_DMT.add_nodes_from(list(T_thresh.nodes()))
    T_DMT.add_nodes_from(list(T_offsets.nodes()))

    for edge in T_thresh.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'black', weight = 1)

    for edge in T_offsets.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'r', weight = 2)

    # Collect some display parameters for output
    edges = T_DMT.edges()
    colors = [T_DMT[u][v]['color'] for u,v in edges]
    weights = [T_DMT[u][v]['weight'] for u,v in edges]

    if draw:
        plt.figure(figsize = (7,7))
        nx.draw_networkx(T_DMT, pos = pos_DMT, edges=edges, edge_color=colors, width=weights,node_size = 0,with_labels = False)
        ax = plt.gca()
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    return T_DMT, pos_DMT, edges, colors, weights

def visualize_DMT_network2(T,height,dgm1,tree_thresh,barcode_thresh,offset = 0.01,draw = True):

    """
    In:
    - T, height define a merge tree. T is a networkx graph, height is a dictionary giving node heights.
    - dgm1 is a persistence diagram
    - tree_thresh is a user-defined threshold level. Branches of the tree below the threshold level are
        merged for improved visualization
    - barcode_thresh is a user-defined threshold level. Bars in the barcode dgm1 with length below the
        threshold level are omitted from the visualization
    - offset controls how far dgm1 bars are pushed off the merge tree. This should be adjusted to get
        the best visualization
    - draw is an option to automatically produce a plot of the DMT

    Out:
    - T_DMT is a networkx graph which simultaneously visualizes the merge tree and its barcode (as an offset
        network in red).
    - pos, edges, colors, weights are parameters used for visualizing a networkx graph. See the final code block
        here for example usage.
    """

    # Make a copy of T, height
    T_tmp = T.copy()
    height_tmp = height.copy()

    # Generate leaf barcode
    leaf_barcode = decorate_merge_tree_networks(T,height,dgm1)

    # Threshold the barcode and leaf barcode according to the user-defined threshold level
    # will be changed later
    dgm1_thresh_tmp = [bar for bar in dgm1 if bar[1]-bar[0] > barcode_thresh]

    # Add a node at infinity for display
    height_vals = list(height.values())
    max_height = max(height_vals)
    root = get_key(height,max_height)[0]

    if len(dgm1_thresh_tmp) < 1:
        dgm1_thresh_tmp.append([max_height,max_height])

    node_inf = max(T_tmp.nodes())+1000
    T_tmp.add_node(node_inf)
    T_tmp.add_edge(root,node_inf)
    max_birth = max([bar[0] for bar in dgm1_thresh_tmp])
    infinity_y_val = max([0.25*(max_height - min(list(height.values()))) + max_height,1.1*max_birth])
    height_tmp[node_inf] = infinity_y_val

    # Fix thresholded diagrams
    dgm1_thresh = [[bar[0],min(bar[1],infinity_y_val)] for bar in dgm1_thresh_tmp]
    if len(dgm1_thresh) < 1:
        dgm1_thresh.append([max_height,max_height])

    # Subdivide tree at birth and death times
    new_heights = [bar[0] for bar in dgm1_thresh] + [bar[1] for bar in dgm1_thresh]
    T_sub, height_sub = subdivide_edges(T_tmp,height_tmp,new_heights)

    # Get node positions
    pos = mergeTree_pos(T_sub,height_sub)

    ### Create new graph object containing offset bars ###

    T_offsets = nx.Graph()
    pos_offsets = {}

    node_offsets = {n:1 for n in T_sub.nodes()}
    bar_counter = 1

    for j,bar in enumerate(dgm1_thresh):

        for leaf, barcode in leaf_barcode.items():

            if list(dgm1_thresh_tmp[j]) in barcode:
                bar_leaf = leaf
                break

        birth = bar[0]
        birth_node_candidates = get_key(height_sub,birth)
        for candidate in birth_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                birth_node = candidate
                break

        death = bar[1]
        death_node_candidates = get_key(height_sub,death)
        for candidate in death_node_candidates:
            if bar_leaf in get_descendent_leaves(T_sub,height_sub,candidate):
                death_node = candidate
                break

        bar_path = nx.shortest_path(T_sub, source = birth_node, target = death_node)

        node_offset_list = [node_offsets[n] for n in bar_path]

        max_offset = max(node_offset_list)
        x_offset = max_offset*offset

        for j in range(len(bar_path)):
            node = (bar_path[j],'B'+str(bar_counter))
            T_offsets.add_node(node)
            pos_offsets[node] = x_offset

        for j in range(1,len(bar_path)):
            T_offsets.add_edge((bar_path[j],'B'+str(bar_counter)),(bar_path[j-1],'B'+str(bar_counter)),color='r',weight=2)

        for n in bar_path:
            node_offsets[n] += max_offset

        bar_counter += 1

    ### Create thresholded merge tree ###

    T_thresh, height_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(T_sub,height_sub,leaf_barcode,tree_thresh)

    """
    Not sure why the next line is necessary. Need to debug.
    """
    T_thresh.remove_edges_from(nx.selfloop_edges(T_thresh))

    ### Create overall node positions dictionary ###
    pos_DMT = mergeTree_pos(T_thresh,height_thresh)
    for node in T_offsets.nodes():
        merge_tree_node = node[0]
        x_offset = pos_offsets[node]
        pos_DMT[node] = (pos_DMT[merge_tree_node][0] + x_offset, pos_DMT[merge_tree_node][1])

    ### Combine the two graph objects to get DMT ###
    T_DMT = nx.Graph()
    T_DMT.add_nodes_from(list(T_thresh.nodes()))
    T_DMT.add_nodes_from(list(T_offsets.nodes()))

    for edge in T_thresh.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'black', weight = 1)

    for edge in T_offsets.edges():
        T_DMT.add_edge(edge[0],edge[1],color = 'r', weight = 2)

    # Collect some display parameters for output
    edges = T_DMT.edges()
    colors = [T_DMT[u][v]['color'] for u,v in edges]
    weights = [T_DMT[u][v]['weight'] for u,v in edges]

    if draw:
        plt.figure(figsize = (7,7))
        nx.draw_networkx(T_DMT, pos = pos_DMT, edges=edges, edge_color=colors, width=weights,node_size = 0,with_labels = False)
        ax = plt.gca()
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    return T_DMT, pos_DMT, edges, colors, weights

def get_barcodes_from_filtered_network_with_2_simplices_image_network(G,f,dist = None, infinity = None):

    # Get distance matrix
    if dist is None:
        dist = np.array(nx.floyd_warshall_numpy(G))

    # Initialize with an empty simplex tree
    spCpx = gd.SimplexTree()

    # Add edges from the adjacency graph
    for edge in G.edges:
        spCpx.insert(list(edge))

    # Add filtration values to vertices
    zero_skeleton = spCpx.get_skeleton(0)
    for (j,spx) in enumerate(zero_skeleton):
        spCpx.assign_filtration(spx[0], filtration=f[j])

    # Extend to filtration of the whole complex
        spCpx.make_filtration_non_decreasing()

    # Insert 2-dimensional simplices
    one_skeleton = gd.RipsComplex(distance_matrix=np.array(dist),max_edge_length=1.1)
    simplex_tree = one_skeleton.create_simplex_tree(max_dimension=2)
    skeleton_list = simplex_tree.get_skeleton(2)

    skelList = [spx for spx in skeleton_list if len(spx[0])==3 and spx[1] < 1.1]

    for spx in skelList:
        spx_val = min([f[n] for n in spx[0]])
        spCpx.insert(spx[0], filtration=spx_val)

    spCpx.make_filtration_non_decreasing()

    # Compute persistence and extract barcodes
    BarCodes = spCpx.persistence()

    dgm0 = spCpx.persistence_intervals_in_dimension(0)
    dgm1 = spCpx.persistence_intervals_in_dimension(1)

    # Truncate infinite deg-1 bars to end at the maximum filtration value
    # OR the predefined value of infinity, which may be data-driven

    dgm1_fixed = []

    if infinity == None:
        max_val = np.max(list(f.values()))
    else:
        max_val = infinity

    for bar in dgm1:
        if bar[1] == np.inf:
            new_bar = [bar[0],max_val]
        else:
            new_bar = bar

        dgm1_fixed.append(new_bar)

    dgm1 = np.array(dgm1_fixed)

    return spCpx, dgm0, dgm1

def simplify_merge_treeV2(T,heights):

    TNew = T.copy()
    deg2Nodes = [n for n in T.nodes() if T.degree(n) == 2]

    for node in deg2Nodes:
        neighbors = [n for n in TNew.neighbors(node)]
        TNew.remove_node(node)
        TNew.add_edge(neighbors[0],neighbors[1])

    heightsNew = {n:heights[n] for n in TNew.nodes()}

    return TNew, heightsNew

def simplify_merge_tree_image_network(T,height,tol = 1e-2):

    T_simplified_1, height_simplified_1 = simplify_merge_treeV2(T,height)

    leaves = [n for n in T_simplified_1.nodes() if T_simplified_1.degree(n) == 1 and n != get_key(height,max(list(height.values())))[0]]
    leaf_lengths = [np.abs(height[[n for n in T_simplified_1.neighbors(n)][0]] - height[n]) for n in leaves]
    iteration = 0

    while sum(np.array(leaf_lengths) < tol) > 0 and iteration < 10000:

        for leaf in leaves:
            neighbor = [n for n in T_simplified_1.neighbors(leaf)][0]
            if np.abs(height[neighbor] - height[leaf]) < tol:
                T_simplified_1.remove_node(leaf)
                height_simplified_1[neighbor] = height[neighbor]

        T_simplified_1, height_simplified_1 = simplify_merge_treeV2(T_simplified_1,height_simplified_1)

        leaves = [n for n in T_simplified_1.nodes() if T_simplified_1.degree(n) == 1 and n != get_key(height,max(list(height.values())))[0]]
        leaf_lengths = [np.abs(height[[n for n in T_simplified_1.neighbors(n)][0]] - height[n]) for n in leaves]

        iteration += 1

    T_simplified = T_simplified_1
    height_simplified = {n:max([height_simplified_1[n] - tol,0]) for n in T_simplified.nodes()}

    return T_simplified, height_simplified
