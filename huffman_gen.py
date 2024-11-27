import torch
import heapq
from collections import defaultdict

class Node:
    def __init__(self, prob, symbol=None):
        self.prob = prob
        self.symbol = symbol
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.prob < other.prob

def huffman_coding(prob_vector):
    # Create initial priority queue
    pq = []
    for i, prob in enumerate(prob_vector):
        heapq.heappush(pq, Node(prob, i))

    # Build Huffman tree
    while len(pq) > 1:
        node1 = heapq.heappop(pq)
        node2 = heapq.heappop(pq)
        merged_prob = node1.prob + node2.prob
        merged_node = Node(merged_prob)
        merged_node.left = node1
        merged_node.right = node2
        heapq.heappush(pq, merged_node)

    # Assign codewords to symbols
    def assign_codewords(node, code, codewords):
        if node.symbol is not None:
            codewords[node.symbol] = code
        else:
            assign_codewords(node.left, code + "0", codewords)
            assign_codewords(node.right, code + "1", codewords)

    huffman_tree = heapq.heappop(pq)
    codewords = {}
    assign_codewords(huffman_tree, "", codewords)

    # Calculate average codeword length
    avg_codeword_length = 0
    for i, prob in enumerate(prob_vector):
        avg_codeword_length += prob * len(codewords[i])

    return avg_codeword_length