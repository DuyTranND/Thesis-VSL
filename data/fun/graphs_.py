import logging, numpy as np
from collections import deque

class Graph():
    def __init__(self, dataset, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts, self.directed_edges = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset != 'mediapipe61':
            logging.error(f"Error: Dataset '{self.dataset}' not supported")
            raise ValueError(f"Unknown dataset {self.dataset}")

        num_node = 61
        
        roots = [42, 0, 21] 
        
        print(f"Init connect_joint {num_node} node.")

        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (0,17),(13,17),(17,18),(18,19),(19,20),
        ]

        POSE_CONNECTIONS = [
            # Face
            (42, 43), (43, 44), (44, 45), (45, 49),
            (42, 46), (46, 47), (47, 48), (48, 50), 
            # mouth
            (51, 52), (42, 51), (42, 52), (42, 53), (42, 54),
            # Body
            (53, 54), (53, 59), (54, 60), (59, 60),
            # Arms
            (53, 55), (55, 57), (54, 56), (56, 58)
        ]

        neighbor_link = []

        # left hand edges
        for a, b in HAND_CONNECTIONS:
            neighbor_link.append((a, b))
        # right hand edges
        for a, b in HAND_CONNECTIONS:
            neighbor_link.append((a + 21, b + 21))
        # body edges
        for a, b in POSE_CONNECTIONS:
            neighbor_link.append((a, b))

        neighbor_link.append((58, 0)) # left wrist to body left wrist
        neighbor_link.append((57, 21)) # right wrist to body right wrist
        
        adj = {i: [] for i in range(num_node)}
        for u, v in neighbor_link:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        directed_edges = []
        queue = deque(roots) # Bắt đầu duyệt từ các gốc
        visited.update(roots)

        while queue:
            parent = queue.popleft()
            for child in adj[parent]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    # Vì ta duyệt từ gốc ra, nên `parent` luôn gần trung tâm hơn `child`
                    directed_edges.append((parent, child))
        
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + directed_edges

        # initialize all to -1 (unset)
        connect_joint = np.full(num_node, -1, dtype=np.int32)
        # designate roots (wrist L=0, wrist R=21, nose/body root=42)
        for r in roots:
            connect_joint[r] = r
        # assign parent for each neighbor edge one‐way only
        for parent, child in directed_edges:
            connect_joint[child] = parent
        
        parts = [
            np.arange(0, 21),    # left hand
            np.arange(21, 42),   # right hand
            np.arange(42, 61),   # body
        ]

        return num_node, edge, connect_joint, parts, directed_edges

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
