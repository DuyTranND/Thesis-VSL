import logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph():
    def __init__(self, dataset, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),              # left_arm
                np.array([2, 3, 4]),              # right_arm
                np.array([11, 12, 13]),           # left_leg
                np.array([8, 9, 10]),             # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
        elif self.dataset == 'ntu':
            num_node = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4, 21]) - 1          # torso
            ]
        elif self.dataset == 'sysu':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,      # left_arm
                np.array([9, 10, 11, 12]) - 1,   # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1       # torso
            ]
        elif self.dataset == 'ucla':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,      # left_arm
                np.array([9, 10, 11, 12]) - 1,   # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1       # torso
            ]
        elif self.dataset == 'cmu':
            num_node = 26
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (12, 13), (13, 14), (12, 15), (15, 16),
                              (16, 17), (17, 18), (18, 19), (17, 20), (12, 21),
                              (21, 22), (22, 23), (23, 24), (24, 25), (23, 26)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,10,10,10,11,12,13,12,15,16,17,18,17,12,21,22,23,24,23]) - 1
            parts = [
                np.array([15, 16, 17, 18, 19, 20]) - 1,  # left_arm
                np.array([21, 22, 23, 24, 25, 26]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,              # left_leg
                np.array([5, 6, 7, 8]) - 1,              # right_leg
                np.array([9, 10, 11, 12, 13, 14]) - 1    # torso
            ]
        elif self.dataset == 'h36m':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (10, 13), (13, 14), (14, 15), (15, 16),
                              (10, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,9,9,10,11,10,13,14,15,10,17,18,19]) - 1
            parts = [
                np.array([13, 14, 15, 16]) - 1,  # left_arm
                np.array([17, 18, 19, 20]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,      # left_leg
                np.array([5, 6, 7, 8]) - 1,      # right_leg
                np.array([9, 10, 11, 12]) - 1    # torso
            ]
##############################################################################################
        elif self.dataset == 'mediapipe61':
            # 1) số khớp
            num_node = 61

            # 2) danh sách cạnh 0-index
            # --- BODY: 19 joints (giữ lại 0-16 + 23,24 trong pose)
            body_edges = [
                (0,1), (1,2), (2,3),
                (0,4), (4,5), (5,6),
                (3,7),               # left ear
                (6,8),               # right ear
                (9,10),              # mouth
                (11,12),             # shoulder-to-shoulder
                (11,13), (13,15),    # L-shoulder → L-elbow → L-wrist
                (12,14), (14,16),    # R-shoulder → R-elbow → R-wrist
                (11,17), (12,18), (17,18)  # hips
            ]

            # --- HAND template (MediaPipe numbering 0–20)
            hand_edges_tpl = [
                (0,1),(1,2),(2,3),(3,4),        # thumb
                (0,5),(5,6),(6,7),(7,8),        # index
                (0,9),(9,10),(10,11),(11,12),   # middle
                (0,13),(13,14),(14,15),(15,16), # ring
                (0,17),(17,18),(18,19),(19,20)  # pinky
            ]

            # offset cho tay
            lh_off = 19
            rh_off = 40
            left_hand_edges  = [(a+lh_off, b+lh_off) for a, b in hand_edges_tpl]
            right_hand_edges = [(a+rh_off, b+rh_off) for a, b in hand_edges_tpl]

            neighbor_link = body_edges + left_hand_edges + right_hand_edges

            # 3) connect_joint: mảng cha-con (-1 cho gốc)
            connect_joint = np.full(num_node, -1, dtype=int)
            # đảm bảo hướng (parent, child)
            for parent, child in neighbor_link:
                connect_joint[child] = parent

            # 4) parts: nhóm khớp (dùng khi muốn highlight vùng)
            parts = [
                np.array([15] + list(range(19, 40))),  # left arm + left hand
                np.array([16] + list(range(40, 61))),  # right arm + right hand
                np.array([0,1,2,3,4,5,6,7,8,11,12]),   # upper-body / head
                np.array([17,18]),                     # hips
                np.array([9,10])                       # mouth
            ]
##############################################################################################
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

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
