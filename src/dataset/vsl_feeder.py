import pickle
import logging
import numpy as np
import os

from torch.utils.data import Dataset

# Thiết lập logging cơ bản (tùy chọn, bạn có thể bỏ qua nếu không muốn)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class VSL_Feeder(Dataset): # Đổi tên lớp cho phù hợp với dataset của bạn
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint # Mối nối giữa các khớp (cho tính toán bone)
        self.debug = debug

        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        # File class_name.pkl chứa tên của các lớp, không phải dữ liệu mẫu
        # self.name thường được dùng để tham chiếu đến tên file gốc hoặc id của mẫu
        # self.seq_len thường là độ dài thực tế của chuỗi trước khi padding
        # Nếu bạn không sử dụng các giá trị này trong mô hình của mình, bạn có thể bỏ qua việc tải/khởi tạo chúng.
        # Tuy nhiên, hàm __getitem__ gốc trả về 'name', nên chúng ta cần giữ nó.
        # Tôi sẽ giả định rằng 'name' có thể được lấy từ 'class_name.pkl' hoặc tạo ngẫu nhiên.
        # Ở đây tôi sẽ tạo tên giả định và seq_len cố định là T.
        class_name_path = os.path.join(dataset_path, 'class_names.npy') # Lấy từ global class_names.npy

        try:
            self.data = np.load(data_path, mmap_mode='r')
            
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f) # Tải danh sách nhãn (list of int)
            
            # Tải global class names từ file .npy đã lưu
            self.class_names_map = np.load(class_name_path, allow_pickle=True)
            
            # Khởi tạo self.name và self.seq_len
            # Tên mẫu có thể không cần thiết nếu bạn chỉ cần nhãn và dữ liệu.
            # Nhưng vì __getitem__ trả về name, chúng ta giữ nó.
            self.name = ['sample_{:06d}'.format(i) for i in range(len(self.label))] # Tạo tên mẫu giả định
            self.seq_len = [self.T] * len(self.label) # Giả sử tất cả các chuỗi đều có độ dài T (đã được pad/truncate)

        except Exception as e:
            logging.error(f'Error: Failed to load data files: {data_path}, {label_path}, or {class_name_path}!')
            logging.error(f'Please ensure these files exist and are correctly formatted.')
            logging.error(f'Details: {e}')
            raise ValueError(f"Data loading failed: {e}")
        
        if self.debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300] # Nếu không dùng, có thể bỏ

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # self.data[idx] có shape (C, T, V, M)
        data = np.array(self.data[idx]) 
        label = self.label[idx]
        name = self.name[idx] # Sử dụng tên mẫu đã tạo
        # seq_len = self.seq_len[idx] # Nếu không dùng, có thể bỏ

        # Hàm multi_input sẽ nhận data đã có shape (C, T, V, M)
        joint, velocity, bone = self.multi_input(data) 
        
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        
        # Nếu inputs là ['J','V','B'], data_new sẽ có shape (3, C*2, T, V, M)
        data_new = np.stack(data_new, axis=0) 

        return data_new, label, name

    def multi_input(self, data):
        # Data đầu vào đã có shape (C, T, V, M)
        C, T, V, M = data.shape 
        
        joint = np.zeros((C*2, T, V, M), dtype=data.dtype) 
        velocity = np.zeros((C*2, T, V, M), dtype=data.dtype)
        bone = np.zeros((C*2, T, V, M), dtype=data.dtype)

        # 1. Joint-based input (J)
        joint[:C,:,:,:] = data 
        
        # Khớp gốc để tính tọa độ tương đối (ví dụ: Spine Base trong NTU)
        # **QUAN TRỌNG**: joint 1 (chỉ mục 1) phải là khớp gốc/tâm phù hợp trong sơ đồ 61 khớp của bạn.
        # Nếu khớp 42 ("Nose") là center_joint của bạn, hãy dùng nó.
        base_joint_idx = 42 # Giả sử khớp 42 là gốc cho việc này
        for i in range(V):
             joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,base_joint_idx,:]

        # 2. Velocity-based input (V)
        # Vận tốc: delta(t+1) - delta(t)
        velocity[:C, :T-1, :, :] = data[:, 1:, :, :] - data[:, :T-1, :, :]
        # Vận tốc: delta(t+2) - delta(t)
        velocity[C:, :T-2, :, :] = data[:, 2:, :, :] - data[:, :T-2, :, :]
        
        # 3. Bone-based input (B)
        # Xương: vector từ khớp cha đến khớp con
        # **QUAN TRỌNG**: self.conn PHẢI được định nghĩa đúng với sơ đồ khớp của bạn.
        # Ví dụ: self.conn[i] là chỉ mục khớp cha của khớp 'i'.
        # Nếu bạn không có mối nối khớp chính xác, tính toán này sẽ không đúng.
        if self.conn is not None:
            for i, parent_idx in enumerate(self.conn):
                if i < V and parent_idx < V: # Đảm bảo chỉ mục nằm trong giới hạn
                    bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, parent_idx, :]
            
            # Tính độ dài vector xương (từ x,y,z của xương)
            bone_length = np.sqrt(np.sum(bone[:C,:,:,:] ** 2, axis=0, keepdims=True)) + 1e-5
            
            # Tính "góc" xương (biến đổi từ vector xương sang dạng khác)
            for i in range(C):
                # Sử dụng bone_length[0,:,:,:] vì sum(axis=0) làm giảm một chiều
                bone[C+i,:,:,:] = np.arccos(np.clip(bone[i,:,:,:] / bone_length[0,:,:,:], -1.0, 1.0))

        return joint, velocity, bone