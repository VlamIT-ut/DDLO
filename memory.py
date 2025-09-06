# PyTorch multi-network MemoryDNN (DDLO style)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

class SingleNet(nn.Module):
    def __init__(self, input_dim:int, h1:int, h2:int, output_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class Memory:
    def __init__(self,
                 net: List[int],
                 net_num: int = 3,
                 learning_rate: float = 0.01,
                 training_interval: int = 10,
                 batch_size: int = 128,
                 memory_size: int = 1024,
                 device: Optional[str] = None):
        self.net = net
        self.net_num = int(net_num)
        self.training_interval = int(training_interval)
        self.lr = float(learning_rate)
        self.batch_size = int(batch_size)
        self.memory_size = int(memory_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_counter = 0
        # memory: rows of [h (input_dim) | m (output_dim)]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]), dtype=np.float32)

        # build net_num models + optimizers + loss histories
        self.models: List[SingleNet] = []
        self.opts = []
        self.cost_his: List[List[float]] = [[] for _ in range(self.net_num)]
        self.criteria = []

        for i in range(self.net_num):
            m = SingleNet(self.net[0], self.net[1], self.net[2], self.net[3]).to(self.device)
            opt = optim.Adam(m.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            self.models.append(m)
            self.opts.append(opt)
            self.criteria.append(nn.BCELoss())

        # used for enumerate actions (knn mode); only for small output_dim
        self.enumerate_actions = None

    def remember(self, h: np.ndarray, m: np.ndarray):
        """Store one sample (h,m) into circular memory."""
        assert h.ndim == 1 and m.ndim == 1
        assert h.size == self.net[0] and m.size == self.net[-1]
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h.astype(np.float32), m.astype(np.float32)))
        self.memory_counter += 1

    def encode(self, h: np.ndarray, m: np.ndarray):
        """
        Store and trigger learn periodically.
        h: input feature vector (1D)
        m: binary label vector chosen (1D)
        """
        self.remember(h, m)
        # learn every training_interval steps (if enough samples)
        if self.memory_counter > 0 and (self.memory_counter % self.training_interval == 0):
            self.learn()

    def learn(self):
        """Train each network on its own random minibatch."""
        filled = min(self.memory_counter, self.memory_size)
        if filled == 0:
            return

        for i in range(self.net_num):
            # sample indices (without replacement if possible)
            if filled >= self.batch_size:
                idxs = np.random.choice(filled, size=self.batch_size, replace=False)
            else:
                idxs = np.random.choice(filled, size=self.batch_size, replace=True)
            batch = self.memory[idxs, :]
            h_batch = torch.tensor(batch[:, :self.net[0]], dtype=torch.float32, device=self.device)
            m_batch = torch.tensor(batch[:, self.net[0]:], dtype=torch.float32, device=self.device)

            self.models[i].train()
            self.opts[i].zero_grad()
            out = self.models[i](h_batch)
            loss = self.criteria[i](out, m_batch)
            loss.backward()
            self.opts[i].step()
            loss_val = float(loss.item())
            self.cost_his[i].append(loss_val)

    def decode(self, h: np.ndarray, mode: str = 'OP'):
        """
        For given input h (1D), produce one candidate per network.
        Returns list length net_num, each entry is a binary numpy array shape (output_dim,)
        mode: 'OP' can be used later to expand to knm; here we return simple thresholding per-net.
        """
        x = torch.tensor(h[np.newaxis, :], dtype=torch.float32, device=self.device)
        m_list = []
        for i in range(self.net_num):
            self.models[i].eval()
            with torch.no_grad():
                out = self.models[i](x).cpu().numpy()[0]
            bin_out = (out > 0.5).astype(int)
            m_list.append(bin_out)
        return m_list

    # Optional: knm across averaged prediction (like order-preserving)
    def decode_knm(self, h: np.ndarray, k:int = 1):
        """
        Produce k candidates using averaged ensemble probability then knm (order-preserving).
        Returns list of k binary arrays.
        """
        x = torch.tensor(h[np.newaxis, :], dtype=torch.float32, device=self.device)
        probs = None
        for i in range(self.net_num):
            self.models[i].eval()
            with torch.no_grad():
                out = self.models[i](x).cpu().numpy()[0]
            if probs is None:
                probs = out
            else:
                probs += out
        probs /= float(self.net_num)
        # first candidate
        cand_list = []
        cand_list.append((probs > 0.5).astype(int))
        if k > 1:
            m_abs = np.abs(probs - 0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for idx in idx_list:
                if probs[idx] > 0.5:
                    cand = (probs - probs[idx] > 0).astype(int)
                else:
                    cand = (probs - probs[idx] >= 0).astype(int)
                cand_list.append(cand)
        return cand_list

    def plot_cost(self):
        import matplotlib.pyplot as plt
        import numpy as np
        tmax = max(len(ch) for ch in self.cost_his)
        for i, ch in enumerate(self.cost_his):
            plt.plot(np.arange(len(ch)) * self.training_interval, ch, label=f'net{i}')
        plt.xlabel('Time Frames')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.show()
