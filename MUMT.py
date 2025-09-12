import numpy as np

class MUMT:
    def __init__(self, N:int, M:int, rand_seed: int = 1):
        self.N = int(N)
        self.M = int(M)
        np.random.seed(rand_seed)
        # parameters (copied from original)
        self.APP = 1900.0
        self.fc  = 10.0 * 1e9
        self.p   = 1.0
        self.a   = 1.5e-7
        self.et  = 1.42e-7
        self.El  = 3.25e-7
        self.Tl  = 4.75e-7
        self.CUL = 100.0 / 8.0
        self.CDL = 100.0 / 8.0
        self.cu = np.full(self.N, self.CUL / float(self.N))
        self.cd = np.full(self.N, self.CDL / float(self.N))
        self._rand_datain = np.random.randint(10, 31, size=(self.N, self.M))

    def compute_Q(self, task_sizes, decisions):
        task_sizes = np.asarray(task_sizes).reshape(self.N, self.M)
        X = np.asarray(decisions).reshape(self.N, self.M).astype(int)
        data_bits = task_sizes * (8.0 * 2**20)
        El = data_bits * self.El
        et = data_bits * self.et
        d = data_bits
        EC = d * self.a + et
        TL = data_bits * self.Tl
        Tc = data_bits * self.APP / self.fc

        utils_task = np.zeros((self.N, self.M))
        lat_task = np.zeros((self.N, self.M))
        energy_task = np.zeros((self.N, self.M))
        Q_sum = 0.0
        for i in range(self.N):
            xi = X[i, :]
            db = data_bits[i, :]
            el_i = El[i, :]
            ec_i = EC[i, :]
            tl_i = TL[i, :]
            tc_i = Tc[i, :]
            cu_i = self.cu[i] if self.cu.size == self.N else self.cu[0]
            sum1_tasks = el_i * (1 - xi) + ec_i * xi
            temp1 = np.sum(tl_i * (1 - xi))
            temp2 = np.sum((task_sizes[i, :] / cu_i + tc_i) * xi)
            user_sum = np.sum(sum1_tasks) + self.p * max(temp1, temp2)
            Q_sum += user_sum
            utils_task[i, :] = sum1_tasks
            lat_task[i, :] = xi * (task_sizes[i, :] / cu_i + tc_i) + (1 - xi) * tc_i
            energy_task[i, :] = sum1_tasks
        return Q_sum, utils_task, lat_task, energy_task
