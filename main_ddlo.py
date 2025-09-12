import scipy.io as sio
import numpy as np
import time
from memory import Memory
from MUMT import MUMT
import matplotlib.pyplot as plt

def plot_gain(gain_his, name=None):
    import pandas as pd
    import matplotlib as mpl
    mpl.style.use('seaborn-v0_8')
    df = pd.DataFrame(gain_his)
    rolling_intv = 60
    df_roll = df.rolling(rolling_intv, min_periods=1).mean()
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(len(df_roll))+1, df_roll, 'b')
    plt.fill_between(np.arange(len(df_roll))+1,
                     df.rolling(rolling_intv, min_periods=1).min()[0],
                     df.rolling(rolling_intv, min_periods=1).max()[0],
                     color='b', alpha=0.2)
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()
    if name:
        sio.savemat('./data/MUMT(%s).mat' % name, {'ratio': gain_his})

if __name__ == "__main__":
    # parameters
    N = 20000
    net_num = 3
    WD_num = 3
    task_num = 3

    # load .mat
    mat = sio.loadmat('./data/MUMT_data_3x3.mat')
    if 'task_size' in mat:
        task_size_all = mat['task_size']
    else:
        raise ValueError("task_size not found. Keys: %s" % list(mat.keys()))
    if 'gain_min' in mat:
        gain = mat['gain_min']
    else:
        print("Warning: gain_min not found. Using placeholder.")
        gain = np.ones((1, task_size_all.shape[0]))

    # split 80:20
    split_idx = int(0.8 * len(task_size_all))
    num_test = min(len(task_size_all) - split_idx, N - int(0.8 * N))

    # build memory and env
    mem = Memory(net=[WD_num*task_num, 120, 80, WD_num*task_num],
                 net_num=net_num,
                 learning_rate=0.01,
                 training_interval=10,
                 batch_size=128,
                 memory_size=1024)

    env = MUMT(3,3,rand_seed=1)

    gain_his = []
    gain_his_ratio = []
    knm_idx_his = []
    m_li = []

    start_time = time.time()
    for i in range(N):
        if i % (N//100) == 0:
            print("progress: %0.2f%%" % (100.0 * i / N))

        # chọn sample cho train/test
        if i < N - num_test:
            i_idx = i % split_idx
        else:
            i_idx = i - N + num_test + split_idx

        t1 = task_size_all[i_idx, :].astype(float).reshape(-1)
        # pretreatment
        t = t1 * 10.0 - 200.0

        # sinh quyết định offloading
        m_list = mem.decode(t)
        m_li.append(m_list)

        r_list = []
        for m in m_list:
            r_val = env.compute_Q(t1, m)
            # nếu compute_Q trả về tuple (MUMT mới)
            if isinstance(r_val, tuple):
                r_val = r_val[0]
            r_list.append(r_val)

        # chọn best (minimization)
        best_idx = int(np.argmin(r_list))
        best_m = m_list[best_idx]
        mem.encode(t, best_m)

        # record
        gain_his.append(np.min(r_list))
        knm_idx_his.append(best_idx)
        try:
            gain_value = gain[0][i_idx]
        except Exception:
            gain_value = gain.flatten()[i_idx] if gain.size > i_idx else 1.0
        gain_his_ratio.append(gain_value / gain_his[-1] if gain_his[-1] != 0 else 1.0)

    total_time = time.time() - start_time
    print("time_cost: %.2f s" % total_time)
    if num_test > 0:
        test_ratios = gain_his_ratio[-num_test: -1] if num_test > 1 else gain_his_ratio[-1:]
        print("gain/max ratio of test: ", sum(test_ratios) / max(1, len(test_ratios)))
    print("The number of net: ", net_num)

    mem.plot_cost()
    plot_gain(gain_his_ratio)
#
# import numpy as np
# import time
# from memory import Memory
#
# if __name__ == "__main__":
#     # parameters
#     N = 5000           # số bước huấn luyện
#     net_num = 3        # số mạng ensemble
#     input_dim = 9      # kích thước đầu vào giả lập
#     output_dim = 9     # kích thước đầu ra giả lập
#
#     # build memory
#     mem = Memory(net=[input_dim, 120, 80, output_dim],
#                  net_num=net_num,
#                  learning_rate=0.01,
#                  training_interval=10,
#                  batch_size=128,
#                  memory_size=1024)
#
#     start_time = time.time()
#     for i in range(N):
#         # sinh dữ liệu ngẫu nhiên
#         h = np.random.randn(input_dim).astype(np.float32)   # input features
#         m = (np.random.rand(output_dim) > 0.5).astype(np.float32)  # nhãn nhị phân
#
#         # huấn luyện
#         mem.encode(h, m)
#
#         if i % (N//10) == 0:
#             print(f"progress: {100*i/N:.1f}%")
#
#     total_time = time.time() - start_time
#     print("time_cost: %.2f s" % total_time)
#     print("The number of net: ", net_num)
#
#     # vẽ loss
#     mem.plot_cost()
