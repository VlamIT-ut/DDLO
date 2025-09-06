
import numpy as np
import time
from memory import Memory

if __name__ == "__main__":
    # parameters
    N = 5000           # số bước huấn luyện
    net_num = 3        # số mạng ensemble
    input_dim = 9      # kích thước đầu vào giả lập
    output_dim = 9     # kích thước đầu ra giả lập

    # build memory
    mem = Memory(net=[input_dim, 120, 80, output_dim],
                 net_num=net_num,
                 learning_rate=0.01,
                 training_interval=10,
                 batch_size=128,
                 memory_size=1024)

    start_time = time.time()
    for i in range(N):
        # sinh dữ liệu ngẫu nhiên
        h = np.random.randn(input_dim).astype(np.float32)   # input features
        m = (np.random.rand(output_dim) > 0.5).astype(np.float32)  # nhãn nhị phân

        # huấn luyện
        mem.encode(h, m)

        if i % (N//10) == 0:
            print(f"progress: {100*i/N:.1f}%")

    total_time = time.time() - start_time
    print("time_cost: %.2f s" % total_time)
    print("The number of net: ", net_num)

    # vẽ loss
    mem.plot_cost()
