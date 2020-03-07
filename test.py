import matplotlib.pyplot as plt

from utils.save import generate_path

# path = generate_path("data/train_result", "tmp", ".png")
# a = [i for i in range(100)]
# b = [(i * 2 +1) for i in range(100)]
# plt.plot(a, color="r")
# plt.plot(b, color="b")
# plt.savefig(path)
# plt.show()

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))