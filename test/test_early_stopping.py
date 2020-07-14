import numpy as np
import matplotlib.pyplot as plt
import json
from pysmg.schedulers.criteria import EarlyStopping, SupportedDenoiseFn


np.random.seed(10)

def train_curve_generator():
    with open("/Users/zhfeing/Desktop/flat_cifar10_resnet18_thin.json") as f:
        dict_his = json.load(f)
    x = np.array(dict_his['jsons']['resnet18_thin epoch loss']["content"]["data"][0]["x"])
    ys = np.array(dict_his['jsons']['resnet18_thin epoch loss']["content"]["data"][0]["y"])
    ys += np.random.rand(*ys.shape) / 10
    plt.plot(np.array(x), np.array(ys))
    for y in ys:
        yield y


es = EarlyStopping(
    patience=5,
    min_delta=1e-4,
    mode="min",
    # baseline=,
    denoise_fn=SupportedDenoiseFn["average_5"]
)
cc = ["r", "g", "b", "c", "y", "k", "w"]
for i, loss in enumerate(train_curve_generator()):
    es.update(loss)
    plt.scatter(i, loss, color=cc[es.reset_times])
    if es.to_stop:
        es.reset()
        print("reset +1")
        if es.reset_times >= 6:
            break
plt.show()



