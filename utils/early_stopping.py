import torch as th
from utils.save import generate_path


class EarlyStopping(object):
    def __init__(self, patience, min_delta=0., cumulative_delta=False, file_name=None):
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None
        self.is_stop = False
        self.path = generate_path("data/model_state", file_name, ".pt")

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                self.is_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        th.save(model.state_dict(), self.path)

    def load_checkpoint(self):
        return th.load(self.path)