import numpy as np
from chainer.dataset import iterator


class TripletIterator(iterator.Iterator):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self._N = len(self.dataset)

        self.reset()

    def __next__(self):
        self._previous_epoch_detail = self.epoch_detail
        anc, pos, neg = [], [], []
        for _ in range(self.batch_size):
            list(map(list.__iadd__, (anc, pos, neg), self.dataset.get_triplet()))
        batch = anc + pos + neg

        self.current_position += self.batch_size
        if self.current_position > self._N:
            self.current_position = self._N - self.current_position
            self.epoch += 1
            self.is_new_epoch = True

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._N

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position', self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
