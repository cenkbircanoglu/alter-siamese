from collections import OrderedDict


class LimitedSizeDictionary(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop('size_limit', None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)
