import typing

import granularity


class Label(typing.Hashable):
    def __init__(self,
                 l_str: str,
                 index: int):
        self.l_str = l_str
        self.index = index
        self.g = None

    def __str__(self):
        return self.l_str

    def __hash__(self):
        return hash(f'{self.g}_{self.l_str}')

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FineGrainLabel(Label):
    def __init__(self,
                 g: granularity.Granularity,
                 l_str: str,
                 fine_grain_classes_str: typing.List[str]):
        super().__init__(l_str=l_str,
                         index=fine_grain_classes_str.index(l_str))
        assert l_str in fine_grain_classes_str
        self.g_str = 'fine'
        self.g = g


class CoarseGrainLabel(Label):
    def __init__(self,
                 g: granularity.Granularity,
                 l_str: str,
                 coarse_grain_classes_str: typing.List[str],
                 ):
        super().__init__(l_str=l_str,
                         index=coarse_grain_classes_str.index(l_str))
        assert l_str in coarse_grain_classes_str
        self.g_str = 'coarse'
        self.g = g
