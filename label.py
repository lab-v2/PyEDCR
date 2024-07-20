import typing


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
                 l_str: str,
                 fine_grain_classes_str: typing.List[str]):
        super().__init__(l_str=l_str,
                         index=fine_grain_classes_str.index(l_str))
        assert l_str in fine_grain_classes_str

        self.g_str = 'fine'

    @classmethod
    def with_index(cls,
                   fine_grain_classes_str: typing.List[str],
                   l_index: int):
        l = fine_grain_classes_str[l_index]
        instance = cls(l_str=l,
                       fine_grain_classes_str=fine_grain_classes_str)

        return instance


class CoarseGrainLabel(Label):
    def __init__(self,
                 l_str: str,
                 coarse_grain_classes_str: typing.List[str]):
        super().__init__(l_str=l_str,
                         index=coarse_grain_classes_str.index(l_str))
        assert l_str in coarse_grain_classes_str
        self.g_str = 'coarse'

    @classmethod
    def with_index(cls,
                   i_l: int,
                   coarse_grain_classes_str: typing.List[str]):
        l = coarse_grain_classes_str[i_l]
        instance = cls(l_str=l,
                       coarse_grain_classes_str=coarse_grain_classes_str)

        return instance
