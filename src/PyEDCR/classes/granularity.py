import typing


class Granularity(typing.Hashable):
    """
    Represents a granular entity that can be hashed and compared for equality.

    This class provides a mechanism to store a unique granular string and
    handle it as a hashable object. Instances of this class can be used in
    hash-based collections like dictionaries or sets. The equality of two
    instances depends on the hash value of their internal string.

    :ivar g_str: The internal string representing the granularity.
    """
    def __init__(self,
                 g_str: str):
        self.g_str = g_str

    def __str__(self):
        return self.g_str

    def __hash__(self):
        return hash(self.g_str)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
