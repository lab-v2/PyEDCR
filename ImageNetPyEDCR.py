import json

with open('ImageNet/Labels.json', 'r') as f:
    labels = json.load(f)

coarse_grain_classes_list = [
    'Bird', 'Snake', 'Spider', 'Small Fish', 'Turtle', 'Lizard', 'Crab', 'Shark'
]

fine_grain_classes_dict = {
    'n01818515': 'macaw',
    'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
    'n02007558': 'flamingo',
    'n02002556': 'white stork, Ciconia ciconia',
    'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus',
    'n01582220': 'magpie',
    'n01806143': 'peacock',
    'n01795545': 'black grouse',
    'n01531178': 'goldfinch, Carduelis carduelis',
    'n01622779': 'great grey owl, great gray owl, Strix nebulosa',
    'n01833805': 'hummingbird',
    'n01740131': 'night snake, Hypsiglena torquata',
    'n01735189': 'garter snake, grass snake',
    'n01755581': 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
    'n01751748': 'sea snake',
    'n01729977': 'green snake, grass snake',
    'n01729322': 'hognose snake, puff adder, sand viper',
    'n01734418': 'king snake, kingsnake',
    'n01728572': 'thunder snake, worm snake, Carphophis amoenus',
    'n01739381': 'vine snake',
    'n01756291': 'sidewinder, horned rattlesnake, Crotalus cerastes',
    'n01773797': 'garden spider, Aranea diademata',
    'n01775062': 'wolf spider, hunting spider',
    'n01773549': 'barn spider, Araneus cavaticus',
    'n01774384': 'black widow, Latrodectus mactans',
    'n01774750': 'tarantula',
    'n01440764': 'tench, Tinca tinca',
    'n01443537': 'goldfish, Carassius auratus',
    'n01667778': 'terrapin',
    'n01667114': 'mud turtle',
    'n01664065': 'loggerhead, loggerhead turtle, Caretta caretta',
    'n01665541': 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    'n01687978': 'agama',
    'n01677366': 'common iguana, iguana, Iguana iguana',
    'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'n01685808': 'whiptail, whiptail lizard',
    'n01978287': 'Dungeness crab, Cancer magister',
    'n01986214': 'hermit crab',
    'n01978455': 'rock crab, Cancer irroratus',
    'n01491361': 'tiger shark, Galeocerdo cuvieri',
    'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    'n01494475': 'hammerhead, hammerhead shark'
}


def assert_sub_dict(d1: dict, d2: dict):
    """
    Asserts that every key in d1 is also in d2 and that for all such keys,
    the values are equal (d1[k] == d2[k]).

    :param d1: The first dictionary to check (sub-dict).
    :param d2: The second dictionary in which we check for the presence of d1's items.
    """
    for key in d1:
        # Check if the key is present in d2 and the values are the same
        assert key in d2 and d1[key] == d2[
            key], (f"Assertion failed: Key '{key}' with value '{d1[key]}'"
                   f" in d1 is not present with the same value in d2.")

assert_sub_dict(fine_grain_classes_dict, labels)

