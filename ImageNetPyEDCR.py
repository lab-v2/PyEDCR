import json

with open('ImageNet/Labels.json', 'r') as f:
    labels = json.load(f)

coarse_grain_classes_list = [
    'Bird', 'Snake', 'Spider', 'Small Fish', 'Turtle', 'Lizard', 'Crab', 'Shark'
]

all_fine_grain_classes_list = [
    'macaw', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'flamingo',
    'white stork, Ciconia ciconia', 'bald eagle, American eagle, Haliaeetus leucocephalus',
    'magpie', 'peacock', 'black grouse', 'goldfinch, Carduelis carduelis',
    'great grey owl, great gray owl, Strix nebulosa', 'hummingbird',
    'night snake, Hypsiglena torquata', 'garter snake, grass snake',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sea snake',
    'green snake, grass snake', 'hognose snake, puff adder, sand viper',
    'king snake, kingsnake', 'thunder snake, worm snake, Carphophis amoenus',
    'vine snake', 'sidewinder, horned rattlesnake, Crotalus cerastes',
    'garden spider, Aranea diademata', 'wolf spider, hunting spider',
    'barn spider, Araneus cavaticus', 'black widow, Latrodectus mactans', 'tarantula',
    'tench, Tinca tinca', 'goldfish, Carassius auratus',
    'terrapin', 'mud turtle', 'loggerhead, loggerhead turtle, Caretta caretta',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    'agama', 'common iguana, iguana, Iguana iguana',
    'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'whiptail, whiptail lizard',
    'Dungeness crab, Cancer magister', 'hermit crab', 'rock crab, Cancer irroratus',
    'tiger shark, Galeocerdo cuvieri',
    'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    'hammerhead, hammerhead shark'
]



