'''
Create cub_preprocess.txt list of concepts given the original one in cub.txt

'''
from loguru import logger
import os
from config import CONCEPT_SETS

save_path = CONCEPT_SETS['cub']
cub_original_concepts_path = "./data/concepts/cub/cub.txt"

indexes = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

concepts = {}
with open(cub_original_concepts_path) as f:
    lines = f.read().split("\n")
    for line in lines:
        try:
            index, desc = line.split(" ")
            concepts[index] = desc
        except:
            logger.warning(f"Line not processed: '{line}'")
            pass

indexes = [i+1 for i in indexes]
preprocessed_concepts = []
print(concepts)
for idx in indexes:
    desc = concepts[str(idx)].replace("::",": ")
    desc = desc.replace("_", " ").replace(":","")
    desc = desc.split(" ")[1:]
    desc.insert(0,desc.pop())
    desc = (" ").join(desc)
    preprocessed_concepts.append(desc)

# Save resulting concepts
with open(save_path, 'w') as f:
    f.write(preprocessed_concepts[0])
    for c in preprocessed_concepts[1:]:
        f.write('\n'+c)
