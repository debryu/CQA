import json

with open("./data/concepts/cub/cub_labo.json", "r") as f:
    per_class_concepts = json.load(f)

concepts = []
for key,value in per_class_concepts.items():
    print(key)
    for conc in value:
        concepts.append(conc)

print(f"Num of concepts: {len(concepts)}")
# Save resulting concepts
with open("./data/concepts/cub/cub_labo.txt", 'w') as f:
    f.write(concepts[0])
    for c in concepts[1:]:
        f.write('\n'+c)
