from CQA.datasets import GenericDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt    
import json
import os
from tqdm import tqdm
split = 'val'
ds_name = 'cub'
data = GenericDataset(ds_name, split = split)

folder = f"./data/VLG_annotations/{ds_name}_{split}/"
save_folder = f"./data/VLG_annotations/{ds_name}_{split}_processed/"
os.makedirs(save_folder, exist_ok=True)

# 0 = female
# 1 = male

debug_labels = {}   
for i in tqdm(range(len(data))):
    img, c, l = data[i]
    new_data = []
    #plt.imshow(img)
    #plt.show()
    
    with open(os.path.join(folder, f"{i}.json")) as f:
        metadata = json.load(f)
    
    # Keep the first = {'img_path': None}    
    new_data.append(metadata[0])
    for annotation in metadata[1:]:
        #print(annotation['label'])
        #input("...")
        annotation['label'] = annotation['label'].replace("seabird bill shape hooked", "hooked seabird bill shape")
        annotation['label'] = annotation['label'].replace("head bill length about the same as", "bill length about the same as head")
        
        #print(annotation['label'])
        #print(annotation['label'])
        if l == 0:  #Which means Woman
            #print('female',annotation['label'])
            if annotation['label'] in ['5_o_clock_shadow','bald','goatee','mustache','receding_hairline','sideburns','wearing_necktie']:
                debug_labels[annotation['label']] = debug_labels.get(annotation['label'],0) + 1
                continue
        if l == 1:  #Which means Man
            #print('male',annotation['label'])
            if annotation['label'] in ['heavy_makeup','bangs','big_lips','no_beard','rosy_cheeks','wearing_earrings','wearing_lipstick','wearing_necklace']:
                debug_labels[annotation['label']] = debug_labels.get(annotation['label'],0) + 1
                continue  
        
        new_data.append(annotation)  
    # Save the updated metadata back to the file
    with open(os.path.join(save_folder, f"{i}.json"), "w") as f:
        json.dump(new_data, f, indent=4)  # Use indent=4 for better formatting
print(debug_labels)
print(len(debug_labels))

