import ollama
from PIL import Image
import io
from loguru import logger
from tqdm import tqdm
from config import CLASSES, CONCEPT_SETS
import torch
import pickle
import os
import json

PREFIXES = {
    'cub':"a",
    'celeba': "a person with",
    'shapes3d': "a",
}
SUFFIXES = {
    'celeba': "feature",
}
def query_llama(dl, queries, folder, args, range=None, missing=None, examples=[]):
    try:
        prefix = PREFIXES[args.dataset] + " "
    except:
        logger.warning("PREFIX NOT FOUND!")
        prefix = ''
    try:
        suffix = " " + SUFFIXES[args.dataset]
    except:
        logger.warning("PREFIX NOT FOUND!")
        suffix = ''

    output = [] 
    split = os.path.basename(folder)
    n_images = len(dl)
    n_concepts = len(queries)
    c_tensors = []
    avg_invalid_responses = 0
    total = 0
    errors = 0

    learn_by_mistakes = []

    # Load Classes
    with open(CLASSES[args.dataset]) as f:
        class_names = f.read().split("\n")
    
    # Load Concepts
    with open(CONCEPT_SETS[args.dataset]) as f:
        concept_names = f.read().split("\n")

    if len(queries) != len(concept_names):
        print(len(queries), len(concept_names))
        raise ValueError("The queries should be same as concepts in number!")
    
    with open(os.path.join(CONCEPT_SETS['root'],f"{args.dataset}/{args.dataset}_per_class.json"), "r") as f:
        per_class_concepts = json.load(f)

    for i, (image, concept, label) in enumerate(tqdm(dl,desc=f"Querying images {split}")):
        # The dataset may start at a specific index != 0, so need to correct that
        # In most cases, this will be 0 so nothing happens.
        
        if hasattr(dl.dataset,f"{split}_subset_indexes"):
            i += getattr(dl.dataset,f"{split}_subset_indexes")[0]
        
        if isinstance(label, torch.Tensor):
            label = label.long()
            class_name = class_names[label.item()]
            class_id = label.item()
        else:
            class_name = class_names[int(label)]
            class_id = int(label)

        if args.dataset == 'celeba':
            # In celeba there are only 2 class, so the concepts to query are 39 minus the concepts 
            # relevant to the opposite class
            concepts_relevant_to_opposite = per_class_concepts[class_names[(class_id+1)%2]]
            #print(concepts_relevant_to_opposite)
            concepts_to_query = [item for item in queries if item not in concepts_relevant_to_opposite]
            #print(concept_names)
            #print(len(concepts_to_query))
        elif args.dataset == 'shapes3d':
            rel_concepts = per_class_concepts[class_names[(class_id)]]
            #print(class_id)
            #print(rel_concepts)
            if rel_concepts == []:
                concepts_to_query = queries
            else:
                concepts_to_query = [item for item in queries if item in rel_concepts]
                #print(concepts_to_query)

            #print(concepts_to_query)
            #asd
        else:
            concepts_to_query = per_class_concepts[class_name]

        
        if range is not None:
            if i < range[0]:
                continue
            if i > range[1]:
                break
        if missing is not None:
            if i not in missing:
                continue
        #logger.debug(f"img shape:{image.shape}, concept shape:{concept.shape}, label shape:{label.shape}")
        image = image[0]
        #plt.imshow(image)
        #plt.show()
        image = image.numpy()
        img = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        llama_img = img_byte_arr.getvalue()
        img_byte_arr.close()

        mask = []
        for definition in queries:
            if definition in concepts_to_query:
                mask.append(True)
            else:
                mask.append(False)

        #chat = []
        c_array = []
        ir = 0
        for c_id,obj in enumerate(queries):
            if mask[c_id]:
                text = f"Does the image contain {obj}?"
                #logger.debug(f"N concepts:{len(queries)}. Query: {text}")
                messages = [
                    {"role": "user", "content": f"This is an image of {class_name}. Does the image contain {prefix}{obj}{suffix}?", "images": [llama_img]},
                ]
                #logger.debug(messages[0]['content'])
                #for mistake in learn_by_mistakes:
                #    messages.append({"role": "user", "content": mistake['prompt'], "images": [mistake['img']]})
                messages.append({"role": "user", "content": f"Please reply only with 'Yes' or 'No'."})
                response = ollama.chat(
                    model = args.ollama_model,
                    options={"temperature":0.0},
                    #model = "x/llama3.2-vision",
                    messages = messages
                )
                messages.append(response['message'])
                llama_response = str(response["message"]['content']).lower().strip()
                
                #logger.info(llama_response)
                
                if llama_response.startswith("y"):
                    c_array.append(1)
                elif llama_response.endswith(" yes") or " yes" in llama_response:
                    c_array.append(1)
                elif llama_response.startswith("n"):
                    c_array.append(0)
                elif llama_response.endswith(" no") or " no" in llama_response:
                    c_array.append(0)
                else:
                    #logger.debug("Reasoning...")
                    messages.append({"role":"assistant", "content":llama_response})
                    messages.append({"role": "assistant", "content": f"Given this information, my 'Yes' or 'No' answer is:\n"})

                    response = ollama.chat(
                        model = args.ollama_model,
                        options={"temperature":0.0},
                        #model = "x/llama3.2-vision",
                        messages = messages
                    )
                    llama_response = str(response["message"]['content']).lower().strip()
                    #print(llama_response)
                    if llama_response.startswith("y"):
                        c_array.append(1)
                    elif llama_response.endswith(" yes") or " yes" in llama_response:
                        c_array.append(1)
                    elif llama_response.startswith("n"):
                        c_array.append(0)
                    elif llama_response.endswith(" no") or " no" in llama_response:
                        c_array.append(0)
                    else:
                        c_array.append(0)
                        ir += 1 
                    #logger.warning(f"Invalid responses: {ir}")
                
                #if int(concept[0][c_id]) == 1 or int(c_array[-1]):
                #    if int(concept[0][c_id]) != int(c_array[-1]):
                #        if 'size' not in obj and len(learn_by_mistakes) < 10:
                #            learn_by_mistakes.append({'img': llama_img,
                #                                    'prompt': f"For example: does the image contain {obj}? Answer:{c_array[-1]}",
                #                                    })
                #        errors += 1
                #        #print(f"{len(learn_by_mistakes)} {obj}")
                #    total += 1
                #    try:
                #        acc = errors/total
                #    except:
                #        acc = 1
                    #logger.debug(f"[{acc}]")
                #logger.debug(f"[{acc}] -> Does the image contain {obj}? Answer:{c_array[-1]} - Truth:{int(concept[0][c_id])}")
                #if int(concept[0][c_id]) == int(c_array[-1]) and int(c_array[-1]) == 1:
                #    errors += 1
                #total += 1

                text += "\n" + response["message"]['content']
                #chat.append(text)
                #print(c_array)
            else:
                c_array.append(0)
        avg_invalid_responses += ir
        logger.warning(f"Average invalid responses: {avg_invalid_responses/(i+1)}")
        c_array = torch.tensor(c_array)
        logger.debug(f"Result: {c_array}")
        pickle.dump(c_array, open(os.path.join(folder,f"query_{i}.pkl"),"wb"))
        #print(c_array)
        #c_tensors.append(c_array)
        #output.append(chat)
    return
    #c_tensors = torch.stack(c_tensors, dim=0)
    #logger.debug(f"Final shape: {c_tensors.shape}")
    #return c_tensors


def unify_pickles(folder, save_path, indexes):
    logger.debug(f"Unifying pickles {indexes}")
    # Save the concepts in a single .pth file
    concepts_ds = []
    files = range(indexes[0], indexes[1])
    for sample_id in tqdm(files, desc = 'Storing concepts in a single file'):
        sample_path = os.path.join(folder, f"query_{sample_id}.pkl")
        with open(sample_path, 'rb') as f:
            c_tensor = pickle.load(f)
        #print(c_tensor)
        concepts_ds.append(c_tensor)
    concepts_ds = torch.stack(concepts_ds, dim=0)
    logger.debug(f"Final shape: {concepts_ds.shape}. Concepts saved as {save_path}")

    torch.save(concepts_ds, os.path.join(save_path))
    return concepts_ds