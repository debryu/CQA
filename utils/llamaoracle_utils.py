import ollama
from PIL import Image
import io
from loguru import logger
from tqdm import tqdm
import torch
import pickle
import os
def query_llama(dl, queries, folder, args, range=None, missing=None):
    output = [] 
    split = os.path.basename(folder)
    n_images = len(dl)
    n_concepts = len(queries)
    c_tensors = []
    avg_invalid_responses = 0
    for i, (image, concept, label) in enumerate(tqdm(dl,desc=f"Querying images {split}")):
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

        #chat = []
        c_array = []
        ir = 0
        for obj in queries:
            text = f"Does the image contain {obj}?"
            logger.debug(f"N concepts:{len(queries)}. Query: {text}")
            messages = [
                {"role": "user", "content": f"Does the image contain {obj}?", "images": [llama_img]},
                {"role": "user", "content": f"Please reply only with 'Yes' or 'No'."},
            ]
            response = ollama.chat(
                model = args.ollama_model,
                options={"temperature":0.0},
                #model = "x/llama3.2-vision",
                messages = messages
            )
            messages.append(response['message'])
            llama_response = str(response["message"]['content']).lower().strip()
            #print(f"Does the image contain {obj}? Reply only with 'Yes' or 'No'.\n")
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
            text += "\n" + response["message"]['content']
            #chat.append(text)
            #print(c_array)
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