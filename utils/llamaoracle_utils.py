import ollama
from PIL import Image
import io
from loguru import logger
from tqdm import tqdm
import torch

def query_llama(dl, queries):
    output = [] 
    n_images = len(dl)
    n_concepts = len(queries)
    c_tensors = []
    for image, concept, label in tqdm(dl,desc="Querying images"):
        logger.debug(f"img shape:{image.shape}, concept shape:{concept.shape}, label shape:{label.shape}")
        image = image[0]
        #plt.imshow(image)
        #plt.show()
        image = image.numpy()
        img = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        llama_img = img_byte_arr.getvalue()
        img_byte_arr.close()

        chat = []
        c_array = []
        for obj in queries:
            text = f"Does the image contain {obj}?"
            messages = [
                {"role": "user", "content": f"Does the image contain {obj}? Reply only with 'Yes' or 'No'.", "images": [llama_img]},
            ]
            response = ollama.chat(
                model = "llama3.2-vision",
                messages = messages
            )
            messages.append(response)
            llama_response = str(response["message"]['content']).lower()
            
            if llama_response.startswith("y"):
                c_array.append(1)
            else:
                c_array.append(0)
            text += "\n" + response["message"]['content']
            chat.append(text)
        c_array = torch.tensor(c_array)
        print(c_array)
        c_tensors.append(c_array)
        output.append(chat)
    
    c_tensors = torch.stack(c_tensors, dim=0)
    logger.debug(f"Final shape: {c_tensors.shape}")
    return c_tensors