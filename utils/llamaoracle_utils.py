import ollama
from PIL import Image
import io
from loguru import logger

def query_llama(dl, queries):
    output = [] 
    for image, concept, label in dl:
        logger.debug(f"img shape:{image.shape}, concept shape:{concept.shape}, label shape:{label.shape}")
        #plt.imshow(image)
        #plt.show()
        image = image.numpy()
        img = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        llama_img = img_byte_arr.getvalue()
        img_byte_arr.close()

        response = []
        for obj in queries:
            text = f"Does the image contain {obj}?"
            messages = [
                {"role": "user", "content": f"Does the image contain {obj}?", "images": [llama_img]},
            ]
            response = ollama.chat(
                model = "x/llama3.2-vision",
                messages = messages
            )
            messages.append(response)
            text += "\n" + response["message"]['content']
            response.append(response)
        output.append(response)
    return output