from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

transform_basic = Compose(
                                        [
                                            ToTensor(),
                                            Resize((224,224)),
                                        ]
                                    )

transform_clip = Compose([
                Resize((224,224), interpolation=BICUBIC),
                CenterCrop((224,224)),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])