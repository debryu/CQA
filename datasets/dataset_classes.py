from torchvision.datasets import CelebA


class CelebACustom(CelebA):
    name = "celeba"
    def __init__(
        self,
        root,
        split: str = "train",
        target_type = "attr",
        transform = None,
        target_transform = None,
        download: bool = False,
        concepts: list = None,
        label:int = 20,
    ) -> None:
      '''
        concepts: list of concepts to use, by choosing the indexes of the celeba attributes
        label: the index of the attribute to use as label
      '''
      assert type(label) == int # label must be an integer
      if split == 'val':
        split = 'valid'
      super().__init__(root=root,split=split,target_type=target_type,transform=transform,target_transform=target_transform,download=download)
      if concepts is None:
        self.concepts = list(range(40))
      else:
        self.concepts = concepts
      self.label = label
      if self.label in self.concepts:
        self.concepts.remove(self.label)
        #print(f"Removed label {self.label} from concepts")

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        concepts = y[self.concepts]
        y = y[self.label]
        return x, concepts, y