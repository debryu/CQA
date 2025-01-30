from datasets import get_dataset

cub = get_dataset('cub')
cel = get_dataset('celeba')
print(cub[0])
print(cel[0])