<div align="center">
  <h1 align="center">If Concept Bottlenecks are the Question, are Foundation Models the Answer?</h1>
</div>
  
  <div align="center">
	  Researchers have recently proposed novel architectures that replace manual annotations with weak supervision from foundation models. It is however unclear what is the impact on the quality of the learned concepts. To answer this question, we put various models to the test, analyzing their learned concepts empirically using a selection of significant metrics.
	  <img src="CQA/prev.jpg" alt="preview" width="700" height="300">
  </div>

 <div align="center">
 <br>
 <br> 
  <a href="https://arxiv.org/abs/2504.19774v2"><img src="https://img.shields.io/badge/%F0%9F%93%84-Paper-blue?style=flat" alt="Paper Badge"></a>&nbsp;&nbsp;<a href="mailto:emanuele.marconato@unitn.it"><img src="https://img.shields.io/badge/%F0%9F%93%AA-Get in touch-green?style=flat" alt="Mail to"></a></div>

# Getting Started
## Installation
1. Clone the repository:
```
git clone https://github.com/debryu/CQA.git
```
2. Move inside the folder:
```
cd CQA
```
## Environment Setup
1. Create a new Conda environment:
```
conda create -n vlmcbm python=3.12
```
```
conda activate vlmcbm
```
2. Install the CQA library
```
pip install -e .
```

4. Customize your ```config.py``` accordingly to your preferences. Set the correct paths if you are not using the default ones.
## Usage
First move into the CQA folder  
```
cd CQA
```
I.E. when running training/evaluations you need to be in this path ```./CQA/CQA```.

#### Train
To train a simple Concept Bottleneck Model use:
```
python train.py -model resnetcbm -dataset celeba -epochs 20 -unfreeze 5
```
To train using the CUB pre-trained backbone:
```
python train.py -model resnetcbm -backbone resnet18_cub -dataset celeba -epochs 20 -unfreeze 5
```

Commands to train different models are available in ```experiments```

#### Test
To evaluate a model run:
```
python main.py -load_dir <YOUR MODEL FOLDER> -all
```
# Sources

- Label-free CBM: <https://github.com/Trustworthy-ML-Lab/Label-free-CBM>
- LaBo CBM: <https://github.com/YueYANG1996/LaBo>
- VLG-CBM: <https://github.com/Trustworthy-ML-Lab/VLG-CBM>
- CUB dataset: <https://www.vision.caltech.edu/datasets/cub_200_2011/>
- SHAPES3D dataset: <https://github.com/google-deepmind/3d-shapes>
- CELEBA dataset: <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
- DCI metrics: <https://github.com/andreinicolicioiu/DCI-ES>
- OIS metric: <https://github.com/mateoespinosa/concept-quality>
- Sparse final layer training: <https://github.com/MadryLab/glm_saga>
- CLIP: <https://github.com/openai/CLIP>
- Grounding DINO: <https://github.com/IDEA-Research/GroundingDINO>
- LLava Phi 3 (using Ollama): <https://ollama.com/library/llava-phi3>

# Cite this work
```
@misc{debole2025conceptbottlenecksquestionfoundation,
      title={If Concept Bottlenecks are the Question, are Foundation Models the Answer?}, 
      author={Nicola Debole and Pietro Barbiero and Francesco Giannini and Andrea Passerini and Stefano Teso and Emanuele Marconato},
      year={2025},
      eprint={2504.19774},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.19774}, 
}
```
