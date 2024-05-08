# ViTImageRecommender
development repo for ViT DINO image recommender

This repo contains code exploring and image2image recommender. In this repo we'll use [FAISS](https://github.com/facebookresearch/faiss) a vector store from facebook used for image embeddings and fast similarity search. 

## Suggested System:
1. using faiss vectorstore, return 100 images
2. using faiss identified images, get prompts from metadata
    - rank 100 prompts for similarity with local vocab
3. using faiss identified images, get tags from metadata
    - rank 100 tag sets for similarity with local vocab
4. Return 20 images to user
    - return top 15 MOST similar images
    - return 5 randomly selected images from index 15-50

## Directory 
### EDA
In this directory we have experiments for building FAISS, loading data from civitai, using MLP to increase the embedding latent space, and an interactive notebook for testing FAISS. Notebooks are named to be informative and have comments and notes inside

1. In the [getDataForEda](https://github.com/civitai/ViTImageRecommender/blob/main/EDA/getDataForEDA.ipynb) notebooks we download images, create metadata jsob, and create the faiss index
2. In the [buildFAISSWithMLP](https://github.com/civitai/ViTImageRecommender/blob/main/EDA/buildFAISSWithMLP.ipynb) notebook we assume images are stored locally, but creates an Multilayer Perceptron to allow for control over embedding size. This was an experiment to determine if increasing/adjusting embedding size increases similarity. Users should note this notebook creates MLPs with non-linearities introduced through the ReLU layers <b> experiments are incomplete - to improve results we should control for random seed and then save MLP for inference</b> 
3. In the [buildFAISSWithMLPNoRelu](https://github.com/civitai/ViTImageRecommender/blob/main/EDA/buildFAISSWithMLPNoRelu.ipynb) notebook we assume images are stored locally, but creates an Multilayer Perceptron to allow for control over embedding size. This was an experiment to determine if increasing/adjusting embedding size increases similarity. Users should note this notebook creates MLPs without non-linearities. Here we remove the ReLU layers to prevent any non-linear compression of the embeddings. <b> experiments are incomplete - to improve results we should control for random seed and then save MLP for inference</b> 
4. In the [loadFAISSandMetadata](https://github.com/civitai/ViTImageRecommender/blob/main/EDA/loadFAISSandMetadata.ipynb) notebook we experiment with loading FAISS, the times it takes to get similar embeddings, and how different FAISS experiments have worked. We also record our take aways
5. In the [UseMetaDataForFilters](https://github.com/civitai/ViTImageRecommender/blob/main/EDA/UseMetaDataForFilters.ipynb) notebook we explore the initial system design and metadata. Here we use an example and show how prompt, tag, and DINO (image) similarities all return slightly different results and then we average them together to rank the images and return a diverse but similar matrix of images. 

### utils
central package repo for functions

### data
This directory is ignored by git, but contains the metadata json used for prompt/tag recommendations. It also contains the imgs for the demos as well as the faiss indexs and MLP weights used in experimenting with latent embedding size. 

### Interactive
1. [image_only_demo.py](https://github.com/civitai/ViTImageRecommender/blob/interactive/interactive/image_only_demo.py) is a script that launches an interactive screen for demo purposes. If there is a faiss index and metadata json appropriately named in '../data/' as well as imgs corresponding to the faiss index in '../data/imgs' then the demo allows users to drag and drop images and will return similar images found in the faiss index. users can then scroll 'endlessly' through the similar images
    - users should note aside from the data challenges, this does not store past indices which means that they may get stuck in a loop of seeing the same images over and over. To avoid this, we should track indices already seen and make sure they can't be added back to the `image_paths` variable...