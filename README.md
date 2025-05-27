Generative AI for CGCNN

A Generative AI Framework for Crystal Graph Convolutional Networks with adaptive feature integration. Automatically handles molecular and crystal datasets with any number of properties (1, 5, 12, 50, 100, 500+).


Basic Usage (Any Dataset with CGCNN):

    # Dataset with extra properties of molecules structures such as band-gap, formation energy, effective masses, Surface area, pore volume, gas uptake of the .cif molecules 

    python train_generative_cgcnn.py ./data/MyData/ --properties ./properties.csv

    # Structure-only CGCNN generation

    python train_generative_cgcnn.py ./data/StructureData/ --structure-only

    python flexible_vae_training.py ./data/MyData/ \    
    --feature-file ./data/MyData/features.csv \
    --balanced-training \
    --structure-weight 1.0 \
    --feature-weight 1.0 \
    --lr 0.0001 \
    --batch-size 32 \
    --beta-max 0.5 \
    --epochs 150 


Conda Installation: 

    conda create -n gen-cgcnn python=3.9

    conda activate gen-cgcnn

    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch 



Acknowledgments:

    Built on Crystal Graph Convolutional Neural Networks (CGCNN) by Tian Xie 
    
    Inspired by beta-VAE and conditional generation techniques
    
