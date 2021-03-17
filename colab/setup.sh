#!/bin/sh

#apt-get purge nvidia*
#apt-get autoremove

while getopts cit flag
do
    case "${flag}" in
        i) install=true;;
        t) transformers=true;;
        c) colab=true;;

    esac
done
echo "Colab: $colab";
echo "Install: $install";

if [ "$install" == true ]; then
    echo "Installing CUDA"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
    dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
    rm /etc/apt/sources.list.d/nvidia*  /etc/apt/sources.list.d/cuda*
    apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
    apt-get update
    sudo apt-get -y install cuda

    python3 -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    python3 -m pip install wandb pytorch_lightning deepspeed mpi4py
fi

if [ "$transformers" == true ]; then
    echo "Installing Transformers"
    git clone https://github.com/huggingface/transformers
    cd transformers
    git checkout 7e662e6a3be0ece4 
    python3 -m pip install .
    cd ..
fi

if [ "$colab" == true ]; then
    gdrive_base="/content/drive/MyDrive/data/wine/"
    wine_file="name_desc_nlp_ready_test.txt"
    echo "Creating symlink from $gdrive_base$wine_file"
    ln -s  "$gdrive_base$wine_file" "/root/"
fi
