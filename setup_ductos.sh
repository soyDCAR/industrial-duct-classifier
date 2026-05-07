conda create -n ductos_env python=3.10 -y
conda activate ductos_env

conda install pytorch torchvision torchaudio matplotlib scikit-learn scipy pandas -c pytorch -y
conda install ipykernel -y

pip install "numpy<2.0" --upgrade
pip install opencv-python pillow tqdm
pip install transformers pycocotools gradio
pip install groundingdino
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

python -m ipykernel install --user --name=ductos_env --display-name "Ductos (PyTorch)"
