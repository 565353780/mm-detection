cd ..
git clone https://github.com/open-mmlab/mmdetection.git

pip install -U torch torchvision torchaudio

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmdetection
pip install -v -e .
