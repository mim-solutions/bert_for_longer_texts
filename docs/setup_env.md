# Environment setup

1. Project requires Python 3.9+.

2. If we want to create a new python environment, it is recommended to use conda:
```
conda create --name bert_long python=3.9
```

3. Activate the environment
```
conda activate bert_long
```

4. Install `pytorch` and `cudatoolkit`. The version of the driver depends on the machine - first, check the version of GPU drivers by the command `nvidia-smi` and choose the newest version compatible with these drivers according to [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (e.g.: 11.1). Then we install `torch` via `conda` to get the compatible build. [Here](https://pytorch.org/get-started/previous-versions/), we find which torch version is compatible with the CUDA version on our machine.

5. Install other packages by running a command from the repo root:
```
pip install -r requirements.txt
```

6. File `requirements.txt` can be updated using the command:
```
bash pip-freeze-without-torch.sh > requirements.txt
```
This script saves all dependencies of the current active environment except `torch`.