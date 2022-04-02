# Environment setup

1. If we want to create a new python environment, it is recommended to use conda:
```
conda create --name bert_for_longer_texts python=3.8
```

2. Activate the environment
```
conda activate bert_for_longer_texts
```

3. Install pytorch and cudatoolkit. This depends on the machine - first check the version of GPU drivers by the command `nvidia-smi` and choose the newest version compatible with this drivers according to [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (e.g. 11.1). Then we install torch via conda to get the compatible build. [Here] we find which torch version is compatible with CUDA version on our machine.

Example command for older version:

```
conda install cudatoolkit=10.1 "pytorch::pytorch==1.7.1" "torchvision==0.8.2" -c pytorch -c conda-forge
```

Example for cuda 11:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

4. Install other packages by running a command from repo root:

```
bash env_setup.sh
```

5. Every new dependency should be added in `./env_setup.sh`.