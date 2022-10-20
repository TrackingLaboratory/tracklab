#!/usr/bin/env bash
USERNAME=user
CURR_DIR=$(pwd)

cp Dockerfile Dockerfile.bkp

echo "RUN adduser --disabled-password --gecos \"\" -u $UID $USERNAME"  >> Dockerfile
echo "USER $USERNAME" >> Dockerfile
echo "WORKDIR /home/$USERNAME" >> Dockerfile

echo "RUN mkdir .cache" >> Dockerfile
echo "RUN mkdir .cache/torch" >> Dockerfile
echo "RUN mkdir .cache/torch/hub/" >> Dockerfile
echo "RUN mkdir .cache/torch/hub/checkpoints" >> Dockerfile
echo "WORKDIR /home/$USERNAME/.cache/torch/hub/checkpoints/" >> Dockerfile
echo "RUN curl https://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth -o bn_inception-52deb4733.pth" >> Dockerfile
echo "RUN curl https://download.pytorch.org/models/resnet50-19c8e357.pth -o resnet50-19c8e357.pth" >> Dockerfile

echo "WORKDIR /home/$USERNAME/baselines" >> Dockerfile

docker build --tag='andoer/pt21-baselines' .

rm Dockerfile
mv Dockerfile.bkp Dockerfile
