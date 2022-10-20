#!/usr/bin/env bash 
USERNAME=user

cp Dockerfile Dockerfile.bkp 
echo "RUN adduser --disabled-password --gecos \"\" -u $UID $USERNAME"  >> Dockerfile
echo "USER $USERNAME" >> Dockerfile

###### COMMENT,  IF YOU DONT NEED AN IDE ######
PATH="/opt/miniconda3/bin:$PATH"
echo "RUN echo a" >> Dockerfile
echo "RUN git clone https://github.com/anDoer/vim-dev.git /home/$USERNAME/vim-dev" >> Dockerfile
echo "WORKDIR /home/$USERNAME/vim-dev" >> Dockerfile
echo "USER root" >> Dockerfile
echo "RUN bash internal_docker_install/install.sh $USERNAME $PATH /home/$USERNAME/vim-dev/" >> Dockerfile
echo "USER $USERNAME" >> Dockerfile
##################################################
echo "WORKDIR /home/$USERNAME/eval" >> Dockerfile 

docker build --tag='andoer/posetrack21_eval_kit' .
rm Dockerfile 
mv Dockerfile.bkp Dockerfile 
