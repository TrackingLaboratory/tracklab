#!/usr/bin/env bash
USERNAME=user
cp Dockerfile Dockerfile.bkp

echo "RUN adduser --disabled-password --gecos \"\" -u $UID $USERNAME"  >> Dockerfile
echo "USER $USERNAME" >> Dockerfile
echo "WORKDIR /home/$USERNAME/MOT_Evaluation" >> Dockerfile

docker build --tag='mot_evaluation' .

rm Dockerfile
mv Dockerfile.bkp Dockerfile
