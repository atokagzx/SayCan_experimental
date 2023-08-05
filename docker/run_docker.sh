#!/bin/bash

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
docker run -ti --rm \
      --gpus all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e "DISPLAY" \
      -e "QT_X11_NO_MITSHM=1" \
      -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      -e XAUTHORITY \
      -e OPENAI_KEY=$OPENAI_KEY \
      -e ROS_MASTER_URI=$ROS_MASTER_URI \
      -e ROS_IP=$ROS_IP \
      -e SPEECH_AUTH_TOKEN=$SPEECH_AUTH_TOKEN \
      -e FOLDER_ID=$FOLDER_ID \
      -e OAUTH_TOKEN=$OAUTH_TOKEN \
      -e ROSCONSOLE_FORMAT="[\${severity}] [\${time:%H:%M:%s}] [\${node}]: \${message}" \
      -v $ROOT_DIR:/workspace \
      -v $ROOT_DIR/cache:/root/.cache \
      -v $ROOT_DIR/ros_ws/src/ds4drv/udev/50-ds4drv.rules:/etc/udev/rules.d/50-ds4drv.rules \
      --device /dev/snd \
      --device /dev/usb \
      --net=host \
      --privileged \
      --name saycan_exp saycan_exp-img \
      bash