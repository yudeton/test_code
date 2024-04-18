#要訪問CUDA開發工具，您應該使用devel映像。這些是相關的標籤：
# 1.nvidia/cuda:10.2-devel-ubuntu18.04

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
#指定docker image存放位置
VOLUME ["/storage"]
MAINTAINER yudeton@gmail.com

# root mode
USER root
# environment
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


ARG SSH_PRIVATE_KEY

COPY ./cmake-3.20.0 /Documents/cmake-3.20.0

RUN mkdir -p /code
WORKDIR /code

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN sed -i 's/http:\/\/archive\.ubuntu\.com/http:\/\/mirrors\.aliyun\.com/g' /etc/apt/sources.list

# RUN apt-get dist-upgrade
RUN apt-get update &&  apt-get install -y --no-install-recommends make g++ && \
#
# Dockerfile for OpenCV with CUDA C++, Python 2.7 / 3.6 development 
# Pulling CUDA-CUDNN image from nvidia
# Basic toolchain 
    apt-get update && \
        apt-get install -y \
        build-essential \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libcurl4-openssl-dev \
        zlib1g-dev \
        nano \
	curl \
        gedit \
        vim && \
    apt-get autoremove -y && \
    #set debconf-utils
    apt-get install software-properties-common -y &&\
    apt-get update &&\
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" &&\
    apt-get  update &&\
    apt install libjasper1 libjasper-dev -y && \
    apt-get update &&  apt-get install --assume-yes apt-utils  && \
    apt-get -y install debconf-utils && \
    #setting system clock
    apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Stockholm /etc/localtime && \
    echo “Asia/Taipei” > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    export DEBIAN_FRONTEND=noninteractive && \
# solve Error debconf
    apt-get install dialog apt-utils -y && \
    echo '* libraries/restart-without-asking boolean true' |  debconf-set-selections && \
# Fix not find lsb-release
    apt-get update && apt-get install -y lsb-release && apt-get clean all && \
# Fix add-apt-repository: command not found error
    apt-get install -y software-properties-common && \
# Install ROS melodic
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \   
    apt update && \
    apt install -y ros-melodic-desktop-full && \
    echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc" && \
    apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential && \
    rosdep init && \
    rosdep update

SHELL ["/bin/bash","-c"]


# install dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    wget \
    python-rosinstall \
    python-catkin-tools \
    ros-melodic-jsk-tools \
    freeglut3-dev \
    libgtk-3-dev \
    libglfw3-dev && \
    # clear cache
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y python3-pip python-pip python3-dev build-essential && \
    apt --fix-broken install && \
    #apt-get install -y python3-catkin-pkg && \
    pip2 install --upgrade setuptools && \
    apt-get install -y python-sympy && \
    pip3 install catkin_pkg && \
    sudo apt-get update && sudo apt-get -y upgrade && \
    pip3 install --upgrade pip && \
    apt-get install -y build-essential && \
    apt-get install -y qttools5-dev-tools && \
    apt-get install -y qtcreator && \
    apt-get install -y qt5-default && \
    apt-get install -y qt5-doc && \
    apt-get install -y qt5-doc-html qtbase5-doc-html && \
    apt-get install -y qtbase5-examples && \
    apt-get install -y libxcb-xinerama0 && \
    apt-get install -y ros-melodic-moveit && \
    apt-get install -y ros-melodic-moveit-visual-tools && \
    apt-get install python3-tk


# install Cmake
# RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
#     apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
#     apt-get update && \
#     apt-get install cmake
WORKDIR /Documents/cmake-3.20.0
RUN cmake . && \
    make -j8 && \
    sudo make install && \
    sudo update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force 

RUN pip3 install rospkg==1.3.0  && \
    pip3 install pyyaml>=5.4  && \
    pip3 install openpyxl==3.0.9  && \
    pip3 install numpy>=1.21 && \
    pip3 install matplotlib==2.1.1  && \
    pip3 install argparse==1.4.0 && \
    pip3 install PySide2==5.15.2 && \
    pip3 install PySide2extn==1.0.0 && \
    pip3 install swift-sim==0.10.0 && \
    pip3 install spatialgeometry==0.2.0 && \
    pip3 install spatialmath-python==0.11  && \
    pip3 install roboticstoolbox-python==0.11.0 && \
    pip3 install sympy==1.9 && \
    pip3 install numpy-stl==2.16.3 && \
    pip3 install alphashape && \
    pip3 install plotly && \
    pip3 install torch && \
    pip3 install torchvision && \
    pip3 install torchaudio && \
    pip3 install pandas && \
    pip3 install tensorflow==2.4.1 && \
    pip3 install tf-agents==0.9.0 && \
    pip3 install tensorflow_probability==0.12.2 && \
    pip3 install tensorboardX && \
    pip3 install imageio && \
    pip3 install imageio-ffmpeg && \
    pip3 install gym==0.20.0 && \
    pip3 install seaborn && \
    pip3 install scikit-learn && \
    pip3 install pybullet --upgrade --user && \
    pip3 install pybullet_planning && \
    pip3 install pytest




# for windows display desktop

# install VS code
# RUN apt install software-properties-common apt-transport-https wget -y && \
#    wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add - && \
#    add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" && \
#    apt install -y code

# RUN apt install -y xfce4
# 使用者新增
RUN useradd -ms/bin/bash iclab && echo "iclab:iclab" | chpasswd && \
adduser iclab sudo
 
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash'
# ADD bashrc /.bashrc


USER iclab
WORKDIR /home/iclab
USER root
RUN echo 'PS1="\[\e[0;33m\][\u@\h \W]\$\[\e[m\] "' >> /root/.bashrc
RUN mkdir -p /drl_robotics_arm_ws/src
WORKDIR /drl_robotics_arm_ws/src
# RUN catkin_make
# RUN . devel/setup.bash
USER iclab
