# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.1.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ```
    conda create --name DRN python=3.6
    ```
    And activate the environment.
    
    ```
    conda activate DRN
    ```
    

1. Install pytorch 1.1.0 and torchvision 0.2.1:

    ```
        pip install torch==1.10
        pip install torchvision==0.2.1
    ```
2. Clone this repo:

    ~~~
    DRN_ROOT=/path/to/clone/DRN
    git clone https://github.com/Panxjia/DRN2020.git $DRN_ROOT
    ~~~

3. Install [COCOAPI](https://github.com/cocodataset/cocoapi) and the variant version for rotation:

    ```
    download cocoapi and move it to $DRN_ROOT/external
    cd $DRN_ROOT/external/cocoapi/PythonAPI
    make
    python setup.py install --user
    
    cd $DRN_ROOT/external/cocoapi_ro/PythonAPI
    make
    python setup.py install --user
    ```
    
4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Compile deformable convolutional v2 (from [DCNv2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0)).

    ```
    cd $DRN_ROOT/src/lib/external
    # git clone -b pytorch_1.0.0 https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch.git
    # mv Deformable-Convolution-V2-PyTorch DCNv2
    cd DCNv2
    ./make.sh
    ```
    
6.  Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $DRN_ROOT/src/lib/external
    make
    ~~~
    
7. [Optional] Compile carafe if your want to use dense predict for DRHs.

    ```
    cd $DRN_ROOT/src/lib/external/carafe
    python setup develop
    ```

8. Download pertained models for [detection]() and move them to `$DRN_ROOT/models/`.
