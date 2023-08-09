# Multi Vehicle tracking in carla simulator 

This repo host a comprehensive solution for multi-object tracking in combination with YOLO and DeepSORT in carla simulator .

Following software are required before installing required packages

1. Carla simulator : Download Carla simulator and follow the instuction guide - https://carla.readthedocs.io/en/latest/start_quickstart/
2. CUDA : Download and install CUDA - https://developer.nvidia.com/cuda-downloads
3. CuDnn : Install Cuda DNN - https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
4. Anacoda : https://www.anaconda.com/download
5. PyMOT : Download / clone this repo for evaluations - https://github.com/Videmo/pymot
6. Create and activate a new virtural conda env 

    ```
    conda create --name <env-name> python=3.8
    conda activate <env-name> 
        NOTE : <env-name> should be the name of your virtural env
    ```
7. Install required packages
    ```
    pip install  requirements.txt
    ```
8. Once the setup is done Run Calra.exe file to launch simulator
9. One commapnd prompt or launch jupyter from conda prompt
10. If using CLI or run track.ipynb file in jupyter 
    ```
    python track.ipynb
    ```
       
