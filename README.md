# Warp Basics
Instructions to install Warp. 


1) Install miniconda
This is a minimal python installation which is very flexible to install, manipulate and remove.
You install miniconda by running the following commands


```
cd
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh
source miniconda3/bin/activate
```
or 

```
cd
curl https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
sh Miniconda3-py37_4.10.3-Linux-x86_64.sh
source miniconda3/bin/activate
```
This installs Python 3.7.4. The latest is 3.8, but there were some issues with Python 3.8 and Warp so let’s stay with 3.7.4

2) Install Python packages needed for Warp (and more)

```
pip install numpy scipy Forthon picmistandard matplotlib ipython
```

Off the top of my head these are the fundamental ones but there could be more.. 

3) Install Warp. 

```
git clone https://bitbucket.org/berkeleylab/warp.git
cd warp/pywarp90
make install3
```
Note that ```gfortran``` (Fortran compiler) is needed. You can install it with ```sudo apt install gfortran```

4) Test your installation

Open an interactive python session and test the installation by running the following commands:

```
ipython       #this opens the python session
from warp import *           #this imports warp in the Python session
```

If new to Linux, you will need to install some extra command tools (curl, make, git) before following these intructions. You can install them using

```
sudo apt install curl
sudo apt install make
sudo apt install git
```
Another useful way is to install the essential tools via ```sudo apt install build-essential``` 
