# Warp Basics
Instructions to install Warp


1) Install miniconda
This is a minimal python installation which is very flexible to install, manipulate and remove.
You install miniconda by running the following commands

```
cd
curl https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
sh Miniconda3-py37_4.10.3-Linux-x86_64.sh
source Miniconda3/bin/activate
```

This install Python 3.7.4. The latest is 3.8, but there were some issues with Python 3.8 and Warp so let’s stay with 3.7.4

2) Install Python packages needed for Warp (and more)

```
pip install numpy scipy Forthon picmistandard matplotlib
```

Off the top of my head these are the fundamental ones but there could be more.. 

3) Install Warp

```
git clone https://bitbucket.org/berkeleylab/warp.git
cd Warp/pywarp90
make install3
```

4) Test your installation

Open an interactive python session and test the installation by running the following commands:

```
ipython       #this opens the python session
from warp import *           #this imports warp in the Python session
```
