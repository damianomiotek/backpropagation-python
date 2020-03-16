
# backpropagation-python
Implementation of the backpropagation algorithm in Python language.

**Usage on Linux**

1. Clone or download repository.
2. It is recommended to install backpropagation package in Python virtual environment. Create and activate python3 virtual environment. You can do this for example through type in terminal:
```shell script
cd <path_to_backpropagation-python_package>/
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
```
3. Fill backpropagation/config/settings.py file.
4. Install backpropagation package. To achieve this, enter in terminal:
```shell script
cd <path_to_backpropagation-python_package>/
pip install .
```
or alternatively:
```shell script
cd <path_to_backpropagation-python_package>/
make install
```
5. Run backpropagation algorithm:
```shell script
cd <path_to_backpropagation-python_package>/backpropagation
python3 algorithm.py
```
or:
```shell script
cd <path_to_backpropagation-python_package>/
make run
```

6. Because code is distributed in package, there is possibility to import needed stuffs in own programs.

**Further usage**

1. Every time you change any of the files from the package, you should update backpropagation package in created virtualenv:
```shell script
make update
```
or
```shell script
pip install .
```
2. You can update and run algorithm.py in one step:
```shell script
make update-run
```

**Other informations**

I wrote this code in order to learn backpropagation algorithm. I was learning from `Artificial Intelligence: A Modern Approach` by Stuart Russell and Peter Norvig . Code was run only on Linux platform.
