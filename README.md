# Models for Vuesic
[![Build Status](https://travis-ci.org/dawg/models.svg?branch=master)](https://travis-ci.org/dawg/models)

## Setting up the project

Install the dependencies using the following command:
```
pipenv install
```

Run the following setup script in the project root dirctory to fetch the packages:
```
python setup.py install
```

## Environment
Setup your environment using the following script:
```
./environment.sh
```

Install the requirements using the following script:
```
./requirements.sh
```
## Jupyter Notebook instructions

1. Install all dependencies and enter the pipenv shell
   ```
   pipenv install
   pipenv shell
   ```

2. Create a kernel for your virtual environment
   ```
   python -m ipykernel install --user --name=my-virtualenv-name
   ```

3. Launch the jupyter notebook from your pipenv shell
   ```
   jupyter notebook
   ```

4. Change the notebook kernel to your virtual environment's kernel
   ![](https://i.stack.imgur.com/htimC.png)


### Formatting
Follow [these instructions](https://github.com/ambv/black#pycharm) to set up black in PyCharm.

