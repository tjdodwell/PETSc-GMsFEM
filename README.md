# PETSc-GMsFEM

## Installation

To install this project you must first clone the project

```
git clone 
cd PETSc-GMsFEM
```

Next is to create a package with all the required packages

```
conda env create -f environment.yml
```

The environment is then activated

```
source activate PETSc-GMsFEM
```

The package can be installed by running setup.py

```
python setup.py install
```

## Example

A basic solve sample can be tested

```
cd examples
```

and running the command

```
mpirun -np 4 python DarcyEx1.py -AMPCG_verbose -ksp_monitor
```
