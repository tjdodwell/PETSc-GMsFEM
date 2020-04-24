import numpy as np

import pytest

from PETScGMsFEM import *

from mpi4py import MPI
from petsc4py import PETSc

@pytest.fixture
def singleElementModel():
    '''Returns a single element Model'''

    model = PETSc.DMDA().create([2, 2, 2], dof=1, stencil_width=1)

    return model

def test_num_elements(singleElementModel):

    elem = singleElementModel.getElements()

    assert elem.shape[0] == 1
