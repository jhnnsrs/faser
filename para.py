from faser.napari.widgets.main_widget import FloatRange, IntRange
from pydantic import BaseModel
from typing import List
import itertools
import dask
import time
import dask.array as da
import numpy as np
  # overwrite default with multiprocessing scheduler

class OptionRange(BaseModel):
    options: List[str]

    def to_list(self):
        return self.options
    


class Model(BaseModel):
    x: float
    y: float
    z: int
    options: str
    f: int






mapper = {
    "x": FloatRange(min=0, max=1, steps=2),
    "y": FloatRange(min=0, max=1, steps=2),
    "z": IntRange(min=0, max=1, steps=2),
    "options": OptionRange(options=["a", "b"]),
}


@dask.delayed
def process_map(model_values):
    return np.zeros((10,10,10), dtype=np.float64)



def start_map():

    model = Model(x=0, y=0, z=0, options="a", f=5)

    permutations = list(itertools.product(*map(lambda x: x.to_list(), mapper.values())))

    models = []

    for i in permutations:
        values = {k: v for k, v in zip(mapper.keys(), i)}
        new_model = model.copy(update=values)
        models.append(new_model)
        print(new_model)


    t = [da.from_delayed(process_map(new_model.dict()), (10,10,10), dtype=np.float64) for new_model in models]
    t = da.stack(t)

    print(t.shape)

    print(t.compute())


    return mapper



if __name__ == "__main__":
    start_map()