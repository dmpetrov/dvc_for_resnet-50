# dvc-in-use-example
Inference of Resnet-50 model using [dvc](https://github.com/iterative/dvc).

## Initialize

```
dvc pull
```

## inference.py:
Command line to run inference.py:
 ```
 python inference.py
 ```
To choose an image add ```--img_path path/to/image/```, 'data/croco.jpg' is set by default.
## tests.py:
Command line to run tests:
```
 pytest -s tests/tests.py
```
