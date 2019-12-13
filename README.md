# dvc-in-use-example
Inference of Resnet-50 model using [dvc](https://github.com/iterative/dvc).

## Initialize

```
dvc pull
```

```
python train.py data/
```

In the beggining of `main()` you can see different model importing options: .
1. From torchvision moodel zoo
2. From `pretrainedmodels` package
3. From DVC (under development)

