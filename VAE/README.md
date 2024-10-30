
# Variational Autoencoder example with CIFAR-10 dataset

Sample deep learning project to train a variational autoencoder on the CIFAR-10 dataset. Different learning rates, batch sizes can be given as input when running the training. Optionally, the model can be saved. Check the main.py file for more options. The weigting hyperparameter baseline beta for the KL divergence loss can be given as a training parameter. It is set to 0.0 as default (annealing is enabled by default). If annealing is disabled, the desired value of baseline beta parameter should be passed.


The plots for losses are saved in 'VAE/results' folder.
The model is saved in 'VAE/results/saved_models' folder.
The samples generated using the trained model are saved in 'VAE/results/generated' folder.

## For help

```python
python main.py --help
```

## To run the training without annealing

```python
python main.py --lr 0.5 --batch-size 64 --nf 32 --baseline-beta 1.0 --annealing-disable --save-model 
```

## To run training with different annealing steps and shape

```python
python main.py --lr 0.1 --batch-size 64 --nf 32 --annealing-steps 20 --annealing-shape 'linear' --save-model
```