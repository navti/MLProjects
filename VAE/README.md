
# Variational Autoencoder example with CIFAR-10 dataset

Sample deep learning project to train a variational autoencoder on the CIFAR-10 dataset. Different learning rates, batch sizes can be given as input when running the training. Optionally, the model can be saved. Check the main.py file for more options. The weigting hyperparameter beta for the KL divergence loss can be given as a training parameter. It is set to 1 as default.





## To run the training

```python
python main.py --lr 0.5 --batch-size 64 --nf 32 --save-model --beta 0.1
```

