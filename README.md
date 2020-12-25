# Text Classifier
This tutorial shows how to classify texts of IMDB.

## Official Docker images for TensorFlow

Docker pull command:

```
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter
```

Running containers:

```
docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter
```

## Download IMDB dataset

https://...

## Using BERT

Install tensorflow-hub:

```
pip install tensorflow-hub==0.9.0
```

Install tensorflow-text (A dependency of the preprocessing for BERT inputs):

```
pip install tensorflow-text==2.3.0
```

Install tf-models-official:

```
pip install tf-models-official==2.3.0
```
