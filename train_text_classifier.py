import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# To create AdamW optmizer
from official.nlp import optimization

import os
import shutil
import matplotlib.pyplot as plt

import built_in

tf.get_logger().setLevel('ERROR')


class TrainModel:
    
    def __init__(self, train_dir, test_dir, bert_model_name):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.bert_model_name = bert_model_name
        
        self.tfhub_handle_encoder = built_in.map_name_to_handle(
            self.bert_model_name)
        self.tfhub_handle_preprocess = built_in.map_model_to_preprocess(
            self.bert_model_name)
    
    def set_up_data(self, batch_size=32, seed=42, validation_split=0.2):
        label_mode='categorical'
        autotune = tf.data.experimental.AUTOTUNE
        
        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.train_dir,
            label_mode=label_mode,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=seed)
        
        class_names = raw_train_ds.class_names
        train_ds = raw_train_ds.cache().prefetch(buffer_size=autotune)
        
        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.train_dir,
            label_mode=label_mode,
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            seed=seed)
        
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)
        
        test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            self.test_dir,
            label_mode=label_mode,
            batch_size=batch_size)
        
        test_ds = test_ds.cache().prefetch(buffer_size=autotune)
        return train_ds, val_ds, test_ds, class_names
    
    def compose_model(self, num_classes, dropout_rate=0.1):
        text_input = tf.keras.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(
            self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(
            self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(dropout_rate)(net)
        net = tf.keras.layers.Dense(
            num_classes, activation='softmax', name='classifier')(net)
        return tf.keras.Model(text_input, net)
    
    def compile_model(self, model, train_ds, epochs=5, init_lr=3e-5):
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = tf.metrics.CategoricalCrossentropy()
        
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)
        
        optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)
        return model
    
    def train_model(self, model, train_ds, val_ds, epochs=5):
        print(f'Training model with {self.tfhub_handle_encoder}')
        history = model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=epochs)
        return model, history
    
    def save_model(self, model, saved_model_path):
        model.save(saved_model_path, include_optimizer=False)
        
    def evaluate(self, model, test_ds):
        # Returns the loss & accuracy for the model in test mode.
        return model.evaluate(test_ds)


def load_model(saved_model_path):
    return tf.saved_model.load(saved_model_path)


def predict(model, sentences):
    return model(tf.constant(sentences))


if __name__ == '__main__':
    # Training model
    train_dir, test_dir = './train', './test'
    bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
    saved_model_path = './saved_model'
    
    tm = TrainModel(train_dir, test_dir, bert_model_name)
    train_ds, val_ds, test_ds, class_names = tm.set_up_data()
    model = tm.compose_model(len(class_names))
    model = tm.compile_model(model, train_ds)
    model, history = tm.train_model(model, train_ds, val_ds)
    tm.save_model(model, saved_model_path)
    
    # Inference
    sentences = [
        'this is such an amazing movie!',
        'The movie was great!',
        'The movie was meh.',
        'The movie was okish.',
        'The movie was terrible...',
    ]
    
    model = load_model(saved_model_path)
    results = predict(model, sentences)
