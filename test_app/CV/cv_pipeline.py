"""
este pipeline utiliza os dados gerados pelo preprocessing_pipeline
treina a rede neural escolhida e exibe o desempenho alcançado
"""

import os
import time
import numpy as np
import matplotlib
from tensorflow import keras
from src import computer_vision

# escolher o modelo de rede neural: cnn ou vgg
MODEL_CHOICE = 'cnn'  # 'cnn' ou 'vgg'

# hiperparametros gerais
BATCH_SIZE = 16
EPOCHS = 1


def main():
    # carregamento do dataset preprocessado

    os.chdir(os.path.dirname(__file__))
    current_dir_path = os.getcwd()
    data_directory_path = current_dir_path + '/preprocessed_data'
    class_names = np.load(os.path.join(data_directory_path, 'class_names.npy'))
    test_data = np.load(os.path.join(data_directory_path, 'test_data.npy'))
    test_labels = np.load(os.path.join(data_directory_path, 'test_labels.npy'))
    train_data = np.load(os.path.join(data_directory_path, 'train_data.npy'))
    train_labels = np.load(os.path.join(data_directory_path, 'train_labels.npy'))
    val_data = np.load(os.path.join(data_directory_path, 'val_data.npy'))
    val_labels = np.load(os.path.join(data_directory_path, 'val_labels.npy'))

    # métricas
    metrics = [keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]

    # Criando o modelo e verificando a estrutura
    model = computer_vision.create_model(model_type=MODEL_CHOICE,
                                         input_shape=train_data.shape[1:],
                                         number_of_classes=len(class_names),
                                         metrics=metrics)

    # exibindo resumo do modelo
    model.summary()

    # Marcando o tempo de início
    start = time.time()

    # Treinamento do modelo
    history_model = model.fit(train_data, train_labels, batch_size=BATCH_SIZE,
                              epochs=EPOCHS, initial_epoch=0,
                              validation_data=(val_data, val_labels))

    # Marcando o tempo final
    end = time.time()
    duration = end - start
    print('\n Modelo CNN - Duração %0.2f segundos (%0.1f minutos) para treinamento de %d epocas' % (
        duration, duration / 60, EPOCHS))

    # exibição dos gráficos de treinamento
    computer_vision.plot_model(history_model.history, EPOCHS)

    # exibição de exemplos de predição
    computer_vision.show_predictions(model=model,
                                     test_data=test_data,
                                     test_labels=test_labels,
                                     class_names=class_names)


if __name__ == '__main__':
    main()
