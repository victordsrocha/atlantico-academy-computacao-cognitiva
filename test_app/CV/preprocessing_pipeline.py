"""
Este pipeline recebe um dataset organizado da seguinte forma:

dataset/
    amostras_classe_1/
    amostras_classe_2/
    ...
    amostras_classe_n/

Os dados são preprocessados e salvos na pasta CV/preprocessed_data no formato npy,
separados em dados de treino, teste e validação
"""

import os
import numpy as np
from src import computer_vision
from tensorflow.python.keras.utils import np_utils


def main():
    # definição do path do diretorio de imagens
    os.chdir(os.path.dirname(__file__))
    current_dir_path = os.getcwd()

    # salvar o dataset completo dentro da pasta data e
    # alterar o path para o dataset completo!
    data_dir = current_dir_path + '/data/data_especular_crop_3_classes_reduzido_para_testes'

    # Carregamento dos dados
    images, class_names = computer_vision.load_dataset(data_dir)

    # redimensionamento
    resized_images = computer_vision.reshape_img_dataset(images, new_shape=(256, 256))

    # processamento de imagem
    filtered_images = computer_vision.apply_bilateral_filter(resized_images)

    # Realizando o split dos dados para treinamento, validação e testes ~ (70/20/10)
    train_images, val_images, test_images = computer_vision.custom_train_test_val_split(filtered_images)

    # Criando os labels para as imagens de treinamento
    train_classe, val_classe, test_classe = computer_vision.create_labels(train_images, val_images, test_images)

    # Convertendo as imagens em formato array do numpy para submeter ao treinamento
    train_data, val_data, test_data = computer_vision.convert_to_numpy(train_images, val_images, test_images)

    # Converter o array de labels para cada classe em vetores one-hot
    train_labels = np_utils.to_categorical(train_classe, len(class_names))
    val_labels = np_utils.to_categorical(val_classe, len(class_names))
    test_labels = np_utils.to_categorical(test_classe, len(class_names))

    # Salvando os valores em disco
    def save_list(my_list, filename):
        np.save(filename, my_list)
        print(f"{filename} Saved successfully!")

    save_list(class_names, 'preprocessed_data/class_names.npy')
    save_list(train_data, 'preprocessed_data/train_data.npy')
    save_list(val_data, 'preprocessed_data/val_data.npy')
    save_list(test_data, 'preprocessed_data/test_data.npy')
    save_list(train_labels, 'preprocessed_data/train_labels.npy')
    save_list(val_labels, 'preprocessed_data/val_labels.npy')
    save_list(test_labels, 'preprocessed_data/test_labels.npy')


if __name__ == '__main__':
    main()
