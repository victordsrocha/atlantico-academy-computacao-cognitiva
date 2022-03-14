import os
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split


def load_dataset(data_dir):
    """
    Recebe o path para o conjunto de imagens e retorna uma lista de listas de imagens, separadas por classe,
    representadas como numpy arrays.

    Retorna também um numpy array contendo o nome das classes.

    :param data_dir: string
    :return: (list[list[np.array]], np.array)
    """
    data_dir = pathlib.Path(data_dir)
    class_names = np.array([item.name for item in data_dir.glob('*')])

    fnames = []
    for classes in class_names:
        images_folder = os.path.join(data_dir, classes)
        file_names = os.listdir(images_folder)
        full_path = [os.path.join(images_folder, file_name) for file_name in file_names]
        fnames.append(full_path)

    # Carregando as imagens das fotos usando CV2
    images = []
    for names in fnames:
        one_class_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
        images.append(one_class_images)

    return images, class_names


def custom_train_test_val_split(images, random_state=42):
    """
    Realiza a separação dos dados em treinamento, validação e testes ~ (70/20/10)

    :param images: list[list[np.array]]
    :param random_state: int
    :return: (list[list[np.array]], list[list[np.array]], list[list[np.array]])
    """
    # Criando as listas vazias
    train_images = []
    val_images = []
    test_images = []
    aux_images = []

    # Loop percorrendo todas as imagens redimensionadas e preenchendo as listas de treino e validação
    for imgs in images:
        train, test = train_test_split(imgs, train_size=0.9, test_size=0.1,
                                       random_state=random_state)
        aux_images.append(train)
        test_images.append(test)

    for imgs2 in aux_images:
        train, val = train_test_split(imgs2, train_size=0.78, test_size=0.22,
                                      random_state=random_state)
        train_images.append(train)
        val_images.append(val)

    return train_images, val_images, test_images


def create_labels(train_images, val_images, test_images):
    """
    Retorna os rótulos de cada conjunto de imagens mapeados em um np.array de inteiros.

    :param train_images: list[list[np.array]]
    :param val_images: list[list[np.array]]
    :param test_images: list[list[np.array]]
    :return: np.array
    """
    # tamanhos das classes
    len_train_images = [len(imgs) for imgs in train_images]
    len_val_images = [len(imgs) for imgs in val_images]
    len_test_images = [len(imgs) for imgs in test_images]

    # arrays para armazenar os labels
    train_classe = np.zeros((np.sum(len_train_images)), dtype='uint8')
    val_classe = np.zeros((np.sum(len_val_images)), dtype='uint8')
    test_classe = np.zeros((np.sum(len_test_images)), dtype='uint8')

    # atribuição
    for i in range(4):
        if i == 0:
            train_classe[:len_train_images[i]] = i
            val_classe[:len_val_images[i]] = i
            test_classe[:len_test_images[i]] = i
        else:
            train_classe[np.sum(len_train_images[:i]):np.sum(len_train_images[:i + 1])] = i
            val_classe[np.sum(len_val_images[:i]):np.sum(len_val_images[:i + 1])] = i
            test_classe[np.sum(len_test_images[:i]):np.sum(len_test_images[:i + 1])] = i

    return train_classe, val_classe, test_classe


def convert_to_numpy(train_images, val_images, test_images):
    """
    Retorna as listas de imagens formatadas em um único np.array

    :param train_images: list[list[np.array]]
    :param val_images: list[list[np.array]]
    :param test_images: list[list[np.array]]
    :return: (np.array, np.array, np.array)
    """
    # Criando listas temporarias
    tmp_train_imgs = []
    tmp_val_imgs = []
    tmp_test_imgs = []

    # Percorrendo o dataset de treinamento e adicionando na lista temporaria
    for imgs in train_images:
        tmp_train_imgs += imgs

    # Percorrendo o dataset de validação e adicionando na lista temporaria
    for imgs in val_images:
        tmp_val_imgs += imgs

    # Percorrendo o dataset de testes e adicionando na lista temporaria
    for imgs in test_images:
        tmp_test_imgs += imgs

    # Convertendo em formato array
    train_images_np = np.array(tmp_train_imgs)
    val_images_np = np.array(tmp_val_imgs)
    test_images_np = np.array(tmp_test_imgs)

    # Transformando os dados para o tipo float32
    train_data = train_images_np.astype('float32')
    val_data = val_images_np.astype('float32')
    test_data = test_images_np.astype('float32')

    return train_data, val_data, test_data


def convert_to_rgb(img):
    """
    Recebe uma imagem no formato BGR e a retorna no formato RGB

    :param img: np.array
    :return: np.array
    """
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)


def create_model(model_type, input_shape, number_of_classes, metrics):
    """
    Cria o modelo de rede neural de acordo com os parâmetros.

    :param model_type: str (cnn ou vgg)
    :param input_shape: (int, int, int)
    :param number_of_classes: int
    :param metrics: list (lista contendo os parâmetros de métricas da biblioteca keras)
    :return: modelo de rede neural da biblioteca keras
    """
    if model_type == 'cnn':
        return create_model_cnn(input_shape, number_of_classes, metrics)
    elif model_type == 'vgg':
        return create_model_vgg19(input_shape, number_of_classes, metrics)


def create_model_cnn(input_shape, number_of_classes, metrics):
    """
    Cria o modelo de rede neural do tipo cnn de acordo com os parâmetros.

    :param input_shape: (int, int, int)
    :param number_of_classes: int
    :param metrics: list (lista contendo os parâmetros de métricas da biblioteca keras)
    :return: modelo de rede neural do tipo cnn da biblioteca keras
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))

    input_shape = input_shape
    model.build((None,) + input_shape)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=metrics)

    return model


def create_model_vgg19(input_shape, number_of_classes, metrics):
    """
    Cria o modelo de rede neural do tipo vgg19 de acordo com os parâmetros.

    :param input_shape: (int, int, int)
    :param number_of_classes: int
    :param metrics: list (lista contendo os parâmetros de métricas da biblioteca keras)
    :return: modelo de rede neural do tipo vgg19 da biblioteca keras
    """

    model = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)

    # Congelando as camadas que não serão treinadas
    for layer in model.layers[:20]:
        layer.trainable = False

    # Adicionando nova camadas ao nosso modelo
    x = model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(number_of_classes, activation="softmax")(x)

    # Criando o modelo final
    final_model = Model(inputs=model.input, outputs=predictions)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=metrics)

    return final_model


def plot_model(history, epochs):
    """
    Exibe os gráficos de treinamento.

    :param history: dict
    :param epochs: int
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history['precision'])), history['precision'], 'r')
    plt.plot(np.arange(1, len(history['val_precision']) + 1), history['val_precision'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Precision vs. Validation Precision')
    plt.xlabel('Nro de Epochs')
    plt.ylabel('Precision')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['recall']) + 1), history['recall'], 'r')
    plt.plot(np.arange(1, len(history['val_recall']) + 1), history['val_recall'], 'g')
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title('Training Recall vs. Validation Recall')
    plt.xlabel('Nro de Epochs')
    plt.ylabel('Recall')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def predict_val(test_data, model):
    """
    Função para realizar previsão da classe das imagens passadas como parâmetro
    Retorna o rótulo e a probabilidade estimada da previsão

    :param test_data: np.array
    :param model: modelo de rede neural da biblioteca keras
    :return: int, float
    """
    val_input = np.reshape(test_data, (1, 256, 256, 3))
    val_input = val_input / 255.
    pred = model.predict(val_input)
    class_num = np.argmax(pred)
    return class_num, np.max(pred)


def desc_label(label):
    """
    Converte a formatação one-hot-encoding para o inteiro correspondente.
    :param label: np.array
    :return: int
    """
    idx = np.where(label == 1)
    return idx[0][0]


def show_predictions(model, test_data, test_labels, class_names):
    """
    Exibe alguns exemplos de imagens com os respectivos rótulos e previsões.

    :param model: modelo de rede neural da biblioteca keras
    :param test_data: np.array
    :param test_labels: np.array
    :param class_names: np.array
    """
    # Realizando as previsões e exibindo as imagens com os labels verdadeiros e previstos
    plt.figure(figsize=(15, 15))
    for i in range(9):
        idx = np.random.randint(len(test_data))

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(convert_to_rgb(test_data.astype('uint8')[idx]))
        class_idx = desc_label(test_labels[idx])

        pred, prob = predict_val(test_data[idx], model)
        plt.title('True: %s || Pred: %s %d%%' % (class_names[class_idx], class_names[pred], round(prob, 2) * 100))
        plt.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()


def reshape_img_dataset(images, new_shape):
    """
    Transforma todas as imagens para a resolução escolhida.

    :param images: list[list[np.array]]
    :param new_shape: (int, int)
    :return: list[list[np.array]]
    """
    img_width, img_height = new_shape

    resized_images = []
    for i, imgs in enumerate(images):
        resized_images.append([cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC) for img in imgs])
    return resized_images


def apply_bilateral_filter(images, d=15, sigma_color=75, sigma_space=75):
    """
    Aplica o processamento de filtro bilateral em todas as imagens.

    :param images: list[list[np.array]]
    :param d: int
    :param sigma_color: int
    :param sigma_space: int
    :return: list[list[np.array]]
    """
    filtered_images = []
    for i, imgs in enumerate(images):
        filtered_images.append([cv2.bilateralFilter(img, d, sigma_color, sigma_space) for img in imgs])
    return filtered_images
