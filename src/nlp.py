import re
import nltk
import string
import stanza
import numpy as np
from src import gen_functions
import csv


def download_stanza_portugues():
    """
    Faz o download do stanza em portugues
    """
    stanza.download(lang='pt')


def tokenizer_and_lemmatizer(text):
    """
        Performs tokenization and lemmatization on input text

    Args:
        text: A string with the content of the text

    Returns:
        A stanza Document with the tokens and lemmas

    """
    nlp = stanza.Pipeline('pt', processors='tokenize,mwt,pos,lemma')
    return nlp(text)


def show_nlp_doc(doc):
    """
    Imprime os tokens (somente para debug)
    """
    sentence_id = 0
    for sentence in doc.sentences:
        sentence_id += 1
        print('\nSentença {}:'.format(sentence_id))
        for word in sentence.words:
            print('palavra = {}, lema = {}, id = {}'.format(word.text, word.lemma, word.id))


def neighborhood_by_sentences(doc, tf_dict, lemma, threshold):
    """
    Retorna a lista de vizinhos, com distancia threshold, do lemma passado como parametro
    considerando somente a sentença em que o lema se encontra

    percorre todas as sentenças e retorna uma lista de vizinhos

    * Não foi usada no trabalho pois utilizou-se a convenção de considerar
    vizinhos entre sentenças vizinhas
    """
    neighbors_list = []
    for sentence in doc.sentences:
        sentence_list = []
        for word in sentence.words:
            sentence_list.append(word.lemma)
        current_neighbors_list = gen_functions.string_proximity(sentence_list, lemma, threshold)

        for current_neighbor in current_neighbors_list:
            if current_neighbor not in neighbors_list:
                neighbors_list.append(current_neighbor)

    neighbors_tf_dict = dict()
    for lemma in neighbors_list:
        neighbors_tf_dict[lemma] = tf_dict[lemma]

    return neighbors_tf_dict


def print_sentences(doc):
    """
    Imprime as sentenças separadamente
    """
    for sentence in doc.sentences:
        sentence_list = []
        for word in sentence.words:
            sentence_list.append(word.lemma)
        print(sentence_list)


def stanza_sentence_to_list_of_lemmas(corpus_lematizado_stanza_doc):
    """
    Recebe um Document da biblioteca stanza

    Retorna uma lista de lemas, contendo todos os lemas do documento em ordem
    """
    list_of_list_of_lemmas = []
    for document in corpus_lematizado_stanza_doc:
        document_list = []
        for sentence in document.sentences:
            for word in sentence.words:
                document_list.append(word.lemma)
        list_of_list_of_lemmas.append(document_list)
    return list_of_list_of_lemmas


def clean_text(text):
    """
    Realiza algumas operações de limpeza em uma string
    remove numeros
    remove pontuações
    substitui maisculas por minusculas
    remove espaços em branco repetidos
    """
    # remove numbers
    text_nonum = re.sub(r'\d+', '', text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    return text_no_doublespace


def removestopwords(texto):
    """
    Utiliza a lista de stopwords em portugues definida pela biblioteca nltk
    para realizar a remoção de stopwords do texto passado como parametro
    """
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    frases = []
    for palavras in texto:
        semstop = [p for p in palavras.split() if p not in stopwords]
        frases.append(semstop)
    return frases


def term_frequency(corpus):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Retorna uma lista de dicionários, um para cada documento
    Cada dicionário contem os lemas presentes no documento como chaves e suas respectivas frequencias
    como valores
    """

    ocorrencias_termo = list(map(gen_functions.map_occurrences, corpus))
    termos_documento = list(map(len, corpus))

    result = {}
    for i in range(len(ocorrencias_termo)):
        result[i] = dict(map(lambda kv: (kv, ocorrencias_termo[i][kv] / termos_documento[i]), ocorrencias_termo[i]))
    return result


def document_frequency(corpus):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Retorna um dicionário com todos os lemas presentes no corpus como chaves
    cada lema com seu respectivo document frequency como valor
    """
    todas_palavras = gen_functions.concatenate_lists_without_repetitions(*corpus)

    df = dict.fromkeys(todas_palavras, 0)

    for word in todas_palavras:
        for doc in corpus:
            if word in doc:
                df[word] += 1

    return df


def inverse_document_frequency(corpus):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Retorna um dicionário com todos os lemas presentes no corpus como chaves
    cada lema com seu respectivo inverse document frequency como valor
    """
    import math
    df = document_frequency(corpus)
    idf = {k: math.log10((len(corpus) / v)) for k, v in df.items()}
    # a divisão é feita por "v" ao invés de "v+1" pois não há tokens não presentes no corpus
    return idf


def calc_tf_idf(corpus, tf=None, idf=None):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Retorna uma lista de dicionários, um para cada documento
    Cada dicionário contem os lemas presentes no documento como chaves e suas respectivas tf-idf
    como valores

    Os dicionários tf e idf podem ser passados como parametros para evitar recalculos
    """

    if tf is None:
        tf = term_frequency(corpus)
    if idf is None:
        idf = inverse_document_frequency(corpus)

    list_of_tf_idf_dicts = []

    for i in range(len(corpus)):
        doc_tf_idf_dict = tf[i].copy()
        for word in doc_tf_idf_dict.keys():
            doc_tf_idf_dict[word] = tf[i][word] * idf[word]
        list_of_tf_idf_dicts.append(doc_tf_idf_dict)

    return list_of_tf_idf_dicts


def calc_termos_maior_tf_idf(corpus, n_words=5, tf_idf_dicts=None):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Retorna uma lista com as palavras de maior tf-idf do corpus

    * Os dicionários de tf-idfs podem ser passados como parametro para evitar recalculos
    """

    if tf_idf_dicts is None:
        tf_idf_dicts = calc_tf_idf(corpus)

    tf_idf_all_list = []
    for i in range(len(corpus)):
        for word in tf_idf_dicts[i].keys():
            tf_idf_all_list.append((word, tf_idf_dicts[i][word]))

    tf_idf_all_list.sort(key=lambda x: x[1], reverse=True)

    n_palavras_mais_significativas = []
    for i in range(n_words):
        n_palavras_mais_significativas.append(tf_idf_all_list[i][0])

    return n_palavras_mais_significativas


def terms_neighborhood(corpus, terms, tf_dicts=None, n_neighbors=2):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Recebe termos, uma lista com os tokens para os quais se quer pesquisar os tokens de maior proximidade

    Retorna um dicionário contendo os termos como chaves e os vizinhos, encontrados em todo o corpus, como
    valores
    """

    if tf_dicts is None:
        tf_dicts = term_frequency(corpus)

    palavras_mais_significativas_dict = {}
    for word in terms:
        full_list_of_neighbors = []
        for i in range(len(corpus)):
            parcial_list_of_neighbors = []
            parcial_list_of_neighbors_plus_tf = []
            if word in corpus[i]:
                parcial_list_of_neighbors += gen_functions.string_proximity(corpus[i], word, n_neighbors)
            for neighbor in parcial_list_of_neighbors:
                parcial_list_of_neighbors_plus_tf.append([neighbor, tf_dicts[i][neighbor]])
            full_list_of_neighbors += parcial_list_of_neighbors_plus_tf
        palavras_mais_significativas_dict[word] = full_list_of_neighbors

    return palavras_mais_significativas_dict


def write_csv_metricas_gerais(corpus, tf_dicts, df_dict, idf_dict, tf_idf_dicts, output_dir):
    """
    Recebe o corpus, formatado como uma lista de documentos, cada documento formatado como uma lista de
    tokens ou lemas

    Cria o csv contendo as metricas de TF, IDF e TF-IDF para cada token presente no corpus
    """
    with open(f'{output_dir}/metricas-gerais.csv', mode='w', newline='') as csv_file:
        fieldnames = ["PALAVRA", "TF-doc1", "TF-doc2", "TF-doc3",
                      "DF", "IDF", "TF-IDF-doc1", "TF-IDF-doc2", "TF-IDF-doc3"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        all_words = gen_functions.concatenate_lists_without_repetitions(*corpus)

        tf = ['-'] * len(corpus)
        tf_idf = ['-'] * len(corpus)
        for word in all_words:
            for i in range(len(corpus)):
                if word in corpus[i]:
                    tf[i] = tf_dicts[i][word]
                    tf_idf[i] = tf_idf_dicts[i][word]
                else:
                    tf[i] = '-'
                    tf_idf[i] = '-'
            df = df_dict[word]
            idf = idf_dict[word]
            writer.writerow({"PALAVRA": word,
                             "TF-doc1": str(tf[0]),
                             "TF-doc2": str(tf[1]),
                             "TF-doc3": str(tf[2]),
                             "DF": str(df),
                             "IDF": str(idf),
                             "TF-IDF-doc1": str(tf_idf[0]),
                             "TF-IDF-doc2": str(tf_idf[1]),
                             "TF-IDF-doc3": str(tf_idf[2])
                             })
    print(f'{output_dir}/metricas-gerais.csv criado com sucesso!')


def write_csv_tokens_maior_tf_idf(tokens_maior_tf_idf_dict, output_dir):
    with open(f'{output_dir}/tokens_maior_tf_idf.csv', mode='w', newline='') as csv_file:
        fieldnames = ["tokens de maiores tf-idf", "lista de prox com tf de cada string"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for word in tokens_maior_tf_idf_dict.keys():
            writer.writerow({"tokens de maiores tf-idf": str(word),
                             "lista de prox com tf de cada string": str(tokens_maior_tf_idf_dict[word]),
                             })
    print(f'{output_dir}/tokens_maior_tf_idf.csv criado com sucesso!')
