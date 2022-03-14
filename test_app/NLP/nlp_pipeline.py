from src import nlp, pdf_reader, gen_functions
import os

if __name__ == '__main__':

    # download stanza em portugues
    nlp.download_stanza_portugues()

    """
    1) Carregar o conjunto de documentos em PDF e armazená-los em alguma estrutura de dados
    """

    os.chdir(os.path.dirname(__file__))
    current_dir_path = os.getcwd()

    data_directory_path = current_dir_path + '/data'
    corpus = pdf_reader.join_and_remove_breaks(pdf_reader.load_pfds(data_directory_path))

    """
    2) Realizar o pré-processamento destes ( tokenização e remoção de stop words, deixar todos 
    os caracteres minúsculos...)
    """

    corpus_clean_text = [nlp.clean_text(doc) for doc in corpus]
    corpus_sem_stopwords = nlp.removestopwords(corpus_clean_text)
    corpus_sem_stopwords = list(map(gen_functions.convert_list_to_string, corpus_sem_stopwords))

    """
    3) lematização
    """

    # tokenização e lematização com a biblioteca stanza
    corpus_lematizado_stanza_doc = [nlp.tokenizer_and_lemmatizer(doc) for doc in corpus_sem_stopwords]
    corpus_lematizado_lists = nlp.stanza_sentence_to_list_of_lemmas(corpus_lematizado_stanza_doc)

    """
    5.1) Term Frequency (TF):
    """

    term_frequency = nlp.term_frequency(corpus_lematizado_lists)

    """
    5.2) Document Frequency (DF)
    """

    document_frequency = nlp.document_frequency(corpus_lematizado_lists)

    """
    5.3) Inverse Document Frequency (IDF)
    """

    inverse_document_frequency = nlp.inverse_document_frequency(corpus_lematizado_lists)

    """
    5.4) TF-IDF
    """

    tf_idf_dicts = nlp.calc_tf_idf(corpus=corpus_lematizado_lists, tf=term_frequency, idf=inverse_document_frequency)

    """
    5.5) Lista de strings com proximidade até 2 dos 5 termos de maior TF-IDF.
    Essas strings devem ser acompanhadas de seu valor de TF.
    """

    termos_de_maior_tf_idf = nlp.calc_termos_maior_tf_idf(corpus=corpus_lematizado_lists,
                                                          n_words=5,
                                                          tf_idf_dicts=tf_idf_dicts)

    termos_de_maior_tf_idf_prox_dict = nlp.terms_neighborhood(corpus=corpus_lematizado_lists,
                                                              terms=termos_de_maior_tf_idf,
                                                              tf_dicts=term_frequency,
                                                              n_neighbors=2)

    """
    6) Gerar um arquivo csv que possui todas as palavras de todos os documentos na primeira coluna,
    em que cada linha é um token. Para cada token, informe nas colunas vizinhas as informações
    determinadas no objetivo 5.
    """

    output_directory = current_dir_path + '/resultados'

    nlp.write_csv_metricas_gerais(corpus=corpus_lematizado_lists,
                                  tf_dicts=term_frequency,
                                  df_dict=document_frequency,
                                  idf_dict=inverse_document_frequency,
                                  tf_idf_dicts=tf_idf_dicts,
                                  output_dir=output_directory
                                  )

    nlp.write_csv_tokens_maior_tf_idf(tokens_maior_tf_idf_dict=termos_de_maior_tf_idf_prox_dict,
                                      output_dir=output_directory)
