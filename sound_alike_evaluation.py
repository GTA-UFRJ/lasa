

import pandas as pd
import argparse
import sys
from unidecode import unidecode
import re
import Levenshtein
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from metaphoneptbr import phonetic



#This class reads the excel file with medication and preprocess it
#The ideia is to generate a list of words and an appropriate dataset
class GenerateWordList:
    def __init__(self,chosenMode, path):
        self.mode = chosenMode
        self.file_path = path

    # Normalize the names
    def normalize_name(self,name):
        # Removes diacritics and making the name upper case
        normalized_name = unidecode(name.upper())
        # Replacing special characters to blank spaces
        normalized_name = re.sub(r'[^a-zA-Z0-9]',' ', normalized_name)
        # Replacing multiple blank spaces to single blank spaces
        normalized_name = re.sub(r'\s+', ' ', normalized_name)
        return normalized_name

    # Generate a dataframe tha replicates the substance name on the product column
    def generate_dcb(self):

        substances_uniques = self.dfComercial['SUBSTANCIA_NORMALIZADA'].unique()

        #Create empty dataframe
        new_rows_list = []

        #Tambem queremos considerar generico. Assim, por simplicidade, fazemos uma nova linha para cada substancia
        # Criar uma nova linha para cada substância única
        for substance in substances_uniques:
            new_line = {"SUBSTÂNCIA": substance, "PRODUTO": substance, "PRODUTO_NORMALIZADO": substance, "SUBSTANCIA_NORMALIZADA": substance}
            new_rows_list.append(new_line)

        return pd.DataFrame(new_rows_list)

    def run(self):
        #Reading excel file
        self.dfComercial = pd.read_excel(self.file_path)

        # Normalizing the columns
        self.dfComercial['PRODUTO_NORMALIZADO'] = self.dfComercial['PRODUTO'].apply(self.normalize_name)
        self.dfComercial['SUBSTANCIA_NORMALIZADA'] = self.dfComercial['SUBSTÂNCIA'].apply(self.normalize_name)

        if self.mode == 'comercial':
            print("Comparando apenas comerciais")
            self.df = self.dfComercial
        elif self.mode == 'dcb':
            print("Comparando apenas DCB (Denominacao Comum Brasileira)")
            self.df = self.generate_dcb()
        elif self.mode == 'todos':
            # Usar pd.concat para adicionar as novas linhas ao DataFrame original
            self.df = pd.concat([self.dfComercial, self.generate_dcb()], ignore_index=True)
        else:
            print("Erro desconhecido na selecao de modos")
            return None

        # Removendo duplicatas com base na coluna normalizada
        self.words = self.df['PRODUTO_NORMALIZADO'].drop_duplicates().tolist()

        # Criando um mapa de produto com substancia
        self.product_substance_map = self.df.set_index('PRODUTO_NORMALIZADO')['SUBSTANCIA_NORMALIZADA'].to_dict()

class EvaluateSimilarity:
    def __init__(self,words, product_substance_map):
        self.words = words
        self.product_substance_map = product_substance_map
        self.similarities = {}

    def print_similarity(self,pair,similarity,outputFile = None):
        word1 = pair.split("-")[0]
        word2 = pair.split("-")[1]
        if (outputFile == None):
            outputFile = sys.stdout
        print("********",file=outputFile)
        print("PRODUTO A - Nome: " + str(word1) + " - Substancia: " + str(self.product_substance_map.get(word1,None)),file=outputFile)
        print("PRODUTO B - Nome: " + str(word2) + " - Substancia: " + str(self.product_substance_map.get(word2,None)),file=outputFile)
        print("Distancia = " + str(similarity),file=outputFile)

    def print_top_k_similarity(self,k,outputFileName=''):

        sorted_similarities = sorted(self.similarities.items(), key=lambda item: item[1], reverse=False)

        sorted_similarities_n_more_similar = sorted_similarities[:k]

        if(outputFileName == ''):
            outputFile = None
        else:
            outputFile = open(outputFileName, 'w')
        for pair, similarity in sorted_similarities_n_more_similar:
            self.print_similarity(pair,similarity,outputFile)
        outputFile.close()

    def same_substance(self, product1, product2):
        substance1 = self.product_substance_map.get(product1, None)
        substance2 = self.product_substance_map.get(product2, None)
        return substance1 == substance2 and substance1 is not None

    def similarity_levenshtein(self,word1, word2):
        # Convert words to their phonetic representation
        #meta1 = doublemetaphone(word1)[0]
        #meta2 = doublemetaphone(word2)[0]
        meta1 = word1
        meta2 = word2
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(meta1, meta2)

        # Convert distance to similarity score
        #similarity = np.exp(-distance)
        similarity = distance

        return similarity
    #Commenting the old way, unparallelized
    """
    def run(self):
        for i, word1 in enumerate(self.words):
            for j, word2 in enumerate(self.words):
                if i < j:  # Avoid repeating pairs and comparing words with themselves
                    #Eliminate pairs that consists of the same substance (no sound-alike problem)
                    if not self.same_substance(word1, word2):
                        key = f"{word1}-{word2}"
                        self.similarities[key] = self.similarity_levenshtein(word1, word2)
    """

    def calculate_similarities_for_batch(self, batch):
        batch_results = {}
        for pair in batch:
            word1, word2 = pair
            if not self.same_substance(word1, word2):
                key = f"{word1}-{word2}"
                #Similarity considering only the written word
                similarityWritten = self.similarity_levenshtein(word1, word2)
                #Similarity considering the phonetic using methaphone br
                similarityPhonetic = self.similarity_levenshtein(phonetic(word1),phonetic(word2))
                batch_results[key] = (float(similarityWritten) + float(similarityPhonetic))/float(2)
        return batch_results

    def divide_into_batches(self, pairs, num_batches):
        # Dividir a lista de pares em num_batches partes
        for i in range(0, len(pairs), num_batches):
            yield pairs[i:i + num_batches]

    #num_processes paralelize the evaluation, dividind the pair into batches
    def run(self, num_processes=None):
        pairs = [(self.words[i], self.words[j]) for i in range(len(self.words)) for j in range(i + 1, len(self.words))]
        num_batches = len(pairs) // num_processes if num_processes else len(pairs)

        batches = list(self.divide_into_batches(pairs, num_batches))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.calculate_similarities_for_batch, batch) for batch in batches]

            for future in futures:
                batch_results = future.result()
                self.similarities.update(batch_results)


def main(mode):
    if (mode not in ['comercial','dcb','todos']):
        print("Modo invalido selecionado. Escolha  'comercial', 'dcb', ou 'todos'.")
        print("\nUsage:")
        print("  comercial: Compara apenas nomes comeciais")
        print("  dcb: Comparando apenas DCB (Denominacao Comum Brasileira)")
        print("  todos: Comparando todos (comerciais e DCB)")
        sys.exit(1)

    #Generating the word list based on the chosen mode
    print("Lendo o arquivo para gerar a lista de palavras")
    word_list_generator = GenerateWordList(mode,'./listaMedicamentos.xls')
    word_list_generator.run()

    print("Calculando as similaridades")
    #Evaluating similarities
    similarity_evaluator = EvaluateSimilarity(word_list_generator.words,word_list_generator.product_substance_map)
    similarity_evaluator.run(8)

    #Print top K similairties
    similarity_evaluator.print_top_k_similarity(100,"top100fonemas.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Programa opera em um dos tres modos.')
    parser.add_argument('mode', help='Modo de operacao (comercial, dcb, todos)')

    args = parser.parse_args()

    main(args.mode)

    sys.exit(1)
