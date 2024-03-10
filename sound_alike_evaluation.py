

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
from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS



#This class reads the excel file with medication and preprocess it
#The ideia is to generate a list of words and an appropriate dataset
class GenerateWordList:
    def __init__(self,chosenMode, path, saltFile='sais.xlsx'):
        self.mode = chosenMode
        self.file_path = path
        # Load the salts file
        self.salts_df = pd.read_excel(saltFile, header=None)

        #Normalizing the salt names in the same way as performed in the main table
        self.salts_df.iloc[:, 0] =  self.salts_df.iloc[:, 0].apply(self.normalize_name)

        # Converting the salts column to a list and preparing the regular expression with unidecode to ignore accents
        self.words_to_remove =  self.salts_df.iloc[:, 0].tolist()

        self.regex_pattern = r'\b(' + '|'.join(self.words_to_remove) + r')\b'

    # Function to remove the words and ignore accents
    def remove_words(self,substance):
        normalized_substance = unidecode(substance)  # Normalize the substance to ignore accents
        # Using a regular expression to replace the words with an empty string
        cleaned_substance = re.sub(self.regex_pattern, '', normalized_substance, flags=re.IGNORECASE).strip()
        return cleaned_substance

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

        #TODO When we print the substance it is normalized with no salts
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

        #Remove salts from substance name
        # Applying the function to remove the words from the SUBSTANCIA_NORMALIZADA column with accent ignoring
        self.dfComercial['SUBSTANCIA_NORMALIZADA'] = self.dfComercial['SUBSTANCIA_NORMALIZADA'].apply(lambda x: self.remove_words(x))

        #Normalizing again just to make sure (e.g., to remove double spaces)
        self.dfComercial['SUBSTANCIA_NORMALIZADA'] = self.dfComercial['SUBSTANCIA_NORMALIZADA'].apply(self.normalize_name)

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
        self.product_substance_original_map = self.df.set_index('PRODUTO_NORMALIZADO')['SUBSTÂNCIA'].to_dict()

class EvaluateSimilarity:
    def __init__(self,words, product_substance_map, product_substance_original_map):
        self.words = words
        self.product_substance_map = product_substance_map
        self.product_substance_original_map = product_substance_original_map
        self.similarities = {}
        self.distances_count = None
        self.sorted_distances = None

    def print_similarity(self,pair,similarity,outputFile = None):
        word1 = pair.split("-")[0]
        word2 = pair.split("-")[1]
        if (outputFile == None):
            outputFile = sys.stdout
        print("********",file=outputFile)
        print("PRODUTO A - Nome: " + str(word1) + " - Substancia: " + str(self.product_substance_original_map.get(word1,None)),file=outputFile)
        print("PRODUTO B - Nome: " + str(word2) + " - Substancia: " + str(self.product_substance_original_map.get(word2,None)),file=outputFile)
        print("Distancia = " + str(similarity),file=outputFile)

    def print_top_k_similarity(self,k=1000,outputFileName=''):

        sorted_similarities = sorted(self.similarities.items(), key=lambda item: item[1], reverse=False)

        sorted_similarities_n_more_similar = sorted_similarities[:k]

        if(outputFileName == ''):
            outputFile = None
        else:
            outputFile = open(outputFileName, 'w')
        for pair, similarity in sorted_similarities_n_more_similar:
            self.print_similarity(pair,similarity,outputFile)
        outputFile.close()

    def print_pairs_with_distance_d(self, d, outputFileName=''):

        sorted_similarities = sorted(self.similarities.items(), key=lambda item: item[1], reverse=False)

        # Filterng similarities to keep those equal or less than d
        filtered_similarities = [(pair, sim) for pair, sim in sorted_similarities if sim <= float(float(d)+0.001)]

        if outputFileName == '':
            outputFile = None
        else:
            outputFile = open(outputFileName, 'w')

        for pair, similarity in filtered_similarities:
            self.print_similarity(pair, similarity, outputFile)

        # Fechando o arquivo, se necessário
        if outputFile is not None:
            outputFile.close()

    # Lembre-se de substituir 'self.similarities' pelo seu dicionário contendo as distâncias
    # E também de adaptar 'self.print_similarity()' se necessário, conforme a sua implementação atual



    def countSimilarities(self):
        #Building a list with all similarities
        distances = list(self.similarities.values())

        #Counting the number of pairs with a given similarity
        self.distances_count = Counter(distances)

        # Sorting the distances
        self.sorted_distances = sorted(self.distances_count.keys())

        print("A lista abaixo mostra a quantidade de pares para uma dada distancia:")
        for distance in self.sorted_distances:
            print("Distancia: " + str(distance) + " Quantidade: " + str(self.distances_count[distance]))


    #Build an Histogram for up to distance N
    def build_distance_graph(self,fileNamePrefix,N=None):

        if ((self.distances_count == None) or (self.sorted_distances == None)):
            self.countSimilarities()

        if (N == None):
            N = self.sorted_distances[len(self.sorted_distances)-1]

        #The counts for a given distance
        counts = [self.distances_count[distance] for distance in self.sorted_distances]

        # Considering only count equal or less than N
        filtered_distances = [distance for distance in self.sorted_distances if distance <= N]
        filtered_count = [counts[i] for i, distance in enumerate(self.sorted_distances) if distance <= N]

        # Generating the graph
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_distances, filtered_count, color='skyblue', edgecolor='black')
        plt.title('Distance Between Pairs Histogram')
        plt.xlabel('Distance')
        plt.ylabel('Number of pairs')
        plt.xticks(filtered_distances)
        plt.grid(axis='y', alpha=0.75)

        # Saving the graph in pdf
        plt.savefig(fileNamePrefix+'_distanceGraph_'+str(N).replace(".", "_")+'.pdf', format='pdf')



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

    #TODO this function shoudl only run when self.similarities is available
    def calculate_distance_matrix_from_dict(self):
        size = len(self.words)
        distance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                key = f"{self.words[i]}-{self.words[j]}"
                if key in self.similarities:
                    dist = self.similarities[key]
                else:
                    # Try the inverse key
                    key = f"{self.words[j]}-{self.words[i]}"
                    if key in self.similarities:
                        dist = self.similarities.get(key, 0)  # Assume 0 se não encontrar; ajuste conforme necessário
                    else:
                        #This case happens when the medications have the same substance
                        #Hence, we put an infinite distance (50) between them
                        #TODO 50 is hardcoded due to previous experimentos (there is no distance larger than 49.5)
                        dist = 50
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def cluster_and_plot_dbscan(self, fileNamePrefix,eps=2, min_samples=2):
        distance_matrix = self.calculate_distance_matrix_from_dict()
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = db.fit_predict(distance_matrix)

        # Aplica MDS para visualização
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        positions = mds.fit_transform(distance_matrix)

        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], c=labels, cmap='viridis', s=100)

        for i, word in enumerate(self.words):
            plt.text(positions[i, 0], positions[i, 1] + 0.02, word, ha='center')

        plt.title('Palavras Clusterizadas com DBSCAN e Distância de Levenshtein')
        plt.xlabel('MDS 1')
        plt.ylabel('MDS 2')

        #Saving graph
        plt.savefig(fileNamePrefix+'_cluster.pdf', format='pdf')


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
    def run(self, num_processes=1):
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

    #Defining the path to store results
    path = './output'
    if not os.path.exists(path):
        # O diretório não existe; criando-o
        os.makedirs(path)

    fileNamePrefix = path + str("/") + mode

    #Generating the word list based on the chosen mode
    print("Lendo o arquivo para gerar a lista de palavras")
    word_list_generator = GenerateWordList(mode,'./listaMedicamentosTeste.xls')
    word_list_generator.run()

    print("Calculando as similaridades")
    #Evaluating similarities
    similarity_evaluator = EvaluateSimilarity(word_list_generator.words,word_list_generator.product_substance_map, word_list_generator.product_substance_original_map)
    #The run method will receive the number of processes to create
    #In this case we will use all the available cores
    similarity_evaluator.run(int(os.cpu_count()))

    print("Salvando arquivo com as k similaridades")
    #Print top K similairties
    #TODO modify hardcoded k
    k = 100
    similarity_evaluator.print_top_k_similarity(k,fileNamePrefix+"_top_" + str(k) + ".txt")

    print("Salvando arquivo com similaridade ate 1")
    #Print top K similairties
    #TODO modify hardcoded k
    d = 1
    similarity_evaluator.print_pairs_with_distance_d(d,fileNamePrefix+"_similarityUpTo_" + str(d) + ".txt")

    print("Salvando arquivo com similaridade ate 2")
    #Print top K similairties
    #TODO modify hardcoded k
    d = 2
    similarity_evaluator.print_pairs_with_distance_d(d,fileNamePrefix+"_similarityUpTo_" + str(d) + ".txt")

    print("Salvando arquivo com histograma")
    similarity_evaluator.build_distance_graph(fileNamePrefix)

    print("Salvando arquivo com histograma até distancia 2")
    similarity_evaluator.build_distance_graph(fileNamePrefix,2)

    similarity_evaluator.cluster_and_plot_dbscan(fileNamePrefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Programa opera em um dos tres modos.')
    parser.add_argument('mode', help='Modo de operacao (comercial, dcb, todos)')

    args = parser.parse_args()

    main(args.mode)

    sys.exit(1)
