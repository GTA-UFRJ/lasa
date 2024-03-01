# -*- coding: utf-8 -*-
"""sound_alike_evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xmKyF3kAl-ZetuPa2u42XRXfQz3lJuBa

Lendo o arquivo com os medicamentos e normalizando
"""

#!pip install metaphone
#!pip install Unidecode

from google.colab import drive
drive.mount('/content/drive/MyDrive/iaMedicamentos/')

import pandas as pd
import re
from unidecode import unidecode

# Função para normalizar nomes dos produtos
def normalize_product_name(name):
    # Removendo acentuação e convertendo para maiúsculas
    normalized_name = unidecode(name.upper())
    # Garantindo que o caractere "+" seja precedido e seguido por um único espaço
    #normalized_name = re.sub(r'\s*\+\s*', ' + ', normalized_name)
    # Garantindo que o caractere "-" seja precedido e seguido por um único espaço
    #normalized_name = re.sub(r'\s*\-\s*', ' - ', normalized_name)
    # Garantindo que o caractere ";" seja precedido e seguido por um único espaço
    #normalized_name = re.sub(r'\s*\;\s*', ' ; ', normalized_name)
    # Garantindo que o caractere "(" seja sempre precedido por um espaço
    #normalized_name = re.sub(r'(?<!\s)\(', ' (', normalized_name)
    # Garantindo que o caractere ")" sempre tenha um espaço após ele
    #normalized_name = re.sub(r'\)(?!\s)', ') ', normalized_name)
    #Garantindo que todo caracter sera substituido por espacp
    normalized_name = re.sub(r'[^a-zA-Z0-9]',' ', normalized_name)
    # Substituindo múltiplos espaços por um único espaço
    normalized_name = re.sub(r'\s+', ' ', normalized_name)
    return normalized_name

# Caminho para o arquivo Excel fornecido
file_path = './listaMedicamentos.xls'

# Lendo o arquivo Excel
df = pd.read_excel(file_path)

# Normalizando a coluna "PRODUTO" e "SUBSTÂNCIA"
df['PRODUTO_NORMALIZADO'] = df['PRODUTO'].apply(normalize_product_name)
df['SUBSTANCIA_NORMALIZADA'] = df['SUBSTÂNCIA'].apply(normalize_product_name)


# Identificar substâncias únicas
substances_uniques = df['SUBSTANCIA_NORMALIZADA'].unique()

#Create empty dataframe
new_rows_list = []

#Tambem queremos considerar generico. Assim, por simplicidade, fazemos uma nova linha para cada substancia
# Criar uma nova linha para cada substância única
for substance in substances_uniques:
    new_line = {"SUBSTÂNCIA": substance, "PRODUTO": substance, "PRODUTO_NORMALIZADO": substance, "SUBSTANCIA_NORMALIZADA": substance}
    new_rows_list.append(new_line)

# Concatenating DataFrames
new_lines_df = pd.DataFrame(new_rows_list)

# Usar pd.concat para adicionar as novas linhas ao DataFrame original
df = pd.concat([df, new_lines_df], ignore_index=True)

# Removendo duplicatas com base na coluna normalizada
unique_products = df['PRODUTO_NORMALIZADO'].drop_duplicates().tolist()

words = unique_products

product_substance_map = df.set_index('PRODUTO_NORMALIZADO')['SUBSTANCIA_NORMALIZADA'].to_dict()

"""Calculando similaridades"""

!pip install Levenshtein
import Levenshtein
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def same_substance(product1, product2, map):
    substance1 = map.get(product1, None)
    substance2 = map.get(product2, None)
    return substance1 == substance2 and substance1 is not None

def similarity(word1, word2):
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

def calculate_similarity_for_list(words):
    similarityDict = {}
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:  # Avoid repeating pairs and comparing words with themselves
                #Eliminate pairs that consists of the same substance (no sound-alike problem)
                if not same_substance(word1, word2, product_substance_map):
                  key = f"{word1}-{word2}"
                  similarityDict[key] = similarity(word1, word2)
    return similarityDict


# Calculate phonetic similarities
similarities = calculate_similarity_for_list(words)

"""Contando a frequência"""

distances = list(similarities.values())

# Calculando a frequência de cada valor de distância no conjunto de dados simulado
distances_frequency = Counter(distances)

# Ordenando os dados para o gráfico
sorted_distances = sorted(distances_frequency.keys())
counts = [distances_frequency[distance] for distance in sorted_distances]

"""Plotando o gráfico"""

# Gerando o gráfico de barras para a PDF discreta
plt.figure(figsize=(10, 6))
plt.bar(sorted_distances, counts, color='skyblue', edgecolor='black')
plt.title('Histograma das Distâncias de Levenshtein entre nomes de Medicamentos')
plt.xlabel('Distância')
plt.ylabel('Quantidade de Pares')
plt.xticks(range(min(sorted_distances), max(sorted_distances) + 1, 5))
plt.grid(axis='y', alpha=0.75)

# Exibindo o gráfico de barras
plt.show()

"""Exibindo apenas as N menores distancias"""

N = 3
# Filtrando os dados para incluir apenas distâncias até x=N
filtered_distances = [distance for distance in sorted_distances if distance <= N]
filtered_count = [counts[i] for i, distance in enumerate(sorted_distances) if distance <= N]

# Gerando o gráfico de barras para a PDF discreta com distâncias até x=10
plt.figure(figsize=(10, 6))
plt.bar(filtered_distances, filtered_count, color='skyblue', edgecolor='black')
plt.title('Histograma das Distâncias de Levenshtein entre nomes de Medicamentos')
plt.xlabel('Distância')
plt.ylabel('Quantidade de Pares')
plt.xticks(filtered_distances)  # Ajustando os xticks para a nova faixa de dados
plt.grid(axis='y', alpha=0.75)

# Exibindo o gráfico de barras
plt.show()
print(filtered_count)

"""Função para imprimir similaridades"""

def printSimilarity(pair,similarity,outputFileName=''):
  word1 = pair.split("-")[0]
  word2 = pair.split("-")[1]
  if (outputFileName != ''):
    outputFile = open(outputFileName, 'a')
  else:
    outputFile = None
  print("********",file=outputFile)
  print("PRODUTO A - Nome: " + str(word1) + " - Substancia: " + str(product_substance_map.get(word1,None)),file=outputFile)
  print("PRODUTO B - Nome: " + str(word2) + " - Substancia: " + str(product_substance_map.get(word2,None)),file=outputFile)
  print("Distancia = " + str(similarity),file=outputFile)
  if (outputFileName != ''):
    outputFile.close()

"""Exibindo os top K similares"""

K = 100

sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=False)

sorted_similarities_n_more_similar = sorted_similarities[:K]

fileName = "/content/drive/MyDrive/iaMedicamentos/top100.txt"

outputFile = open(fileName, 'w')
print("Lista dos " + str(K) + " primeiros similares",file=outputFile)
outputFile.close()

for pair, similarity in sorted_similarities_n_more_similar:
    printSimilarity(pair,similarity,fileName)

"""Exibindo os pares que tem distância D"""

# Definindo o valor D para o qual queremos encontrar pares com distância igual
D = 4  # Exemplo, ajuste conforme necessário

for pair, distance in sorted_similarities:
  if (distance > D):
    break
  word1 = pair.split("-")[0]
  word2 = pair.split("-")[1]
  print(f"Similaridade entre {word1} e {word2} = {distance}")
