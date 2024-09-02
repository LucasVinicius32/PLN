import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Baixe os pacotes necessários
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Exemplo de texto em inglês
texto_ingles = """
Once upon a time, in a land far, far away, there was a small village nestled between rolling hills and lush forests. The village was known for its beautiful gardens and friendly inhabitants. People from neighboring towns would visit just to see the vibrant flowers and hear the melodious birdsong that filled the air. Every spring, the village held a grand festival to celebrate the changing of the seasons. It was a time of joy, music, and community spirit. The festival was the highlight of the year, bringing together families and friends from all around.
"""

# Tokenização por sentenças
sentencas_ingles = sent_tokenize(texto_ingles, language='english')

# Tokenização por palavras
palavras_ingles = word_tokenize(texto_ingles, language='english')

print("Tokenização por Sentenças (Inglês):")
print(sentencas_ingles[:3])  # Exibe as primeiras 3 sentenças tokenizadas

print("\nTokenização por Palavras (Inglês):")
print(palavras_ingles[:20])  # Exibe as primeiras 20 palavras tokenizadas

# POS Tagging para o inglês
tagged_sentences_ingles = nltk.pos_tag(palavras_ingles)
print("\nPOS Tags de exemplo (Inglês):")
for word, tag in tagged_sentences_ingles[:10]:  # Exibe as primeiras 10 palavras com suas tags
    print(f"{word}: {tag}")

# Exemplos de POS tags em Inglês
# Tags como: 'NN' (substantivo), 'JJ' (adjetivo), 'VB' (verbo), 'RB' (advérbio), etc.
