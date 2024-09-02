import spacy

# Carregar o modelo de linguagem em inglês do spaCy
nlp = spacy.load("en_core_web_sm")

# Texto em inglês
texto_en = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.
"""

# Processar o texto
doc = nlp(texto_en)

# Tokenização por sentenças
sentencas_en = [sent.text for sent in doc.sents]
print("Tokenização por Sentenças (Inglês):")
print(sentencas_en)

# Tokenização por palavras
palavras_en = [token.text for token in doc]
print("\nTokenização por Palavras (Inglês):")
print(palavras_en)
