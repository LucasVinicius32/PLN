import spacy

# Carregar os modelos de linguagem
nlp_en = spacy.load("en_core_web_sm")  # Modelo de inglês
nlp_pt = spacy.load("pt_core_news_sm")  # Modelo de português

# Texto em português
texto_pt = """
Era uma vez, em uma terra distante, um pequeno vilarejo situado entre colinas ondulantes e florestas exuberantes. O vilarejo era conhecido por seus belos jardins e habitantes amigáveis. Pessoas de cidades vizinhas vinham apenas para ver as flores vibrantes e ouvir o canto melodioso dos pássaros que preenchia o ar. A cada primavera, o vilarejo realizava um grande festival para celebrar a mudança das estações. Era um tempo de alegria, música e espírito comunitário. O festival era o ponto alto do ano, reunindo famílias e amigos de todos os lugares.
"""

# Texto em inglês
texto_en = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.
"""

# Processar os textos
doc_pt = nlp_pt(texto_pt)
doc_en = nlp_en(texto_en)

# Tokenização por sentenças
sentencas_pt = [sent.text for sent in doc_pt.sents]
sentencas_en = [sent.text for sent in doc_en.sents]

print("Tokenização por Sentenças (Português):")
print(sentencas_pt)

print("\nTokenização por Sentenças (Inglês):")
print(sentencas_en)

# Tokenização por palavras
palavras_pt = [token.text for token in doc_pt]
palavras_en = [token.text for token in doc_en]

print("\nTokenização por Palavras (Português):")
print(palavras_pt)

print("\nTokenização por Palavras (Inglês):")
print(palavras_en)

# POS Tagging
print("\nPOS Tags (Português):")
for token in doc_pt:
    print(f"{token.text}: {token.pos_}")

print("\nPOS Tags (Inglês):")
for token in doc_en:
    print(f"{token.text}: {token.pos_}")
