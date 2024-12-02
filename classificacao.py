import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, RSLPStemmer
import string

nltk.download('stopwords')
nltk.download('omw-1.4')  
nltk.download('wordnet') 

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()  # Usando lematizador
    stemmer = RSLPStemmer()  # Usando stemmer para o português
    stop_words = stopwords.words('portuguese')
    
    # Converte para minúsculas
    text = text.lower()
    
    # Remove pontuações
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokeniza o texto
    tokens = text.split()
    
    # Aplica lematização, stemização e remove stop words
    tokens = [
        stemmer.stem(lemmatizer.lemmatize(word))  # Primeiro lematiza, depois faz a stemização
        for word in tokens if word not in stop_words
    ]
    
    return ' '.join(tokens)

# Corpus de revisões
corpus = [
    "Adorei o produto, é excelente!",  # Positiva
    "Detestei, é horrível e não funciona.",  # Negativa
    "Produto mediano, razoável pelo preço.",  # Neutra
    "Ótima qualidade, recomendo muito!",  # Positiva
    "Terrível, não gostei nada.",  # Negativa
    "Cumpre o que promete, nada excepcional.",  # Neutra
    "Fantástico, superou as expectativas!",  # Positiva
    "Péssimo, nunca mais compro dessa marca.",  # Negativa
    "Produto aceitável, mas já vi melhores.",  # Neutra
    "Maravilhoso, estou apaixonado!",  # Positiva
    "Horrível, muito abaixo do esperado.",  # Negativa
    "É um produto ok, sem grandes surpresas.",  # Neutra,
]
labels = ["positiva", "negativa", "neutra", "positiva", "negativa", "neutra",
          "positiva", "negativa", "neutra", "positiva", "negativa", "neutra"]

# Pré-processar o corpus
preprocessed_corpus = [preprocess_text(review) for review in corpus]

# Vetorização do texto
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(preprocessed_corpus)
y = labels

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento dos modelos
# Multilayer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=42, max_iter=1000)
mlp.fit(X_train, y_train)

# K-Nearest Neighbors (KNN) com número de vizinhos ajustado
knn = KNeighborsClassifier(n_neighbors=3)  # Ajustei o número de vizinhos
knn.fit(X_train, y_train)

# Avaliação inicial dos modelos
mlp_predictions = mlp.predict(X_test)
knn_predictions = knn.predict(X_test)

# print("Relatório de Classificação - MLP:\n", classification_report(y_test, mlp_predictions))
# print("Relatório de Classificação - KNN:\n", classification_report(y_test, knn_predictions))

# Salvando os modelos treinados em arquivos pickle
with open("mlp_model.pkl", "wb") as mlp_file:
    pickle.dump(mlp, mlp_file)

with open("knn_model.pkl", "wb") as knn_file:
    pickle.dump(knn, knn_file)

# Recarregando os modelos treinados
with open("mlp_model.pkl", "rb") as mlp_file:
    loaded_mlp = pickle.load(mlp_file)

with open("knn_model.pkl", "rb") as knn_file:
    loaded_knn = pickle.load(knn_file)

# Entrada de uma nova revisão
nova_revisao = input("Digite uma frase para classificar: ")
nova_revisao_preprocessada = preprocess_text(nova_revisao)  # Pré-processa a nova revisão
nova_revisao_vetorizada = vectorizer.transform([nova_revisao_preprocessada])

# Classificando a nova revisão
mlp_result = loaded_mlp.predict(nova_revisao_vetorizada)[0]
knn_result = loaded_knn.predict(nova_revisao_vetorizada)[0]

print(f"\nClassificação MLP: {mlp_result}")
print(f"Classificação KNN: {knn_result}")
