import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

#  Corpus de revisões 
corpus = [
    "Adorei o produto, é excelente!",  # Positiva
    "Detestei, é horrível e não funciona.",  # Negativa
    "Produto mediano, razoavel pelo preço.",  # Neutra
    "Otima qualidade, recomendo muito!",  # Positiva
    "Terrível, não gostei nada.",  # Negativa
    "Cumpre o que promete, nada excepcional.",  # Neutra
    "Fantastico, superou as expectativas!",  # Positiva
    "Pessimo, nunca mais compro dessa marca.",  # Negativa
    "Produto aceitavel, mas já vi melhores.",  # Neutra
    "Maravilhoso, estou apaixonado!",  # Positiva
    "Horrivel, muito abaixo do esperado.",  # Negativa
    "É um produto ok, sem grandes surpresas.",  # Neutra
]
labels = ["positiva", "negativa", "neutra", "positiva", "negativa", "neutra",
          "positiva", "negativa", "neutra", "positiva", "negativa", "neutra"]

# Vetorização do texto
stop_words = stopwords.words('portuguese')
vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)
y = labels

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Treinamento dos modelos
# Multilayer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=42, max_iter=1000)
mlp.fit(X_train, y_train)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
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

# Exibindo somente 5 itens das informações dos modelos carregados
# print("\nPrimeiros 5 coeficientes do modelo MLP carregado:")
# # Exibe os primeiros 5 
# print(loaded_mlp.coefs_[0][:5])
# print("\nPrimeiras 5 amostras do modelo KNN carregado:")
# # Exibe as 5 primeiras amostras de treinamento no modelo KNN
# print(loaded_knn._fit_X[:5])

# Entrada do usuário para classificação
nova_revisao = input("Digite uma revisão para classificar: ")
nova_revisao_vetorizada = vectorizer.transform([nova_revisao])

# Classificando a nova revisão
mlp_result = loaded_mlp.predict(nova_revisao_vetorizada)[0]
knn_result = loaded_knn.predict(nova_revisao_vetorizada)[0]

print(f"\nClassificação MLP: {mlp_result}")
print(f"Classificação KNN: {knn_result}")





