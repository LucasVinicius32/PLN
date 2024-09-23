from flask import Flask, request, jsonify, render_template
import spacy

# Carregar o modelo de linguagem em português
nlp = spacy.load("pt_core_news_sm")

app = Flask(__name__)

def tag_sentences(text):
    # Processar o texto usando o spaCy
    doc = nlp(text)
    
    tagged_sentences = []
    
    for sent in doc.sents:
        tagged = [(token.text, token.pos_) for token in sent]
        tagged_sentences.append(tagged)
    
    return tagged_sentences

def convert_pos_tags(tagged_sentences):
    # Dicionário para traduzir os rótulos do spaCy
    pos_translation = {
        "PRON": "pronome",
        "VERB": "verbo",
        "AUX": "verbo auxiliar",
        "ADJ": "adjetivo",
        "NOUN": "substantivo",
        "ADV": "advérbio",
        "DET": "determinante",
        "ADP": "preposição",
        "PROPN": "nome próprio",
        "NUM": "número",
        "PART": "partícula",
        "CCONJ": "conjunção coordenativa",
        "SCONJ": "conjunção subordinativa",
        "INTJ": "interjeição",
        "SYM": "símbolo",
        "X": "outro"
    }

    results = []
    for tagged in tagged_sentences:
        for word, tag in tagged:
            results.append({word: pos_translation.get(tag, tag)})
    
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['text']
    tagged_sentences = tag_sentences(input_text)
    results = convert_pos_tags(tagged_sentences)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
