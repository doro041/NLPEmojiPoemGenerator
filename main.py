from flask import Flask, render_template, request
import spacy
from textblob import TextBlob
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load T5 model and tokenizer
emoji_model_path = "KomeijiForce/t5-base-emojilm"
emoji_tokenizer = T5Tokenizer.from_pretrained(emoji_model_path)
emoji_generator = T5ForConditionalGeneration.from_pretrained(emoji_model_path)

def generate_emojis(text):
    
    inputs = emoji_tokenizer.encode("translate into emojis: " + text, return_tensors="pt", padding=True, truncation=True)
    outputs = emoji_generator.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    return emoji_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")

def emoji_enhance_poem(poem):
    doc = nlp(poem)
    enhanced_poem = []

    sentiment = TextBlob(poem).sentiment.polarity
    if sentiment > 0.1:
        sentiment_label = "positive"
    elif sentiment < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Generate emojis for each sentence
    for sentence in doc.sents:
        emojis = generate_emojis(sentence.text)
        enhanced_poem.append(sentence.text + " " + emojis)

    enhanced_poem.append(f"(Overall sentiment: {sentiment_label})")
    return " ".join(enhanced_poem)

@app.route('/', methods=['GET', 'POST'])
def home():
    enhanced_poem = ""
    if request.method == 'POST':
        poem = request.form.get('poem', '')
        if poem.strip(): 
            enhanced_poem = emoji_enhance_poem(poem)
  
    return render_template('index.html', enhanced_poem=enhanced_poem)

if __name__ == "__main__":
    app.run(debug=True)
