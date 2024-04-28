import spacy
import textblob
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load T5 model and tokenizer
emoji_model_path = "KomeijiForce/t5-base-emojilm"
emoji_tokenizer = T5Tokenizer.from_pretrained(emoji_model_path)
emoji_generator = T5ForConditionalGeneration.from_pretrained(emoji_model_path)

def generate_emojis(text):
    inputs = emoji_tokenizer("translate into emojis: " + text, return_tensors="pt", padding=True, truncation=True)
    outputs = emoji_generator.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
    return emoji_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")

def emoji_enhance_poem(poem):
    doc = nlp(poem)
    enhanced_poem = []

    # Analyze sentiment using TextBlob (optional, if you want to include this information)
    sentiment = textblob.TextBlob(poem).sentiment.polarity
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

    # Optionally append overall sentiment at the end (adjust according to your preference)
    enhanced_poem.append(f"(Overall sentiment: {sentiment_label})")
    
    return " ".join(enhanced_poem)

if __name__ == "__main__":
    print("Please enter your poem:")
    input_poem = input()
    print("Transformed Poem:")
    print(emoji_enhance_poem(input_poem))
