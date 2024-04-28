import spacy
import textblob
import os
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


    enhanced_poem.append(f"(Overall sentiment: {sentiment_label})")
    
    return " ".join(enhanced_poem)

def main_menu():
    print("Welcome to the Emoji Poem Generator!")
    print("Please select an option:")
    print("1. Generate a new poem")
    print("2. Enhance an existing poem with emojis")
    print("3. Load a saved poem")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ")
    return choice

def generate_new_poem():
    print("Please enter your poem:")
    input_poem = input()
    enhanced_poem = emoji_enhance_poem(input_poem)
    print("Transformed Poem:")
    print(enhanced_poem)
    save_poem(enhanced_poem)

def enhance_existing_poem():
    print("Please enter the poem you want to enhance:")
    input_poem = input()
    enhanced_poem = emoji_enhance_poem(input_poem)
    print("Transformed Poem:")
    print(enhanced_poem)
    save_poem(enhanced_poem)

def load_saved_poem():
    if os.path.exists("saved_poems.txt"):
        with open("saved_poems.txt", "r") as file:
            saved_poems = file.readlines()
        for i, poem in enumerate(saved_poems):
            print(f"{i+1}. {poem.strip()}")
        print("Select a poem to load (enter the number):")
        choice = int(input()) - 1
        if 0 <= choice < len(saved_poems):
            return saved_poems[choice].strip()
    print("No saved poems found.")
    return None

def save_poem(poem):
    with open("saved_poems.txt", "a") as file:
        file.write(poem + "\n")
    print("Poem saved successfully.")

if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == "1":
            generate_new_poem()
        elif choice == "2":
            enhance_existing_poem()
        elif choice == "3":
            loaded_poem = load_saved_poem()
            if loaded_poem:
                enhanced_poem = emoji_enhance_poem(loaded_poem)
                print("Transformed Poem:")
                print(enhanced_poem)
                save_poem(enhanced_poem)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
