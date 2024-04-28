import spacy
import textblob
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Emoji mappings based on parts of speech or sentiment
emoji_map = {
    "NOUN": {"sun": "🌞", "moon": "🌜", "star": "⭐", "heart": "❤️"},
    "ADJ": {"beautiful": "🌟", "sad": "😢", "happy": "😊", "dark": "🌑"},
    "SENTIMENT": {
        "positive": "😊",
        "negative": "😢",
        "neutral": "😐"
    }
}


def emoji_enhance_poem(poem):
    doc = nlp(poem)
    enhanced_poem = []
    
    # Analyze sentiment using TextBlob
    sentiment = textblob.TextBlob(poem).sentiment.polarity
    if sentiment > 0.1:  # threshold can be adjusted
        sentiment_emoji = emoji_map["SENTIMENT"]["positive"]
    elif sentiment < -0.1:
        sentiment_emoji = emoji_map["SENTIMENT"]["negative"]
    else:
        sentiment_emoji = emoji_map["SENTIMENT"]["neutral"]
    
    # Iterate through tokens and replace with emojis when appropriate
    for token in doc:
        emoji = emoji_map.get(token.pos_, {}).get(token.text.lower(), token.text)
        enhanced_poem.append(emoji)
    
    # Append sentiment emoji at the end of the poem
    enhanced_poem.append(sentiment_emoji)
    
    return " ".join(enhanced_poem)

# Example poem
poem = "The dark sky was filled with stars, shining bright above the calm sea."
print(emoji_enhance_poem(poem))

