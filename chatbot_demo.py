import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from transformers import pipeline

nltk.download('punkt', quiet=True)  # Download punkt tokenizer data if you haven't yet.

stemmer = PorterStemmer()

def stem_words(words):
  return [stemmer.stem(word) for word in words]


def process_input(user_input):
    """Processes user input by tokenizing and stemming."""
    tokens = word_tokenize(user_input.lower())
    return stem_words(tokens)


# Load a pre-trained conversational pipeline
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

def get_response(user_input):
    """Generates a conversational response using a pre-trained model and fallback."""
    try:
      response = chatbot(user_input)
      return response[0]['generated_text']
    except:
      processed_input = process_input(user_input)
      if any(keyword in processed_input for keyword in ["hello", "hi", "hey"]):
           return "Hello there! How are you feeling today?"
      elif any(keyword in processed_input for keyword in ["anxious", "nervous", "stressed"]):
          return "I'm sorry to hear that. It sounds like you might be feeling some stress. Would you like to try a calming technique?"
      elif any(keyword in processed_input for keyword in ["sad", "unhappy", "depressed"]):
          return "I'm sorry you're feeling sad. Would you like me to share some resources for support?"
      elif any(keyword in processed_input for keyword in ["calm", "relax", "breathe"]):
          return "Okay, let's try some deep breathing. Inhale deeply, hold for a few seconds, and exhale slowly. Repeat 5 times. "
      elif any(keyword in processed_input for keyword in ["help", "resource", "support"]):
          return "Okay, here are some links you may find helpful: [Resource 1], [Resource 2], or [Resource 3]. You can also try contacting a professional."
      elif any(keyword in processed_input for keyword in ["thank", "thanks"]):
          return "You're welcome. Is there anything else I can help you with today?"
      elif any(keyword in processed_input for keyword in ["bye", "goodbye"]):
          return "Goodbye! Take care and I am here if you need me again."
      else:
        return "I'm not sure I understand. How else can I help you?"


def main():
    print("Hi! I'm your mental health support bot. I'm here to help, how can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()