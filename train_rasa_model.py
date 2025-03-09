import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.needs_assessment.intent_recognition import IntentRecognizer
from config import RASA_CONFIG_PATH, INTENT_MODEL_PATH

# Create an instance of the IntentRecognizer
intent_recognizer = IntentRecognizer(config_path=RASA_CONFIG_PATH)

# Train the model
training_data_path = 'data/intent_data/nlu.yml'
success = intent_recognizer.train_model(training_data_path, output_path=INTENT_MODEL_PATH)

if success:
    print("Rasa NLU model trained successfully!")
else:
    print("Rasa NLU model training failed.")