import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
from collections import Counter
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super(IntentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_with_confidence(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = self.softmax(logits)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item(), probabilities

class SyntheticDataGenerator:
    def __init__(self):
        self.templates = {
            "chit-chat": [
                "Hello there!", "How are you doing?", "What's up?", "Nice weather today",
                "Good morning", "Have a great day", "Thanks a lot", "You're welcome",
                "How's your day going?", "What are you up to?", "Long time no see",
                "Take care", "See you later", "Good to see you"
            ],
            "factual QA": [
                "What is the capital of {}?", "Tell me about {}", "How does {} work?",
                "When was {} invented?", "Who created {}?", "What are the benefits of {}?",
                "Explain {} to me", "What's the difference between {} and {}?",
                "How many {} are there?", "Where can I find {}?"
            ],
            "schedule meeting": [
                "Can we schedule a meeting for {}?", "Let's set up a call",
                "I'd like to book an appointment", "When are you available?",
                "Can we meet tomorrow?", "Schedule a meeting with {}",
                "Book a conference room", "Set up a video call",
                "Arrange a meeting for next week", "Block my calendar for {}"
            ],
            "code generation": [
                "Write a function to {}", "Generate code for {}",
                "Create a {} in Python", "How do I implement {}?",
                "Show me code to {}", "Write a script that {}",
                "Generate a {} algorithm", "Create a class for {}",
                "Build a {} application", "Code a {} function"
            ],
            "database query": [
                "Show me all records where {}", "Select {} from the database",
                "Find users with {}", "Get all {} entries",
                "Query the database for {}", "Retrieve {} data",
                "Search for {} in the table", "List all {} records",
                "Find entries with {} equals {}", "Show {} statistics"
            ]
        }
        
        self.fillers = [
            "Python", "machine learning", "artificial intelligence", "database",
            "algorithm", "software", "technology", "programming", "data science",
            "web development", "mobile app", "cloud computing", "cybersecurity",
            "blockchain", "IoT", "automation", "analytics", "visualization"
        ]
    
    def generate_synthetic_data(self, intent: str, num_samples: int = 50) -> List[str]:
        """Generate synthetic training data for a given intent"""
        if intent not in self.templates:
            logger.warning(f"No templates found for intent: {intent}")
            return []
        
        templates = self.templates[intent]
        synthetic_data = []
        
        for _ in range(num_samples):
            template = random.choice(templates)
            
            # Fill in placeholders with random fillers
            if '{}' in template:
                num_placeholders = template.count('{}')
                fillers = random.sample(self.fillers, min(num_placeholders, len(self.fillers)))
                try:
                    filled_template = template.format(*fillers)
                except:
                    filled_template = template.replace('{}', random.choice(self.fillers))
            else:
                filled_template = template
            
            synthetic_data.append(filled_template)
        
        return synthetic_data
    
    def add_variations(self, text: str) -> List[str]:
        """Add variations to existing text"""
        variations = [text]
        
        # Add question variations
        if not text.endswith('?'):
            variations.append(text + "?")
        
        # Add please variations
        if "please" not in text.lower():
            variations.append(f"Please {text.lower()}")
            variations.append(f"{text} please")
        
        # Add can you variations
        if not text.lower().startswith(("can you", "could you")):
            variations.append(f"Can you {text.lower()}?")
            variations.append(f"Could you {text.lower()}?")
        
        return variations

class IntentDetectionSystem:
    def __init__(self, model_dir: str = "intent_models", data_dir: str = "intent_data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.confidence_threshold = 0.7
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Load existing model if available
        self.load_latest_model()
    
    def save_training_data(self, df: pd.DataFrame, version: str = None):
        """Save training data with versioning"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"training_data_v{version}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Training data saved to {filepath}")
        return filepath
    
    def load_training_data(self, filepath: str = None) -> pd.DataFrame:
        """Load training data"""
        if filepath is None:
            # Load latest training data
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith("training_data_v")]
            if not data_files:
                return pd.DataFrame(columns=['text', 'intent'])
            filepath = os.path.join(self.data_dir, sorted(data_files)[-1])
        
        return pd.read_csv(filepath)
    
    def add_new_intent(self, intent_name: str, examples: List[str] = None, 
                      generate_synthetic: bool = True, num_synthetic: int = 50) -> str:
        """Add a new intent to the system"""
        logger.info(f"Adding new intent: {intent_name}")
        
        # Load existing data
        df = self.load_training_data()
        
        new_data = []
        
        # Add provided examples
        if examples:
            for example in examples:
                new_data.append({'text': example, 'intent': intent_name})
                # Add variations
                variations = self.synthetic_generator.add_variations(example)
                for variation in variations:
                    new_data.append({'text': variation, 'intent': intent_name})
        
        # Generate synthetic data
        if generate_synthetic:
            synthetic_examples = self.synthetic_generator.generate_synthetic_data(
                intent_name, num_synthetic
            )
            for example in synthetic_examples:
                new_data.append({'text': example, 'intent': intent_name})
        
        # Add unknown intent examples (negative samples)
        unknown_examples = [
            "I don't know what I want", "This doesn't make sense", "Random gibberish",
            "asdfghjkl", "completely unrelated query", "nonsensical request",
            "what is this even", "I'm confused", "this is not a real request"
        ]
        for example in unknown_examples:
            new_data.append({'text': example, 'intent': 'unknown'})
        
        # Combine with existing data
        new_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
        
        # Balance dataset
        new_df = self.balance_dataset(new_df)
        
        # Save with version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.save_training_data(new_df, version)
        
        logger.info(f"Added {len(new_data)} examples for intent '{intent_name}'")
        return filepath
    
    def balance_dataset(self, df: pd.DataFrame, target_samples_per_class: int = None) -> pd.DataFrame:
        """Balance the dataset by sampling equal numbers from each class"""
        if target_samples_per_class is None:
            # Use the median count
            counts = df['intent'].value_counts()
            target_samples_per_class = int(counts.median())
        
        balanced_data = []
        for intent in df['intent'].unique():
            intent_data = df[df['intent'] == intent]
            
            if len(intent_data) >= target_samples_per_class:
                # Sample down
                sampled = intent_data.sample(n=target_samples_per_class, random_state=42)
            else:
                # Sample up (with replacement)
                sampled = intent_data.sample(n=target_samples_per_class, replace=True, random_state=42)
            
            balanced_data.append(sampled)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        logger.info(f"Dataset balanced to {target_samples_per_class} samples per class")
        return balanced_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Prepare data for training"""
        # Create label encoders
        unique_intents = sorted(df['intent'].unique())
        self.label_encoder = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.label_decoder = {idx: intent for intent, idx in self.label_encoder.items()}
        
        texts = df['text'].tolist()
        labels = [self.label_encoder[intent] for intent in df['intent']]
        
        return texts, labels
    
    def train_model(self, df: pd.DataFrame, epochs: int = 5, batch_size: int = 16, 
                   learning_rate: float = 2e-5, validation_split: float = 0.2):
        """Train the intent classification model"""
        logger.info("Starting model training...")
        
        # Prepare data
        texts, labels = self.prepare_data(df)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        num_classes = len(self.label_encoder)
        self.model = IntentClassifier(num_classes).to(self.device)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
            
            val_accuracy = correct_predictions / total_predictions
            val_accuracies.append(val_accuracy)
            
            logger.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save model
        self.save_model()
        
        # Generate training report
        self.generate_training_report(train_losses, val_accuracies, val_texts, val_labels)
        
        logger.info("Training completed successfully!")
    
    def save_model(self):
        """Save the trained model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"model_v{timestamp}.pt")
        metadata_path = os.path.join(self.model_dir, f"metadata_v{timestamp}.json")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'num_classes': len(self.label_encoder),
            'confidence_threshold': self.confidence_threshold
        }, model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_path': model_path,
            'num_classes': len(self.label_encoder),
            'intents': list(self.label_encoder.keys()),
            'confidence_threshold': self.confidence_threshold,
            'device': str(self.device)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_latest_model(self):
        """Load the latest trained model"""
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_v") and f.endswith(".pt")]
        
        if not model_files:
            logger.info("No trained model found")
            return False
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(self.model_dir, latest_model)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model
            num_classes = checkpoint['num_classes']
            self.model = IntentClassifier(num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load encoders
            self.label_encoder = checkpoint['label_encoder']
            self.label_decoder = checkpoint['label_decoder']
            self.confidence_threshold = checkpoint.get('confidence_threshold', 0.7)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_intent(self, text: str) -> Dict:
        """Predict intent for a given text"""
        if self.model is None:
            return {'error': 'No trained model available'}
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        predicted_class, confidence, probabilities = self.model.predict_with_confidence(
            input_ids, attention_mask
        )
        
        predicted_intent = self.label_decoder[predicted_class]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            predicted_intent = 'unknown'
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], min(3, len(self.label_decoder)))
        top_predictions = [
            {
                'intent': self.label_decoder[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            'text': text,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'is_unknown': confidence < self.confidence_threshold
        }
    
    def generate_training_report(self, train_losses: List[float], val_accuracies: List[float],
                               val_texts: List[str], val_labels: List[int]):
        """Generate a training report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.model_dir, f"training_report_v{timestamp}.json")
        
        # Generate predictions for validation set
        self.model.eval()
        val_predictions = []
        
        for text in val_texts:
            result = self.predict_intent(text)
            val_predictions.append(result['predicted_intent'])
        
        # Convert back to labels
        val_pred_labels = [self.label_encoder.get(intent, -1) for intent in val_predictions]
        
        # Calculate metrics
        val_true_intents = [self.label_decoder[label] for label in val_labels]
        
        report = {
            'timestamp': timestamp,
            'training_history': {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            },
            'final_metrics': {
                'final_train_loss': train_losses[-1],
                'final_val_accuracy': val_accuracies[-1],
                'max_val_accuracy': max(val_accuracies)
            },
            'model_info': {
                'num_classes': len(self.label_encoder),
                'intents': list(self.label_encoder.keys()),
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for unknown intent detection"""
        self.confidence_threshold = new_threshold
        logger.info(f"Confidence threshold updated to {new_threshold}")

# Example usage and testing
def main():
    # Initialize the system
    intent_system = IntentDetectionSystem()
    
    # Add initial intents with examples
    initial_intents = {
        "chit-chat": [
            "Hello", "Hi there", "How are you?", "Good morning",
            "Thanks", "You're welcome", "Have a great day"
        ],
        "factual QA": [
            "What is Python?", "Explain machine learning", "How does AI work?",
            "Tell me about databases", "What's the weather like?"
        ],
        "schedule meeting": [
            "Schedule a meeting", "Book an appointment", "Set up a call",
            "Can we meet tomorrow?", "Arrange a conference"
        ],
        "code generation": [
            "Write a Python function", "Generate code for sorting",
            "Create a web scraper", "Build a calculator", "Code a login system"
        ],
        "database query": [
            "Show all users", "Find records with status active",
            "Get customer data", "List all products", "Query the sales table"
        ]
    }
    
    # Add all intents
    for intent_name, examples in initial_intents.items():
        intent_system.add_new_intent(intent_name, examples)
    
    # Load and train
    df = intent_system.load_training_data()
    print(f"Training data shape: {df.shape}")
    print(f"Intent distribution:\n{df['intent'].value_counts()}")
    
    # Train the model
    intent_system.train_model(df, epochs=3)
    
    # Test predictions
    test_queries = [
        "Hello there!",
        "What is machine learning?",
        "Can we schedule a meeting for tomorrow?",
        "Write a function to sort a list",
        "Show me all customer records",
        "This is completely random text that doesn't fit anywhere"
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    for query in test_queries:
        result = intent_system.predict_intent(query)
        print(f"\nQuery: {query}")
        print(f"Predicted Intent: {result['predicted_intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Is Unknown: {result['is_unknown']}")
        print(f"Top 3 Predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['intent']}: {pred['confidence']:.3f}")

if __name__ == "__main__":
    main()