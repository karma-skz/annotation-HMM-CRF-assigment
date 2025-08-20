#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict, Counter
from hmm import HMMTagger, read_conllu_file, read_custom_tags_file, map_ud_to_bis
from crf import CRFTagger

class POSTagger:
    def __init__(self, config):
        self.config = config
        self.model_dir = config["MODEL_DIR"]
        self.output_dir = config["OUTPUT_DIR"]
        self.hmm_model = HMMTagger()
        self.crf_model = CRFTagger()
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tokenize_sentence(self, sentence):
        import re
        return re.findall(r"\b\w+\b|[^\w\s]", sentence)

    def prepare_training_data(self, sentences, split_ratio=0.8):
        random.shuffle(sentences)
        split_idx = int(len(sentences) * split_ratio)
        return sentences[:split_idx], sentences[split_idx:]

    def extract_words_tags(self, tagged_sentences):
        words, tags = [], []
        for sentence in tagged_sentences:
            sent_words, sent_tags = zip(*sentence)
            words.append(list(sent_words))
            tags.append(list(sent_tags))
        return words, tags

    def evaluate_model(self, true_tags, pred_tags, all_tags):
        true_flat = [tag for sent in true_tags for tag in sent]
        pred_flat = [tag for sent in pred_tags for tag in sent]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average="weighted", zero_division=0
        )
        accuracy = sum(t == p for t, p in zip(true_flat, pred_flat)) / len(true_flat)
        cm = confusion_matrix(true_flat, pred_flat, labels=all_tags)

        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "confusion_matrix": cm}

    def plot_confusion_matrix(self, cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
        if normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
            xticklabels=classes, yticklabels=classes
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title.replace(' ', '_').lower()}.png"))
        print(f"Saved {title}")

    def plot_tag_distribution(self, tagged_sentences, title="Tag Distribution"):
        all_tags = [tag for sentence in tagged_sentences for _, tag in sentence]
        if not all_tags:
            print(f"No tags found in dataset for {title}")
            return
        tag_counts = Counter(all_tags)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        tags, counts = zip(*sorted_tags)
        plt.figure(figsize=(12, 6))
        plt.bar(tags, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.xlabel("POS Tags")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title.replace(' ', '_').lower()}.png"))
        print(f"Saved {title}")

    def train_models(self):
        hindi_data = read_conllu_file(self.config["HINDI_TRAIN_FILE"])
        eng_data = read_conllu_file(self.config["ENG_TRAIN_FILE"])
        combined_data = map_ud_to_bis(hindi_data + eng_data)
        self.plot_tag_distribution(combined_data, title="POS Tag Distribution - Training Data")
        train_data, _ = self.prepare_training_data(combined_data, 0.9)
        self.hmm_model.train(train_data)
        self.crf_model.train(train_data)
        self._save_models()
        print("Training complete. Models saved.")
        
    def _save_models(self):
        """Save trained models to disk."""
        try:
            hmm_path = os.path.join(self.model_dir, "hmm_model.pkl")
            crf_path = os.path.join(self.model_dir, "crf_model.pkl")
            
            # Handle saving HMM model - extract and save picklable components
            hmm_data = {
                'transition_probs': dict(self.hmm_model.transition_probs),
                'emission_probs': dict(self.hmm_model.emission_probs),
                'initial_probs': dict(self.hmm_model.initial_probs),  # Changed from start_probs to initial_probs
                'all_tags': self.hmm_model.all_tags,
                'all_words': self.hmm_model.vocabulary  # Changed from all_words to vocabulary
            }
            
            with open(hmm_path, "wb") as f:
                pickle.dump(hmm_data, f)
            
            with open(crf_path, "wb") as f:
                pickle.dump(self.crf_model, f)
                
            print("Models saved successfully")
                
        except Exception as e:
            print(f"Error saving models: {e}")
            raise

    def _load_models(self):
        """Load trained models from disk."""
        try:
            hmm_path = os.path.join(self.model_dir, "hmm_model.pkl")
            crf_path = os.path.join(self.model_dir, "crf_model.pkl")
            
            if not os.path.exists(hmm_path) or not os.path.exists(crf_path):
                raise FileNotFoundError("Model files not found. Train models first.")
            
            # Load HMM data and reconstruct model    
            with open(hmm_path, "rb") as f:
                hmm_data = pickle.load(f)
                
                # Recreate defaultdict structures with proper factory functions
                self.hmm_model.transition_probs = defaultdict(lambda: defaultdict(float))
                self.hmm_model.emission_probs = defaultdict(lambda: defaultdict(float))
                self.hmm_model.initial_probs = defaultdict(float)  # Changed from start_probs to initial_probs
                
                # Populate with saved data
                for tag1, transitions in hmm_data['transition_probs'].items():
                    for tag2, prob in transitions.items():
                        self.hmm_model.transition_probs[tag1][tag2] = prob
                        
                for tag, emissions in hmm_data['emission_probs'].items():
                    for word, prob in emissions.items():
                        self.hmm_model.emission_probs[tag][word] = prob
                        
                for tag, prob in hmm_data['initial_probs'].items():  # Changed from start_probs to initial_probs
                    self.hmm_model.initial_probs[tag] = prob
                    
                self.hmm_model.all_tags = hmm_data['all_tags']
                self.hmm_model.vocabulary = hmm_data['all_words']  # Changed to match the correct attribute name
                
            with open(crf_path, "rb") as f:
                self.crf_model = pickle.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def _evaluate_language(self, lang, test_file, all_tags):
        """Evaluate models on a specific language test set."""
        test_sentences = read_custom_tags_file(test_file)
        self.plot_tag_distribution(test_sentences, title=f"{lang} Test Data - POS Tag Distribution")
        
        words, true_tags = self.extract_words_tags(test_sentences)
        
        # Run predictions
        hmm_pred = [self.hmm_model.predict(sentence) for sentence in words]
        crf_pred = [self.crf_model.predict(sentence) for sentence in words]
        
        # Evaluate and plot results
        hmm_results = self.evaluate_model(true_tags, hmm_pred, all_tags)
        crf_results = self.evaluate_model(true_tags, crf_pred, all_tags)
        
        self.plot_confusion_matrix(
            hmm_results["confusion_matrix"], 
            all_tags, 
            normalize=True, 
            title=f"HMM Confusion Matrix ({lang})"
        )
        
        self.plot_confusion_matrix(
            crf_results["confusion_matrix"], 
            all_tags, 
            normalize=True, 
            title=f"CRF Confusion Matrix ({lang})"
        )
        
        # Print results
        print(f"{lang} HMM - Precision: {hmm_results['precision']:.4f}, "
              f"Recall: {hmm_results['recall']:.4f}, "
              f"F1: {hmm_results['f1']:.4f}, "
              f"Accuracy: {hmm_results['accuracy']:.4f}")
              
        print(f"{lang} CRF - Precision: {crf_results['precision']:.4f}, "
              f"Recall: {crf_results['recall']:.4f}, "
              f"F1: {crf_results['f1']:.4f}, "
              f"Accuracy: {crf_results['accuracy']:.4f}")
        
        return hmm_results, crf_results

    def test_models(self):
        if not self._load_models():
            print("Failed to load models. Exiting test.")
            return
            
        all_tags = list(self.hmm_model.all_tags)
        
        languages = [
            ("English", self.config["ENG_TEST_FILE"]), 
            ("Hindi", self.config["HIN_TEST_FILE"])
        ]
        
        results = {}
        for lang, test_file in languages:
            print(f"\nEvaluating {lang} test data...")
            hmm_results, crf_results = self._evaluate_language(lang, test_file, all_tags)
            results[lang] = {"hmm": hmm_results, "crf": crf_results}
            
        return results

if __name__ == "__main__":
    tagger = POSTagger({
        "TRAIN_MODELS": True,
        "TEST_MODELS": True,
        "HINDI_TRAIN_FILE": "tagged_hi.conllu",
        "ENG_TRAIN_FILE": "tagged_eng.conllu",
        "HIN_TEST_FILE": "my annotations/annotated_hindi.txt",
        "ENG_TEST_FILE": "my annotations/annotated_eng.txt",
        "MODEL_DIR": "models",
        "OUTPUT_DIR": "results",
    })
    if tagger.config["TRAIN_MODELS"]:
        tagger.train_models()
    if tagger.config["TEST_MODELS"]:
        tagger.test_models()
