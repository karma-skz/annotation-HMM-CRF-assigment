import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
from typing import List, Dict, Tuple, Any


class CRFTagger:
    """
    Conditional Random Fields implementation for sequence tagging tasks.
    """
    
    def __init__(self, algorithm: str = "lbfgs", c1: float = 0.1, c2: float = 0.1, max_iterations: int = 100):
        """
        Initialize the CRF tagger with specified parameters.
        
        Args:
            algorithm: The algorithm for optimization
            c1: L1 regularization parameter
            c2: L2 regularization parameter
            max_iterations: Maximum number of iterations
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self.tag_counter = Counter()

    def _word_features(self, word: str) -> Dict[str, Any]:
        """Extract features from the word itself."""
        return {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "word.contains_hyphen": "-" in word,
            "word.contains_digit": any(char.isdigit() for char in word),
        }
    
    def _get_affixes(self, word: str) -> Dict[str, str]:
        """Extract prefix and suffix features."""
        features = {}
        for i in range(1, 5):
            if len(word) >= i:
                features[f"prefix-{i}"] = word[:i]
                features[f"suffix-{i}"] = word[-i:]
        return features
    
    def _contextual_features(self, sentence: List[str], index: int) -> Dict[str, Any]:
        """Extract features from surrounding words."""
        features = {}
        
        # Previous word features
        if index > 0:
            prev_word = sentence[index - 1]
            features.update({
                "prev_word.lower()": prev_word.lower(),
                "prev_word.istitle()": prev_word.istitle(),
                "prev_word[-3:]": prev_word[-3:],
            })
        else:
            features["BOS"] = True  # Beginning of sequence
        
        # Next word features
        if index < len(sentence) - 1:
            next_word = sentence[index + 1]
            features.update({
                "next_word.lower()": next_word.lower(),
                "next_word.istitle()": next_word.istitle(),
                "next_word[-3:]": next_word[-3:],
            })
        else:
            features["EOS"] = True  # End of sequence
            
        return features

    def extract_features(self, sentence: List[str], index: int) -> Dict[str, Any]:
        """
        Extract features for a word in context.
        
        Args:
            sentence: List of words in the sentence
            index: Index of the current word
            
        Returns:
            Dictionary of features
        """
        word = sentence[index]
        
        # Combine all feature types
        features = {}
        features.update(self._word_features(word))
        features.update(self._get_affixes(word))
        features.update(self._contextual_features(sentence, index))
        
        return features

    def prepare_data(self, tagged_sentences: List[List[Tuple[str, str]]]) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
        """
        Prepare training data from tagged sentences.
        
        Args:
            tagged_sentences: List of sentences with (word, tag) tuples
            
        Returns:
            X: List of feature dictionaries
            y: List of tags
        """
        X = []
        y = []

        for sentence in tagged_sentences:
            words = [word for word, _ in sentence]
            tags = [tag for _, tag in sentence]
            
            # Update tag statistics
            self.tag_counter.update(tags)
            
            # Extract features for each word
            sentence_features = [self.extract_features(words, i) for i in range(len(words))]
            
            X.append(sentence_features)
            y.append(tags)

        return X, y

    def train(self, tagged_sentences: List[List[Tuple[str, str]]]) -> None:
        """
        Train the CRF model on tagged sentences.
        
        Args:
            tagged_sentences: List of sentences with (word, tag) tuples
        """
        X, y = self.prepare_data(tagged_sentences)
        self.model.fit(X, y)

    def predict(self, sentence: List[str]) -> List[str]:
        """
        Predict tags for a new sentence.
        
        Args:
            sentence: List of words
            
        Returns:
            List of predicted tags
        """
        features = [self.extract_features(sentence, i) for i in range(len(sentence))]
        return self.model.predict([features])[0]
