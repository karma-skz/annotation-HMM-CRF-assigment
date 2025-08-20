import math
import pickle
from collections import defaultdict, Counter


def default_dict_factory(factory_type):
    """Create a nested defaultdict with the specified factory type."""
    return lambda: defaultdict(factory_type)


class HMMTagger:
    """Hidden Markov Model implementation for part-of-speech tagging."""
    
    def __init__(self, smoothing_alpha=0.5):
        """
        Initialize the HMM tagger with empty probability distributions.
        
        Args:
            smoothing_alpha (float): Laplace smoothing parameter
        """
        self.transition_probs = defaultdict(default_dict_factory(float))
        self.emission_probs = defaultdict(default_dict_factory(float))
        self.initial_probs = defaultdict(float)
        self.all_tags = set()
        self.tag_counts = defaultdict(int)
        self.vocabulary = set()
        self.alpha = smoothing_alpha
        
        # Counts used during training
        self._transition_counts = defaultdict(default_dict_factory(int))
        self._emission_counts = defaultdict(default_dict_factory(int))
        self._initial_counts = defaultdict(int)
        self._total_sentences = 0
        
        # For log probabilities
        self._use_log_probabilities = True

    def _collect_counts(self, tagged_sentences):
        """
        Collect counts from the tagged sentences for training.
        
        Args:
            tagged_sentences: List of sentences with (word, tag) tuples
        """
        for sentence in tagged_sentences:
            if not sentence:
                continue
                
            self._total_sentences += 1

            # Count initial tag
            first_tag = sentence[0][1]
            self._initial_counts[first_tag] += 1

            # Count emissions and tags
            for word, tag in sentence:
                self.vocabulary.add(word)
                self.all_tags.add(tag)
                self._emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1

            # Count transitions
            for i in range(len(sentence) - 1):
                current_tag = sentence[i][1]
                next_tag = sentence[i + 1][1]
                self._transition_counts[current_tag][next_tag] += 1

    def _calculate_probabilities(self):
        """Calculate transition, emission and initial probabilities from counts."""
        # Calculate initial probabilities
        for tag, count in self._initial_counts.items():
            prob = count / self._total_sentences
            self.initial_probs[tag] = math.log(prob) if self._use_log_probabilities else prob

        # Calculate emission probabilities with smoothing
        for tag in self.all_tags:
            tag_count = self.tag_counts[tag]
            vocab_size = len(self.vocabulary)
            
            for word in self.vocabulary:
                count = self._emission_counts[tag].get(word, 0)
                prob = (count + self.alpha) / (tag_count + self.alpha * vocab_size)
                self.emission_probs[tag][word] = math.log(prob) if self._use_log_probabilities else prob

        # Calculate transition probabilities with smoothing
        for tag1 in self.all_tags:
            tag1_count = sum(self._transition_counts[tag1].values())
            num_tags = len(self.all_tags)
            
            for tag2 in self.all_tags:
                count = self._transition_counts[tag1].get(tag2, 0)
                prob = (count + self.alpha) / (tag1_count + self.alpha * num_tags)
                self.transition_probs[tag1][tag2] = math.log(prob) if self._use_log_probabilities else prob

    def train(self, tagged_sentences):
        """
        Train the HMM tagger on the given tagged sentences.
        
        Args:
            tagged_sentences: List of sentences with (word, tag) tuples
        """
        self._collect_counts(tagged_sentences)
        self._calculate_probabilities()

    def predict(self, sentence):
        """
        Predict tags for a given sentence using the Viterbi algorithm.
        
        Args:
            sentence: List of words to tag
            
        Returns:
            List of predicted tags
        """
        return self.viterbi_algorithm(sentence)

    def get_emission_prob(self, tag, word):
        """Get emission probability, handling unknown words."""
        if word in self.vocabulary:
            return self.emission_probs[tag].get(word, self._get_unknown_word_prob(tag, word))
        else:
            return self._get_unknown_word_prob(tag, word)
            
    def _get_unknown_word_prob(self, tag, word):
        """Calculate probability for unknown words."""
        # Incorporate suffix-based probabilities
        suffixes = ["ing", "ed", "ly", "s", "es"]
        for suffix in suffixes:
            if word.endswith(suffix):
                prob = self.alpha / (self.tag_counts[tag] + self.alpha * len(self.vocabulary))
                return math.log(prob) if self._use_log_probabilities else prob
        # Default smoothing
        prob = self.alpha / (self.tag_counts[tag] + self.alpha * len(self.vocabulary))
        return math.log(prob) if self._use_log_probabilities else prob

    def viterbi_algorithm(self, sentence):
        """
        Implement the Viterbi algorithm for sequence decoding.
        
        Args:
            sentence: List of words to tag
            
        Returns:
            List of predicted tags
        """
        if not sentence:
            return []

        n = len(sentence)
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # Initialize first step
        for tag in self.all_tags:
            word_prob = self.get_emission_prob(tag, sentence[0])
            init_prob = self.initial_probs.get(tag, float('-inf') if self._use_log_probabilities else 0)
            
            if self._use_log_probabilities:
                viterbi[0][tag] = init_prob + word_prob
            else:
                viterbi[0][tag] = init_prob * word_prob
                
            backpointer[0][tag] = None

        # Forward pass
        for t in range(1, n):
            for tag in self.all_tags:
                max_prob = float('-inf') if self._use_log_probabilities else 0
                best_prev_tag = None
                word_prob = self.get_emission_prob(tag, sentence[t])

                for prev_tag in self.all_tags:
                    if prev_tag not in viterbi[t-1]:
                        continue
                        
                    trans_prob = self.transition_probs[prev_tag].get(tag, 
                                float('-inf') if self._use_log_probabilities else 0)
                    
                    if self._use_log_probabilities:
                        prob = viterbi[t-1][prev_tag] + trans_prob + word_prob
                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag = prev_tag
                    else:
                        prob = viterbi[t-1][prev_tag] * trans_prob * word_prob
                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_prob
                    backpointer[t][tag] = best_prev_tag

        # Backward pass to find best path
        best_path = self._backtrack_best_path(viterbi, backpointer, n)
        return best_path
        
    def _backtrack_best_path(self, viterbi, backpointer, n):
        """Backtrack to find the best tag sequence."""
        if not self.all_tags:
            return []
            
        # Find best final tag
        best_last_tag = None
        max_prob = float('-inf') if self._use_log_probabilities else 0
        
        for tag, prob in viterbi[n-1].items():
            if prob > max_prob:
                max_prob = prob
                best_last_tag = tag

        # Handle case where no valid path exists
        if best_last_tag is None:
            best_last_tag = next(iter(self.all_tags))
        
        # Backtrack
        best_path = [best_last_tag]
        for t in range(n-1, 0, -1):
            prev_tag = backpointer[t][best_last_tag]
            if prev_tag is None:
                # Fall back if path is broken
                prev_tag = best_path[0]
            best_path.insert(0, prev_tag)

        return best_path
        
    def save(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Helper functions for data loading
def read_conllu_file(filepath):
    """Read and parse a CoNLL-U format file."""
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip() 
            if not line or line.startswith("#"):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue  
                
            fields = line.split("\t")
            if len(fields) >= 4 and "-" not in fields[0] and "." not in fields[0]:
                word = fields[1]
                tag = fields[3]  
                current_sentence.append((word, tag))
                
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def read_custom_tags_file(filepath):
    """Read custom format tag files."""
    sentences = []
    current_sentence = []
    
    sentence_ending_tags = {"PUNC", ".", "?", "!"}
    sentence_ending_symbols = {".", "?", "!"}

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                word, tag = parts
                current_sentence.append((word, tag))
                
                if tag in sentence_ending_tags or any(
                    word.endswith(symbol) for symbol in sentence_ending_symbols
                ):
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def map_ud_to_bis(tagged_sentences, mapping=None):
    """Map Universal Dependencies tags to BIS tagset."""
    if mapping is None:
        mapping = {
            "ADJ": "JJ",
            "ADP": "PSP",
            "ADV": "RB",
            "AUX": "VAUX",
            "CCONJ": "CC",
            "DET": "DT",
            "INTJ": "INJ",
            "NOUN": "NN",
            "NUM": "QC",
            "PART": "RP",
            "PRON": "PRP",
            "PROPN": "NNP",
            "PUNCT": "PUNC",
            "SCONJ": "CC",
            "SYM": "SYM",
            "VERB": "VM",
            "X": "UNK",
        }

    mapped_sentences = []
    for sentence in tagged_sentences:
        mapped_sentence = []
        for word, tag in sentence:
            mapped_tag = mapping.get(tag, tag)
            mapped_sentence.append((word, mapped_tag))
        mapped_sentences.append(mapped_sentence)

    return mapped_sentences
