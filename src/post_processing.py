import re
import string
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
import difflib


class LanguageModel:
    """
    N-gram Language Model để đánh giá độ hợp lý của câu
    """

    def __init__(self, n: int = 2):
        self.n = n
        self.ngrams = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train_from_text(self, text: str):
        """Train từ corpus text"""
        words = text.lower().split()
        self.vocab.update(words)

        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n])
            context = ngram[:-1]
            self.ngrams[ngram] += 1
            self.context_counts[context] += 1

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Tính xác suất P(word|context) với Laplace smoothing"""
        ngram = context + (word.lower(),)
        ngram_count = self.ngrams.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        # Laplace smoothing
        vocab_size = len(self.vocab) if self.vocab else 1
        return (ngram_count + 1) / (context_count + vocab_size)

    def sentence_probability(self, words: List[str]) -> float:
        """Tính xác suất của cả câu"""
        if len(words) < self.n:
            return 1.0

        log_prob = 0.0
        for i in range(self.n - 1, len(words)):
            context = tuple(words[i - self.n + 1:i])
            word = words[i]
            prob = self.probability(word, context)
            log_prob += np.log(prob + 1e-10)

        return np.exp(log_prob / len(words))


class SpellCorrector:
    """
    Statistical Spell Checker sử dụng:
    - Edit distance (Levenshtein)
    - Character confusion matrix (OCR specific errors)
    - Word frequency
    """

    def __init__(self, vocab: set = None):
        self.vocab = vocab or set()
        self.word_freq = Counter()

        # OCR confusion matrix - các ký tự dễ bị nhầm trong OCR
        self.confusion_pairs = {
            ('l', 'i'), ('l', '1'), ('i', '1'),
            ('o', '0'), ('O', '0'),
            ('s', '5'), ('S', '5'),
            ('b', 'h'), ('h', 'b'),
            ('n', 'u'), ('u', 'n'),
            ('m', 'rn'), ('rn', 'm'),
            ('d', 'cl'), ('cl', 'd'),
            ('vv', 'w'), ('w', 'vv'),
            ('ii', 'll'), ('ll', 'ii'),
        }

    def train(self, words: List[str]):
        """Train từ danh sách từ"""
        self.word_freq.update(words)
        self.vocab.update(words)

    def _edits1(self, word: str) -> set:
        """Tất cả các biến thể cách 1 edit"""
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> set:
        """Các biến thể cách 2 edits"""
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def _confusion_edits(self, word: str) -> set:
        """Edits dựa trên OCR confusion matrix"""
        candidates = set()
        word_lower = word.lower()

        for (wrong, correct) in self.confusion_pairs:
            if wrong in word_lower:
                candidates.add(word_lower.replace(wrong, correct))

        return candidates

    def _candidates(self, word: str) -> set:
        """Tìm các ứng viên sửa lỗi"""
        word_lower = word.lower()

        # Ưu tiên: từ có trong vocab > confusion edits > edit1 > edit2
        known = self.vocab.intersection([word_lower])
        if known:
            return known

        confusion = self.vocab.intersection(self._confusion_edits(word))
        if confusion:
            return confusion

        edit1 = self.vocab.intersection(self._edits1(word_lower))
        if edit1:
            return edit1

        edit2 = self.vocab.intersection(self._edits2(word_lower))
        if edit2:
            return edit2

        return {word_lower}

    def correct(self, word: str) -> str:
        """Sửa lỗi chính tả cho 1 từ"""
        if not word or len(word) < 2:
            return word

        candidates = self._candidates(word)

        # Chọn candidate có frequency cao nhất
        best = max(candidates, key=lambda w: self.word_freq.get(w, 0))

        # Giữ nguyên capitalization
        if word[0].isupper():
            best = best.capitalize()
        if word.isupper():
            best = best.upper()

        return best


class ViterbiCorrector:
    """
    Viterbi Algorithm để tìm chuỗi từ tốt nhất
    Kết hợp: spell correction + language model
    """

    def __init__(self, spell_corrector: SpellCorrector, language_model: LanguageModel):
        self.spell_corrector = spell_corrector
        self.lm = language_model

    def correct_sequence(self, words: List[str]) -> List[str]:
        """
        Sử dụng Viterbi để tìm chuỗi từ tối ưu
        """
        if not words:
            return []

        n = len(words)

        # candidates[i] = list of correction candidates for words[i]
        candidates = []
        for word in words:
            word_candidates = list(self.spell_corrector._candidates(word))
            if not word_candidates:
                word_candidates = [word.lower()]
            candidates.append(word_candidates)

        # Viterbi DP
        # dp[i][j] = (max_prob, best_prev_candidate_idx)
        dp = [[(-float('inf'), -1) for _ in range(len(candidates[i]))]
              for i in range(n)]

        # Base case: first word
        for j, candidate in enumerate(candidates[0]):
            freq = self.spell_corrector.word_freq.get(candidate, 1)
            dp[0][j] = (np.log(freq + 1), -1)

        # Fill DP table
        for i in range(1, n):
            for j, curr_candidate in enumerate(candidates[i]):
                best_prob = -float('inf')
                best_prev = -1

                for k, prev_candidate in enumerate(candidates[i - 1]):
                    # Transition probability từ prev -> curr
                    context = (prev_candidate,)
                    trans_prob = self.lm.probability(curr_candidate, context)

                    # Edit distance penalty
                    edit_dist = self._edit_distance(words[i], curr_candidate)
                    edit_penalty = np.exp(-0.5 * edit_dist)

                    total_prob = dp[i - 1][k][0] + np.log(trans_prob + 1e-10) + np.log(edit_penalty + 1e-10)

                    if total_prob > best_prob:
                        best_prob = total_prob
                        best_prev = k

                dp[i][j] = (best_prob, best_prev)

        # Backtrack để tìm best path
        best_last_idx = max(range(len(candidates[-1])),
                            key=lambda j: dp[-1][j][0])

        path = [best_last_idx]
        for i in range(n - 1, 0, -1):
            prev_idx = dp[i][path[-1]][1]
            path.append(prev_idx)

        path.reverse()

        # Reconstruct corrected words với capitalization
        corrected = []
        for i, idx in enumerate(path):
            corrected_word = candidates[i][idx]

            # Preserve original capitalization
            if words[i][0].isupper():
                corrected_word = corrected_word.capitalize()
            if words[i].isupper():
                corrected_word = corrected_word.upper()

            corrected.append(corrected_word)

        return corrected

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1.lower() != c2.lower())
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class TextPostProcessor:
    """
    Complete Post-Processing Pipeline cho OCR
    """

    def __init__(self, corpus_path: Optional[str] = None):
        self.spell_corrector = SpellCorrector()
        self.language_model = LanguageModel(n=2)

        # Load corpus nếu có
        if corpus_path:
            self._load_corpus(corpus_path)
        else:
            # Default: train trên một corpus nhỏ
            self._init_default_corpus()

        self.viterbi = ViterbiCorrector(self.spell_corrector, self.language_model)

    def _init_default_corpus(self):
        """Khởi tạo corpus mặc định (có thể thay bằng corpus lớn hơn)"""
        # Đây là corpus mẫu, trong production nên load từ file lớn
        default_text = """
        the be to of and a in that have i it for not on with he as you do at
        this but his by from they we say her she or an will my one all would
        there their what so up out if about who get which go me when make can
        like time no just him know take people into year your good some could
        them see other than then now look only come its over think also back
        after use two how our work first well way even new want because any
        these give day most us is was are been has had were said did having
        may should am being gospel fourth greek text scholars manuscripts john
        written changes plan possible necessary examine certainly modern compared
        worked version likely original wording establish such part simple coherent
        understand generations scripts showing that such changes are part of a
        simple and coherent plan to understand how this is possible it is
        necessary to examine the text of the gospel the fourth gospel was almost
        certainly written in greek a modern text of the gospel represents the
        work of generations of scholars who have compared the many manuscripts
        of john and worked out the version which is most likely to have been
        the original wording it is not possible to establish any one
        """

        words = default_text.lower().split()
        self.spell_corrector.train(words)
        self.language_model.train_from_text(default_text)

    def _load_corpus(self, path: str):
        """Load corpus từ file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            words = text.lower().split()
            self.spell_corrector.train(words)
            self.language_model.train_from_text(text)
        except Exception as e:
            print(f"Warning: Could not load corpus from {path}: {e}")
            self._init_default_corpus()

    def _fix_spacing(self, text: str) -> str:
        """Sửa các vấn đề spacing"""
        # Xóa space trước punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Thay multiple spaces bằng single space
        text = re.sub(r'\s{2,}', ' ', text)

        # Thêm space sau punctuation nếu thiếu
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)

        # Xử lý hyphen
        text = re.sub(r'-\s+', '-', text)

        return text.strip()

    def _fix_capitalization(self, text: str) -> str:
        """Sửa capitalization"""
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?]\s+)([a-z])',
                      lambda m: m.group(1) + m.group(2).upper(), text)

        return text

    def process(self, text: str, use_viterbi: bool = True) -> str:
        """
        Main processing pipeline

        Args:
            text: Input text từ OCR
            use_viterbi: Có sử dụng Viterbi correction không

        Returns:
            Corrected text
        """
        if not text or not text.strip():
            return text

        # Step 1: Fix spacing issues
        text = self._fix_spacing(text)

        # Step 2: Tokenize
        # Preserve punctuation positions
        tokens = []
        current_word = []

        for char in text:
            if char.isalnum() or char == '-':
                current_word.append(char)
            else:
                if current_word:
                    tokens.append(('WORD', ''.join(current_word)))
                    current_word = []
                if char.strip():  # Non-space punctuation
                    tokens.append(('PUNCT', char))
                elif char == ' ':
                    tokens.append(('SPACE', ' '))

        if current_word:
            tokens.append(('WORD', ''.join(current_word)))

        # Step 3: Extract words for correction
        words = [token[1] for token in tokens if token[0] == 'WORD']

        # Step 4: Apply correction
        if use_viterbi and len(words) > 0:
            corrected_words = self.viterbi.correct_sequence(words)
        else:
            corrected_words = [self.spell_corrector.correct(w) for w in words]

        # Step 5: Reconstruct text
        result = []
        word_idx = 0
        for token_type, token_value in tokens:
            if token_type == 'WORD':
                result.append(corrected_words[word_idx])
                word_idx += 1
            else:
                result.append(token_value)

        text = ''.join(result)

        # Step 6: Final capitalization fix
        text = self._fix_capitalization(text)

        return text


# === TEST ===
if __name__ == "__main__":
    processor = TextPostProcessor()

    test_cases = [
        "THE Faurth Gospel was almast certainly",
        "written in Greek . A modern test of the",
        "Bospel represents the worl of generations of",
        "scholars who have compared the many manu -",
        "scripe of John and worled out the version",
        "which is most likdy to have been the original",
        "wording . It isnot possible to estoblish any one",
    ]

    print("=" * 80)
    print("ADVANCED POST-PROCESSING TEST")
    print("Using: Viterbi Algorithm + Language Model + Spell Correction")
    print("=" * 80)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Input:     {text}")
        processed = processor.process(text, use_viterbi=True)
        print(f"Output:    {processed}")
        print("-" * 80)