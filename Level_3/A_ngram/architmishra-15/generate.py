import argparse
import random
import sys

class NGramGenerator:
    def __init__(self, n):
        if n < 1:
            raise ValueError("N-gram size (n) must be at least 1")
        self.n = n
        # models[k] holds the context of length k mapping to a list of next words.
        # k ranges from 0 to n-1. 
        # k=0 is the unigram fallback (empty context tuple maps to all words).
        self.models = {k: {} for k in range(n)}

    def train(self, tokens):
        if not tokens:
            return
        
        for i in range(len(tokens)):
            target = tokens[i]
            for k in range(self.n):
                if i - k < 0:
                    continue
                
                ctx = tuple(tokens[i-k:i]) if k > 0 else ()
                
                if ctx not in self.models[k]:
                    self.models[k][ctx] = []
                self.models[k][ctx].append(target)

    def generate(self, length, seed):
        random.seed(seed)
        
        # If the corpus was empty or training failed, return empty.
        if not self.models.get(0) or () not in self.models[0]:
            return []
        
        generated = []
        for _ in range(length):
            # Backoff: Start from the longest possible context (n-1) down to 0.
            for k in range(self.n - 1, -1, -1):
                ctx = tuple(generated[-k:]) if k > 0 else ()
                
                if ctx in self.models[k] and self.models[k][ctx]:
                    next_word = random.choice(self.models[k][ctx])
                    generated.append(next_word)
                    break
                    
        return generated

def main():
    parser = argparse.ArgumentParser(description="N-gram text generator with backoff")
    parser.add_argument("--n", type=int, required=True, help="N-gram size")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--length", type=int, required=True, help="Number of words to generate")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not read file {args.input}", file=sys.stderr)
        sys.exit(1)

    tokens = text.split()
    
    if not tokens or args.length == 0:
        print("")
        return

    try:
        generator = NGramGenerator(args.n)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    generator.train(tokens)
    result = generator.generate(args.length, args.seed)
    
    print(" ".join(result))

if __name__ == "__main__":
    main()
