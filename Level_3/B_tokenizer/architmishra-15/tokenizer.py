class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, num_merges):
        tokens = list(text.encode('utf-8'))
        
        for i in range(num_merges):
            counts = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                counts[pair] = counts.get(pair, 0) + 1
            
            if not counts:
                break  
                
            # Tie-breaking logic:
            # -counts[p] ensures we pick the highest frequency first.
            # p[0] and p[1] ensure we pick the smallest (first_id, second_id) if frequencies tie.
            best_pair = min(counts.keys(), key=lambda p: (-counts[p], p[0], p[1]))
            
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        
        while len(tokens) >= 2:
            # Find the adjacent pair in the current sequence that was merged earliest (lowest rank)
            best_pair = None
            min_rank = float('inf')
            
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                if pair in self.merges and self.merges[pair] < min_rank:
                    min_rank = self.merges[pair]
                    best_pair = pair
            
            if best_pair is None:
                break
                
            # Apply the highest priority merge everywhere in the sequence
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(min_rank)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            
        return tokens

    def decode(self, ids):
        byte_sequence = b''.join(self.vocab[i] for i in ids)
        return byte_sequence.decode('utf-8')
