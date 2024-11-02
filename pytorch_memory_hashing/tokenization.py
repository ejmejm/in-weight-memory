from typing import List


class CharacterTokenizer:
    """Simple character-level tokenizer that treats each character as a token."""
    
    def __init__(self):
        # Create vocabulary from printable ASCII characters
        self.chars = [chr(i) for i in range(32, 127)] + ['\n']
        self.vocab = {char: i for i, char in enumerate(self.chars)}
        self.vocab['<pad>'] = len(self.vocab)  # Add padding token
        self.vocab['<bos>'] = len(self.vocab)  # Add beginning of sequence token
        self.vocab['<eos>'] = len(self.vocab)  # Add end of sequence token
        
        # Create reverse mapping
        self.id_to_char = {i: char for char, i in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token_id = self.vocab[self.pad_token]
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __call__(self, texts: List[str], truncation: bool = False, max_length: int = None, **kwargs) -> dict:
        """
        Tokenize a batch of texts into token ids.
        
        Args:
            texts: List of strings to tokenize
            truncation: Whether to truncate sequences longer than max_length
            max_length: Maximum sequence length (including special tokens)
            
        Returns:
            Dictionary with input_ids for each text
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_ids = []
        for text in texts:
            ids = [self.vocab.get(c, self.vocab['<pad>']) for c in text[:max_length]]
            all_ids.append(ids)
            
        return {'input_ids': all_ids}