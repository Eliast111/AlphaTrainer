class Vocabulary:
    """Converte tokens em IDs e IDs em tokens."""

    SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = []

        for token in self.SPECIAL_TOKENS:
            self.add_token(token)

    @property
    def size(self):
        return len(self.id_to_token)

    @property
    def pad_id(self):
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self):
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self):
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self):
        return self.token_to_id["<EOS>"]

    def add_token(self, token):
        if not isinstance(token, str) or not token:
            raise ValueError("O token deve ser uma string não vazia.")

        if token not in self.token_to_id:
            token_id = len(self.id_to_token)
            self.token_to_id[token] = token_id
            self.id_to_token.append(token)

        return self.token_to_id[token]

    def build(self, sequences):
        if not isinstance(sequences, list):
            raise TypeError("As sequências devem estar numa lista.")

        for sequence in sequences:
            if not isinstance(sequence, list):
                raise TypeError("Cada sequência deve ser uma lista.")

            for token in sequence:
                self.add_token(token)

        return self

    def get_id(self, token):
        return self.token_to_id.get(token, self.unk_id)

    def get_token(self, token_id):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("O ID deve ser um número inteiro.")

        if token_id < 0 or token_id >= self.size:
            raise ValueError("ID fora do vocabulário.")

        return self.id_to_token[token_id]

    def encode(self, tokens, add_bos=False, add_eos=False):
        if not isinstance(tokens, list):
            raise TypeError("Os tokens devem estar numa lista.")

        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("Cada token deve ser uma string.")

            ids.append(self.get_id(token))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special_tokens=False):
        if not isinstance(ids, list):
            raise TypeError("Os IDs devem estar numa lista.")

        tokens = []

        for token_id in ids:
            token = self.get_token(token_id)

            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return tokens
