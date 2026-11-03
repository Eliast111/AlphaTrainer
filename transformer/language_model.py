"""Alpha Language Model
Modelo de linguagem educativo baseado em Transformer.
"""

import math
import random

from alpha_math.tensor import Tensor
from language.tokenizer import Tokenizer
from language.vocabulary import Vocabulary
from language.embedding import Embedding
from transformer.positional_encoding import PositionalEncoding
from transformer.layer import TransformerLayer


class LanguageModel:

    def __init__(
        self,
        vocabulary,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        max_length=128,
        seed=42
    ):
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("É necessário fornecer um Vocabulary.")

        for name, value in (
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
            ("num_layers", num_layers),
            ("max_length", max_length)
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} deve ser inteiro.")

            if value <= 0:
                raise ValueError(f"{name} deve ser positivo.")

        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed deve ser inteiro ou None.")

        self.vocabulary = vocabulary
        self.tokenizer = Tokenizer()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = Embedding(
            vocabulary.size,
            embedding_dim,
            seed=seed
        )

        self.positional_encoding = PositionalEncoding(
            max_length,
            embedding_dim
        )

        self.layers = [
            TransformerLayer(
                embedding_dim,
                hidden_dim,
                causal=True
            )
            for _ in range(num_layers)
        ]

        generator = random.Random(seed)

        scale = 1.0 / math.sqrt(embedding_dim)

        self.output_weights = [
            [
                generator.uniform(-scale, scale)
                for _ in range(vocabulary.size)
            ]
            for _ in range(embedding_dim)
        ]

        self.output_bias = [0.0] * vocabulary.size

    def _validate_tokens(self, token_ids):
        if not isinstance(token_ids, list):
            raise TypeError("Os IDs devem estar numa lista.")

        if not token_ids:
            raise ValueError("A sequência não pode estar vazia.")

        if len(token_ids) > self.max_length:
            raise ValueError("Sequência maior que max_length.")

        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Cada ID deve ser inteiro.")

            if token_id < 0 or token_id >= self.vocabulary.size:
                raise ValueError("ID fora do vocabulário.")

    def softmax(self, values):
        if not values:
            raise ValueError("A lista não pode estar vazia.")

        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exponentials)

        return [
            value / total
            for value in exponentials
        ]

    def forward(self, token_ids):
        self._validate_tokens(token_ids)

        embeddings = self.embedding.forward(token_ids)

        hidden = self.positional_encoding.forward(embeddings)

        for layer in self.layers:
            hidden = layer.forward(hidden)

        logits = []

        for row in hidden.data:
            values = []

            for token_index in range(self.vocabulary.size):
                value = sum(
                    row[i] * self.output_weights[i][token_index]
                    for i in range(self.embedding_dim)
                ) + self.output_bias[token_index]

                values.append(value)

            logits.append(self.softmax(values))

        return Tensor(logits)

    def predict_next(self, token_ids):
        probabilities = self.forward(token_ids)

        last_row = probabilities.data[-1]

        return max(
            range(len(last_row)),
            key=lambda index: last_row[index]
        )

    def encode_text(self, text):
        tokens = self.tokenizer.tokenize(text)

        return self.vocabulary.encode(tokens)

    def predict_next_text(self, text):
        token_ids = self.encode_text(text)

        return self.predict_next(token_ids)

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = self.vocabulary.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

        return self.tokenizer.detokenize(tokens)
