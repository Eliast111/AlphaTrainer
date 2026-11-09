"""Alpha Language Model
Modelo de linguagem educativo baseado em Transformer.
"""

import math
import random

from alpha_math.tensor import Tensor
from alpha_math.linalg import matmul, transpose, column_sums
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
                causal=True,
                seed=None if seed is None else seed + 1000 + index
            )
            for index in range(num_layers)
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

    def _forward_cache(self, token_ids):
        """Forward completo sobre listas, guardando caches para o backward."""
        embedding_rows = self.embedding.weights.data
        encoding = self.positional_encoding.encoding.data

        hidden = [
            [e + p for e, p in zip(embedding_rows[token_id], encoding[pos])]
            for pos, token_id in enumerate(token_ids)
        ]

        layer_caches = []

        for layer in self.layers:
            hidden, cache = layer.forward_cache(hidden)
            layer_caches.append(cache)

        logits = [
            [a + b for a, b in zip(row, self.output_bias)]
            for row in matmul(hidden, self.output_weights)
        ]

        probabilities = [self.softmax(row) for row in logits]

        return probabilities, {"hidden": hidden, "layers": layer_caches}

    def forward(self, token_ids):
        self._validate_tokens(token_ids)

        probabilities, _ = self._forward_cache(token_ids)

        return Tensor(probabilities)

    # ------------------------------------------------------------------
    # Treino

    def parameters(self):
        """Lista (nome, array) de todos os parametros treinaveis."""
        params = [
            ("embedding", self.embedding.weights.data),
            ("out_w", self.output_weights),
            ("out_b", self.output_bias),
        ]

        for index, layer in enumerate(self.layers):
            for name, array in layer.parameters():
                params.append((f"layer{index}.{name}", array))

        return params

    def parameter_count(self):
        total = 0

        for _, array in self.parameters():
            if array and isinstance(array[0], list):
                total += sum(len(row) for row in array)
            else:
                total += len(array)

        return total

    def loss_and_gradients(self, input_ids, target_ids):
        """Perda (entropia cruzada media) e gradientes para um par
        (entradas, alvos) do mesmo tamanho."""
        self._validate_tokens(input_ids)
        self._validate_tokens(target_ids)

        if len(input_ids) != len(target_ids):
            raise ValueError("Entradas e alvos devem ter o mesmo tamanho.")

        probabilities, cache = self._forward_cache(input_ids)
        count = len(input_ids)

        loss = 0.0
        d_logits = []

        for row, target in zip(probabilities, target_ids):
            loss -= math.log(max(row[target], 1e-300))

            grad = [p / count for p in row]
            grad[target] -= 1.0 / count
            d_logits.append(grad)

        loss /= count

        hidden = cache["hidden"]

        grads = {
            "out_w": matmul(transpose(hidden), d_logits),
            "out_b": column_sums(d_logits),
        }

        d_hidden = matmul(d_logits, transpose(self.output_weights))

        for index in range(len(self.layers) - 1, -1, -1):
            d_hidden, layer_grads = self.layers[index].backward(
                d_hidden, cache["layers"][index]
            )

            for name, value in layer_grads.items():
                grads[f"layer{index}.{name}"] = value

        embedding_grad = [
            [0.0] * self.embedding_dim for _ in range(self.vocabulary.size)
        ]

        for token_id, row in zip(input_ids, d_hidden):
            target_row = embedding_grad[token_id]

            for i, value in enumerate(row):
                target_row[i] += value

        grads["embedding"] = embedding_grad

        return loss, grads

    def make_training_pairs(self, texts, add_special_tokens=True):
        """Converte textos em pares (entradas, alvos) para treino."""
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Forneca pelo menos um texto.")

        pairs = []
        window = self.max_length + 1

        for text in texts:
            if isinstance(text, str):
                tokens = self.tokenizer.tokenize(text)
                ids = self.vocabulary.encode(
                    tokens,
                    add_bos=add_special_tokens,
                    add_eos=add_special_tokens
                )
            else:
                ids = list(text)

            for start in range(0, max(len(ids) - 1, 1), self.max_length):
                chunk = ids[start:start + window]

                if len(chunk) >= 2:
                    pairs.append((chunk[:-1], chunk[1:]))

        if not pairs:
            raise ValueError("Os textos nao produzem nenhum par de treino.")

        return pairs

    def fit(self, texts, epochs=50, **options):
        """Treina o modelo com textos. Ver LanguageModelTrainer.fit."""
        from training.lm_trainer import LanguageModelTrainer

        learning_rate = options.pop("learning_rate", 0.01)

        return LanguageModelTrainer(
            self, learning_rate=learning_rate
        ).fit(texts, epochs=epochs, **options)

    def evaluate(self, texts):
        """Perda media e perplexidade sobre textos."""
        pairs = self.make_training_pairs(texts)

        total = 0.0
        tokens = 0

        for inputs, targets in pairs:
            probabilities, _ = self._forward_cache(inputs)

            for row, target in zip(probabilities, targets):
                total -= math.log(max(row[target], 1e-300))
                tokens += 1

        loss = total / tokens

        return {
            "loss": loss,
            "perplexity": math.exp(min(loss, 700.0)),
            "tokens": tokens,
        }

    @classmethod
    def from_texts(cls, texts, **kwargs):
        """Constroi o vocabulario a partir dos textos e cria o modelo."""
        if isinstance(texts, str):
            texts = [texts]

        tokenizer = Tokenizer()
        vocabulary = Vocabulary().build(
            [tokenizer.tokenize(text) for text in texts]
        )

        return cls(vocabulary, **kwargs)

    # ------------------------------------------------------------------
    # Geracao

    def generate(
        self,
        prompt="",
        max_new_tokens=20,
        temperature=0.8,
        top_k=None,
        seed=None,
        stop_at_eos=True
    ):
        """Gera texto a continuar o prompt."""
        if not isinstance(prompt, str):
            raise TypeError("O prompt deve ser uma string.")

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens deve ser um inteiro >= 0.")

        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature deve ser >= 0.")

        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k deve ser um inteiro positivo ou None.")

        vocabulary = self.vocabulary
        rng = random.Random(seed)

        ids = [vocabulary.bos_id] + self.encode_text(prompt)
        generated = []
        blocked = {vocabulary.pad_id, vocabulary.bos_id, vocabulary.unk_id}

        for _ in range(max_new_tokens):
            context = ids[-self.max_length:]
            probabilities, _ = self._forward_cache(context)
            row = list(probabilities[-1])

            for token_id in blocked:
                row[token_id] = 0.0

            if top_k is not None and top_k < len(row):
                threshold = sorted(row, reverse=True)[top_k - 1]
                row = [p if p >= threshold else 0.0 for p in row]

            if temperature == 0:
                next_id = max(range(len(row)), key=lambda i: row[i])
            else:
                weights = [p ** (1.0 / temperature) for p in row]
                total = sum(weights)

                if total <= 0:
                    break

                next_id = rng.choices(
                    range(len(weights)), weights=weights
                )[0]

            if next_id == vocabulary.eos_id and stop_at_eos:
                break

            ids.append(next_id)
            generated.append(next_id)

        prompt_ids = self.encode_text(prompt)

        return self.decode(prompt_ids + generated)

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
