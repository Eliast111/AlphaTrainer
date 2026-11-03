import re


class Tokenizer:
    """Converte texto em tokens e tokens em texto."""

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("O texto deve ser uma string.")

        text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def detokenize(self, tokens):
        if not isinstance(tokens, list):
            raise TypeError("Os tokens devem estar numa lista.")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Cada token deve ser uma string.")

        return " ".join(tokens)
