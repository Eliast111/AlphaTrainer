"""Tokenizador simples, deterministico e sem dependencias externas.

Separa palavras, numeros e pontuacao; preserva acentos e caracteres Unicode.
"""
import re


class Tokenizer:
    _pattern = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
    _closing = set(",.;:!?%)]}»”’")
    _opening = set("([{«“")

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("O texto deve ser uma string.")
        return self._pattern.findall(text)

    def detokenize(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            raise TypeError("Os tokens devem estar numa lista ou tuplo.")
        if any(not isinstance(token, str) for token in tokens):
            raise TypeError("Cada token deve ser uma string.")
        return " ".join(token for token in tokens if token)
