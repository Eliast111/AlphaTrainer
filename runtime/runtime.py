"""Lightweight inference runtime for AlphaTrainer's native models."""


class AlphaRuntime:
    """Run inference without requiring an external ML framework."""

    @staticmethod
    def run(model, inputs):
        forward = getattr(model, "forward", None)
        if not callable(forward):
            raise TypeError("O modelo deve disponibilizar um método forward().")
        return forward(inputs)

    @staticmethod
    def predict_next(model, token_ids):
        predict = getattr(model, "predict_next", None)
        if not callable(predict):
            raise TypeError("O modelo não suporta previsão do próximo token.")
        return predict(token_ids)

    def generate(self, model, token_ids, max_new_tokens=20, eos_id=None):
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("token_ids deve ser uma lista não vazia.")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens deve ser um inteiro não negativo.")

        generated = list(token_ids)
        model_limit = getattr(model, "max_length", None)
        for _ in range(max_new_tokens):
            context = generated
            if isinstance(model_limit, int) and len(context) > model_limit:
                context = context[-model_limit:]
            next_token = self.predict_next(model, context)
            generated.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break
        return generated

    def generate_text(self, model, text, max_new_tokens=20):
        encode = getattr(model, "encode_text", None)
        decode = getattr(model, "decode", None)
        vocabulary = getattr(model, "vocabulary", None)
        if not callable(encode) or not callable(decode) or vocabulary is None:
            raise TypeError("O modelo não suporta geração de texto.")
        token_ids = encode(text)
        generated = self.generate(
            model,
            token_ids,
            max_new_tokens=max_new_tokens,
            eos_id=vocabulary.eos_id,
        )
        return decode(generated)


__all__ = ["AlphaRuntime"]
