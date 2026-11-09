# AlphaTrainer Cloudflare API

## Recursos

- Firebase Authentication para validar o utilizador.
- Cloudflare R2 para guardar fotografias.
- Upload limitado a 5 MB.
- Formatos aceites: JPEG, PNG e WebP.
- Cada utilizador só pode consultar a própria fotografia.

## Configuração

1. Criar um bucket R2 chamado `alphatrainer-photos`.
2. Instalar Wrangler e autenticar a conta Cloudflare.
3. Configurar o segredo FIREBASE_WEB_API_KEY.
4. Executar `wrangler deploy`.

A chave Firebase deve ser configurada como segredo no Worker.
Nunca colocar credenciais R2 na aplicação Android.
