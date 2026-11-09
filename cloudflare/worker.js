const MAX_PHOTO_SIZE = 5 * 1024 * 1024;

const ALLOWED_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
};

function json(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
        },
    });
}

async function authenticate(request, env) {
    const header = request.headers.get("Authorization") || "";
    const match = header.match(/^Bearer\s+(.+)$/i);

    if (!match || !env.FIREBASE_WEB_API_KEY) {
        return null;
    }

    const response = await fetch(
        "https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=" +
        encodeURIComponent(env.FIREBASE_WEB_API_KEY),
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ idToken: match[1] }),
        }
    );

    if (!response.ok) return null;

    const data = await response.json();
    const user = data.users && data.users[0];

    if (!user || !user.localId) return null;

    return user.localId;
}

async function uploadPhoto(request, env, uid) {
    const contentType = (
        request.headers.get("Content-Type") || ""
    ).split(";")[0].trim().toLowerCase();

    if (!ALLOWED_TYPES[contentType]) {
        return json({ error: "Formato nao permitido." }, 415);
    }

    const declaredSize = Number(
        request.headers.get("Content-Length") || 0
    );

    if (declaredSize > MAX_PHOTO_SIZE) {
        return json({ error: "Imagem excede 5 MB." }, 413);
    }

    const bytes = await request.arrayBuffer();

    if (bytes.byteLength === 0 || bytes.byteLength > MAX_PHOTO_SIZE) {
        return json({ error: "Tamanho de imagem invalido." }, 413);
    }

    const extension = ALLOWED_TYPES[contentType];
    const key = `profiles/${uid}/avatar.${extension}`;

    await env.PHOTOS.put(key, bytes, {
        httpMetadata: { contentType },
        customMetadata: { uid },
    });

    const photoUrl =
        `${new URL(request.url).origin}/v1/profile/photo/${uid}`;

    return json({
        success: true,
        photo_url: photoUrl,
        content_type: contentType,
        size: bytes.byteLength,
    });
}

async function getPhoto(env, uid) {
    const prefix = `profiles/${uid}/avatar.`;

    for (const extension of ["jpg", "png", "webp"]) {
        const object = await env.PHOTOS.get(prefix + extension);

        if (object) {
            return new Response(object.body, {
                headers: {
                    "Content-Type":
                        object.httpMetadata?.contentType || "application/octet-stream",
                    "Cache-Control": "private, max-age=300",
                    "X-Content-Type-Options": "nosniff",
                },
            });
        }
    }

    return json({ error: "Fotografia nao encontrada." }, 404);
}

export default {
    async fetch(request, env) {
        try {
            const url = new URL(request.url);

            if (request.method === "GET" && url.pathname === "/health") {
                return json({ service: "AlphaTrainer API", status: "ok" });
            }

            if (
                request.method !== "POST" &&
                request.method !== "GET"
            ) {
                return json({ error: "Metodo nao permitido." }, 405);
            }

            const uid = await authenticate(request, env);

            if (!uid) {
                return json({ error: "Autenticacao invalida." }, 401);
            }

            if (
                request.method === "POST" &&
                url.pathname === "/v1/profile/photo"
            ) {
                return uploadPhoto(request, env, uid);
            }

            const photoMatch = url.pathname.match(
                /^\/v1\/profile\/photo\/([^/]+)$/
            );

            if (request.method === "GET" && photoMatch) {
                const requestedUid = decodeURIComponent(photoMatch[1]);

                if (requestedUid !== uid) {
                    return json({ error: "Acesso negado." }, 403);
                }

                return getPhoto(env, uid);
            }

            return json({ error: "Rota nao encontrada." }, 404);
        } catch (error) {
            return json({ error: "Erro interno do servidor." }, 500);
        }
    },
};
