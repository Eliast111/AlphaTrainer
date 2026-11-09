const fs = require("fs");
const adminApp = require("firebase-admin/app");
const adminAuth = require("firebase-admin/auth");

const serviceAccount = require("../secrets/firebase-adminsdk.json");
const uid = "CPf0ij83nUd60mMGalPB0V25Ik42";

const sql = `
-- AlphaTrainer: políticas de segurança das fotografias
-- Cada utilizador só pode aceder à sua própria pasta.
-- Caminho esperado: UID/avatar.jpg (ou png/webp)

DROP POLICY IF EXISTS "AlphaTrainer upload own photos" ON storage.objects;
DROP POLICY IF EXISTS "AlphaTrainer read own photos" ON storage.objects;
DROP POLICY IF EXISTS "AlphaTrainer update own photos" ON storage.objects;
DROP POLICY IF EXISTS "AlphaTrainer delete own photos" ON storage.objects;

CREATE POLICY "AlphaTrainer upload own photos"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.uid()::text)
);

CREATE POLICY "AlphaTrainer read own photos"
ON storage.objects
FOR SELECT
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.uid()::text)
);

CREATE POLICY "AlphaTrainer update own photos"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.uid()::text)
)
WITH CHECK (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.uid()::text)
);

CREATE POLICY "AlphaTrainer delete own photos"
ON storage.objects
FOR DELETE
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.uid()::text)
);
`;

async function main() {
    let app;

    try {
        if (serviceAccount.project_id !== "allphatrainer") {
            throw new Error("O projeto da chave não corresponde a allphatrainer.");
        }

        app = adminApp.initializeApp({
            credential: adminApp.cert(serviceAccount)
        });

        const auth = adminAuth.getAuth(app);
        const user = await auth.getUser(uid);
        const claims = user.customClaims || {};

        console.log("Projeto:", serviceAccount.project_id);
        console.log("UID:", user.uid);
        console.log("Email:", user.email);
        console.log("Role:", claims.role || "Não encontrada");

        if (claims.role !== "authenticated") {
            throw new Error("A claim authenticated ainda não está confirmada.");
        }

        fs.writeFileSync("supabase/profile_photos_policies.sql", sql);

        console.log("\nSUCESSO: claim confirmada.");
        console.log("SQL criado: supabase/profile_photos_policies.sql");
        console.log("Políticas preparadas: 4");
    } finally {
        if (app) {
            await adminApp.deleteApp(app);
        }
    }
}

main().catch(error => {
    console.error("\nERRO:", error.message);
    process.exitCode = 1;
});
