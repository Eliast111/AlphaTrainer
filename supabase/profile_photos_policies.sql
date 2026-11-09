
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
    AND (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

CREATE POLICY "AlphaTrainer read own photos"
ON storage.objects
FOR SELECT
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

CREATE POLICY "AlphaTrainer update own photos"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
)
WITH CHECK (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);

CREATE POLICY "AlphaTrainer delete own photos"
ON storage.objects
FOR DELETE
TO authenticated
USING (
    bucket_id = 'profile-photos'
    AND (storage.foldername(name))[1] = (select auth.jwt()->>'sub')
);
