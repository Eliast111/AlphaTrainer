const fs = require("fs");
const path = require("path");

const { initializeApp, cert } = require("firebase-admin/app");
const { getAuth } = require("firebase-admin/auth");

const keyPath = path.join(
  __dirname,
  "../secrets/firebase-adminsdk.json"
);

if (!fs.existsSync(keyPath)) {
  throw new Error("Chave privada não encontrada.");
}

const serviceAccount = require(keyPath);

if (serviceAccount.project_id !== "allphatrainer") {
  throw new Error("A chave não pertence ao projeto AlphaTrainer.");
}

const app = initializeApp({
  credential: cert(serviceAccount)
});

const auth = getAuth(app);
const APPLY = process.argv.includes("--apply");

async function main() {
  let pageToken;
  let total = 0;
  let updated = 0;
  let skipped = 0;

  console.log("Projeto:", serviceAccount.project_id);
  console.log("Modo:", APPLY ? "APLICAR ALTERAÇÕES" : "SIMULAÇÃO");

  do {
    const result = await auth.listUsers(1000, pageToken);

    for (const user of result.users) {
      total++;

      const claims = user.customClaims || {};

      if (claims.role === "authenticated") {
        skipped++;
        continue;
      }

      if (APPLY) {
        await auth.setCustomUserClaims(user.uid, {
          ...claims,
          role: "authenticated"
        });
      }

      updated++;
    }

    pageToken = result.pageToken;
  } while (pageToken);

  console.log("\n=== RESULTADO ===");
  console.log("Utilizadores encontrados:", total);
  console.log("A adicionar role:", updated);
  console.log("Já configurados:", skipped);
  console.log("Alterações aplicadas:", APPLY);
}

main()
  .catch(error => {
    console.error("Erro:", error.message);
    process.exitCode = 1;
  })
  .finally(() => app.delete());
