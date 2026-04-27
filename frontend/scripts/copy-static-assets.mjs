import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(here, "..");
const distRoot = resolve(frontendRoot, "dist");

await mkdir(distRoot, { recursive: true });

for (const assetName of ["app.html", "research-template.csv"]) {
  await copyFile(resolve(frontendRoot, assetName), resolve(distRoot, assetName));
}
