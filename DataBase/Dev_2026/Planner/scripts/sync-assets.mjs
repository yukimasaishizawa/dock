#!/usr/bin/env node
/**
 * リポジトリ直下の「人が差し替えるファイル」を、配信される `public/` へコピーします。
 *
 *   image/logo.png   → public/image/logo.png   （サイトのロゴ・PWA アイコン）
 *   user/*.csv       → public/user/*.csv       （家族の一覧）
 *
 * 静的サイトでは `public/` の中だけがブラウザから読めるため、このコピーが必要です。
 * 元のファイルが無い場合は何もしないので、`public/` 側を直接編集しても構いません。
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

function copyLogo() {
  const source = 'image/logo.png';
  if (!existsSync(source)) return;
  mkdirSync('public/image', { recursive: true });
  copyFileSync(source, 'public/image/logo.png');
  console.log(`[assets] ${source} → public/image/logo.png`);
}

function copyFamilyCsv() {
  const source = 'user';
  if (!existsSync(source) || !statSync(source).isDirectory()) return;

  const csvFiles = readdirSync(source).filter((file) => file.toLowerCase().endsWith('.csv'));
  if (csvFiles.length === 0) return;

  mkdirSync('public/user', { recursive: true });
  for (const file of csvFiles) {
    copyFileSync(join(source, file), join('public/user', file));
    console.log(`[assets] ${join(source, file)} → public/user/${file}`);
  }
}

copyLogo();
copyFamilyCsv();
