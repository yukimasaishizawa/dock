export interface CsvPerson {
  name: string;
  email: string;
}

/**
 * `"test1";"test1's.gmail.com"` のような行を読み取ります。
 * - 区切りは `;` と `,` の両方に対応（ファイル内で自動判定）
 * - 前後の空白、BOM、CRLF、空行、`#` コメント行を許容
 * - 1行目がヘッダー（name / ユーザー名 / email など）なら読み飛ばす
 */
export function parseFamilyCsv(text: string): CsvPerson[] {
  const cleaned = text.replace(/^\uFEFF/, '');
  const lines = cleaned
    .split(/\r\n|\n|\r/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'));

  if (lines.length === 0) return [];

  const delimiter = detectDelimiter(lines[0]);
  const rows = lines.map((line) => splitRow(line, delimiter));

  const first = rows[0];
  if (first && looksLikeHeader(first)) rows.shift();

  const people: CsvPerson[] = [];
  for (const row of rows) {
    const name = (row[0] ?? '').trim();
    const email = (row[1] ?? '').trim();
    if (!name) continue;
    people.push({ name, email });
  }
  return people;
}

function detectDelimiter(line: string): string {
  const semicolons = (line.match(/;/g) ?? []).length;
  const commas = (line.match(/,/g) ?? []).length;
  return semicolons >= commas && semicolons > 0 ? ';' : ',';
}

function splitRow(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let current = '';
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
      continue;
    }

    if (char === delimiter && !quoted) {
      cells.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  cells.push(current);
  return cells;
}

function looksLikeHeader(cells: string[]): boolean {
  const head = (cells[0] ?? '').toLowerCase();
  const second = (cells[1] ?? '').toLowerCase();
  const nameLike = /^(name|user|username|ユーザー名|名前|氏名)$/.test(head);
  const mailLike = /^(e-?mail|メール|メールアドレス|mail|email address)$/.test(second);
  return nameLike && mailLike;
}
