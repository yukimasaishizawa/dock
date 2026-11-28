import { parseFamilyCsv, type CsvPerson } from '../lib/csv';
import type { FamilyMember, FamilySource } from './types';

/** メンバーに割り当てる色。CSV の並び順で先頭から使います。 */
export const MEMBER_COLORS = ['#d3562b', '#3f7ca6', '#5c8a5e', '#a8703f', '#8a6ba8', '#b25573'];

/**
 * CSV がまだ配置されていないときのサンプル。
 * 本番では `public/user/users.csv` を置けばそちらが優先されます。
 */
const SAMPLE_FAMILY: CsvPerson[] = [
  { name: 'お父さん', email: 'papa@example.com' },
  { name: 'お母さん', email: 'mama@example.com' },
  { name: 'はると', email: 'haruto@example.com' },
  { name: 'みお', email: 'mio@example.com' },
];

/**
 * 家族 CSV の配信パス。リポジトリ直下の `user/` に置いたファイルは
 * `scripts/sync-user-csv.mjs` がビルド時に `public/user/` へコピーします。
 */
export const FAMILY_CSV_PATHS = ['user/users.csv', 'user/family.csv', 'user/members.csv'];

export async function loadFamily(): Promise<FamilySource> {
  for (const path of FAMILY_CSV_PATHS) {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}${path}`, { cache: 'no-cache' });
      if (!response.ok) continue;
      const people = parseFamilyCsv(await response.text());
      if (people.length > 0) {
        return { members: toMembers(people), source: 'csv' };
      }
    } catch {
      // オフラインやファイル未配置。次の候補・サンプルへ進みます。
    }
  }

  return { members: toMembers(SAMPLE_FAMILY), source: 'fallback' };
}

function toMembers(people: CsvPerson[]): FamilyMember[] {
  return people.map((person, index) => {
    const name = person.name.trim();
    const email = person.email.trim();
    const color = MEMBER_COLORS[index % MEMBER_COLORS.length];
    return {
      // メールアドレスを ID にしておくと、CSV の行を入れ替えても予定との紐付けが壊れません。
      id: `m:${(email || name).toLowerCase()}`,
      name,
      email,
      color,
      initial: name.slice(0, 1).toUpperCase(),
    };
  });
}
