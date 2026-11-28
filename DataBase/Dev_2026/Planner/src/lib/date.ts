import type { ISODate, Plan } from '../data/types';

export const WEEKDAYS_JP = ['日', '月', '火', '水', '木', '金', '土'] as const;

const pad = (n: number): string => String(n).padStart(2, '0');

/** ローカル時間の Date を `YYYY-MM-DD` に変換します（UTC 変換による日付ズレを避ける）。 */
export function toKey(date: Date): ISODate {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function fromKey(key: ISODate): Date {
  const [y, m, d] = key.split('-').map((part) => Number(part));
  return new Date(y, (m || 1) - 1, d || 1);
}

export function todayKey(): ISODate {
  return toKey(new Date());
}

export function addDaysKey(key: ISODate, days: number): ISODate {
  const date = fromKey(key);
  date.setDate(date.getDate() + days);
  return toKey(date);
}

export function isSameMonth(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth();
}

/**
 * カレンダーのマス目（日曜はじまり）。月をまたぐ前後の日も含み、
 * 行数はその月に必要な週数だけにします。
 */
export function monthGrid(year: number, monthIndex: number): Date[] {
  const firstWeekday = new Date(year, monthIndex, 1).getDay();
  const daysInMonth = new Date(year, monthIndex + 1, 0).getDate();
  const totalCells = Math.ceil((firstWeekday + daysInMonth) / 7) * 7;
  return Array.from({ length: totalCells }, (_, i) => new Date(year, monthIndex, 1 - firstWeekday + i));
}

export function monthLabel(year: number, monthIndex: number): string {
  return `${year}年${monthIndex + 1}月`;
}

/** `11月12日(水)` */
export function formatJP(key: ISODate): string {
  const date = fromKey(key);
  return `${date.getMonth() + 1}月${date.getDate()}日(${WEEKDAYS_JP[date.getDay()]})`;
}

/** `2026年11月12日 水曜日` */
export function formatLongJP(key: ISODate): string {
  const date = fromKey(key);
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日 ${WEEKDAYS_JP[date.getDay()]}曜日`;
}

/** `11/12` */
export function formatShort(key: ISODate): string {
  const date = fromKey(key);
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

export function relativeLabel(key: ISODate): '' | '今日' | '明日' | '昨日' {
  const today = todayKey();
  if (key === today) return '今日';
  if (key === addDaysKey(today, 1)) return '明日';
  if (key === addDaysKey(today, -1)) return '昨日';
  return '';
}

export function isValidDateKey(value: string | undefined): value is ISODate {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = fromKey(value);
  return !Number.isNaN(date.getTime()) && toKey(date) === value;
}

export function formatTimeRange(start: string | null, end: string | null): string {
  if (start && end) return `${start} – ${end}`;
  if (start) return start;
  return '終日';
}

export function timeLabel(start: string | null, end: string | null): { primary: string; secondary: string | null } {
  if (start && end) return { primary: start, secondary: end };
  if (start) return { primary: start, secondary: null };
  return { primary: '終日', secondary: null };
}

/** 時間ありの予定を時刻順に、時間なし（終日）の予定を先頭に並べます。 */
export function sortPlans(plans: Plan[]): Plan[] {
  return [...plans].sort((a, b) => {
    if (!a.startTime && !b.startTime) return a.createdAt - b.createdAt;
    if (!a.startTime) return -1;
    if (!b.startTime) return 1;
    if (a.startTime !== b.startTime) return a.startTime < b.startTime ? -1 : 1;
    return a.createdAt - b.createdAt;
  });
}

export function groupPlansByDate(plans: Plan[]): Map<ISODate, Plan[]> {
  const map = new Map<ISODate, Plan[]>();
  for (const plan of plans) {
    const bucket = map.get(plan.date);
    if (bucket) bucket.push(plan);
    else map.set(plan.date, [plan]);
  }
  for (const [key, bucket] of map) map.set(key, sortPlans(bucket));
  return map;
}

export function plansOn(plans: Plan[], key: ISODate): Plan[] {
  return sortPlans(plans.filter((plan) => plan.date === key));
}
