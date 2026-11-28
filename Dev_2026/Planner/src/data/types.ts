/** Calendar day key in local time, formatted as `YYYY-MM-DD`. */
export type ISODate = string;

/** Firebase との接続状態。ヘッダーのインジケーターがこの値で切り替わります。 */
export type ConnectionState = 'online' | 'syncing' | 'offline';

/** 家族のひとり。`user/users.csv` の1行、または将来の Firestore のドキュメントに対応します。 */
export interface FamilyMember {
  id: string;
  name: string;
  email: string;
  /** アバターやカレンダーのタイルに使う固有色（CSS カラー）。 */
  color: string;
  /** アバターに表示する1文字。 */
  initial: string;
}

/**
 * 1日分の予定。複数日をまとめて登録した場合も、日付ごとに1件ずつ生成し、
 * `groupId` で同じ登録操作から生まれた予定をひも付けます。
 */
export interface Plan {
  id: string;
  groupId: string;
  title: string;
  date: ISODate;
  /** `HH:mm`。時間帯を空欄にした予定は `null`（終日）。 */
  startTime: string | null;
  endTime: string | null;
  /** 対象者。1人以上必須。 */
  memberIds: string[];
  /** 登録者。あとから「誰が登録したか」を表示するために保存します。 */
  createdById: string;
  createdByName: string;
  createdAt: number;
}

/** 新規登録フォームがバックエンドへ渡す内容。 */
export interface PlanDraft {
  title: string;
  dates: ISODate[];
  startTime: string | null;
  endTime: string | null;
  memberIds: string[];
}

/** ログイン中のセッション。Firebase 実装では uid がそのまま入ります。 */
export interface Session {
  userId: string;
  userName: string;
  email: string;
  signedInAt: number;
  provider: 'mock' | 'firebase';
}

export interface FamilySource {
  members: FamilyMember[];
  /** `csv` = user/users.csv を読み込めた / `fallback` = 同梱サンプル。 */
  source: 'csv' | 'fallback';
}
