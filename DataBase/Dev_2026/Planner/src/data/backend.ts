import { createFirebaseBackend } from './firebaseBackend';
import type {
  ConnectionState,
  FamilyMember,
  FamilySource,
  Plan,
  PlanDraft,
  Session,
} from './types';

/**
 * アプリがデータ層に求めるすべての操作。
 *
 * 実装は `firebaseBackend.ts`（Firebase Authentication + Cloud Firestore）。
 * ネットワーク無しで画面だけ確認したいときは、下の `backend` を
 * `createMockBackend()` に差し替えると localStorage で動きます。
 */
export interface PlannerBackend {
  readonly id: 'mock' | 'firebase';

  /** user/users.csv から家族を読み込みます。 */
  listMembers(): Promise<FamilySource>;

  /** 保存済みセッションの復元。未ログインなら null。 */
  restoreSession(): Promise<Session | null>;

  /** ユーザー名に紐づくメールアドレスとパスワードで Firebase にサインインします。 */
  signIn(member: FamilyMember, password: string): Promise<Session>;

  signOut(): Promise<void>;

  /** 予定の購読。ログイン後にだけ Firestore を購読します。 */
  subscribePlans(onPlans: (plans: Plan[]) => void): () => void;

  subscribeConnection(onChange: (state: ConnectionState) => void): () => void;

  createPlans(draft: PlanDraft, author: Session): Promise<Plan[]>;

  deletePlan(id: string): Promise<void>;

  /** 削除の取り消し用。同じ ID の予定を書き戻します。 */
  restorePlan(plan: Plan): Promise<void>;
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthError';
  }
}

/**
 * アプリが使うデータ層。
 *
 *   Firebase を使う      : createFirebaseBackend()
 *   オフラインで試す     : createMockBackend()   ← `./mockBackend` から import
 */
export const backend: PlannerBackend = createFirebaseBackend();
