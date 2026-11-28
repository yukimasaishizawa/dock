import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  updateProfile,
  type Unsubscribe,
  type User,
} from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  onSnapshot,
  orderBy,
  query,
  setDoc,
  writeBatch,
  type DocumentData,
  type QuerySnapshot,
} from 'firebase/firestore';
import { AuthError, type PlannerBackend } from './backend';
import { auth, db, PLANS_COLLECTION } from './firebase';
import { loadFamily } from './family';
import { sortPlans } from '../lib/date';
import type {
  ConnectionState,
  FamilyMember,
  FamilySource,
  Plan,
  PlanDraft,
  Session,
} from './types';

/** 画面側へエラーを知らせるための合図（ToastProvider が拾います）。 */
function reportError(message: string) {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<string>('planner:error', { detail: message }));
  }
}

function toPlan(id: string, data: DocumentData): Plan {
  return {
    id,
    groupId: typeof data.groupId === 'string' ? data.groupId : id,
    title: typeof data.title === 'string' ? data.title : '',
    date: typeof data.date === 'string' ? data.date : '',
    startTime: typeof data.startTime === 'string' ? data.startTime : null,
    endTime: typeof data.endTime === 'string' ? data.endTime : null,
    memberIds: Array.isArray(data.memberIds)
      ? data.memberIds.filter((value): value is string => typeof value === 'string')
      : [],
    createdById: typeof data.createdById === 'string' ? data.createdById : '',
    createdByName: typeof data.createdByName === 'string' ? data.createdByName : '',
    createdAt: typeof data.createdAt === 'number' ? data.createdAt : 0,
  };
}

function toAuthError(cause: unknown): AuthError {
  const code =
    typeof cause === 'object' && cause !== null && 'code' in cause
      ? String((cause as { code: unknown }).code)
      : '';

  switch (code) {
    case 'auth/invalid-credential':
    case 'auth/invalid-login-credentials':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return new AuthError(
        'メールアドレスまたはパスワードが違います。Firebase Authentication にこのメールアドレスのユーザーが登録されているか確認してください。',
      );
    case 'auth/invalid-email':
      return new AuthError('メールアドレスの形式が正しくありません。user/users.csv を確認してください。');
    case 'auth/too-many-requests':
      return new AuthError('試行回数が多すぎます。しばらく待ってからもう一度お試しください。');
    case 'auth/network-request-failed':
      return new AuthError('ネットワークに接続できません。通信状態を確認してください。');
    case 'auth/operation-not-allowed':
      return new AuthError(
        'Firebase で「メール / パスワード」のログインが有効になっていません。Authentication → ログイン方法 から有効にしてください。',
      );
    case 'auth/user-disabled':
      return new AuthError('このアカウントは無効になっています。管理者に確認してください。');
    case 'auth/unauthorized-domain':
      return new AuthError(
        'このドメインは Firebase の承認済みドメインに含まれていません。Authentication → 設定 → 承認済みドメイン に追加してください。',
      );
    default:
      return new AuthError('ログインできませんでした。時間をおいてもう一度お試しください。');
  }
}

export function createFirebaseBackend(): PlannerBackend {
  let membersPromise: Promise<FamilySource> | null = null;

  // ---- 接続状態 -----------------------------------------------------------
  let online = typeof navigator === 'undefined' ? true : navigator.onLine;
  let pendingWrites = 0;
  let fromCache = false;
  const connectionListeners = new Set<(state: ConnectionState) => void>();

  const computeState = (): ConnectionState => {
    if (!online) return 'offline';
    if (pendingWrites > 0 || fromCache) return 'syncing';
    return 'online';
  };

  const emitConnection = () => {
    const state = computeState();
    connectionListeners.forEach((listener) => listener(state));
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => {
      online = true;
      emitConnection();
    });
    window.addEventListener('offline', () => {
      online = false;
      emitConnection();
    });
  }

  /** 書き込み中はインジケーターを「連携中」にします。 */
  async function track<T>(task: () => Promise<T>): Promise<T> {
    pendingWrites += 1;
    emitConnection();
    try {
      return await task();
    } finally {
      pendingWrites = Math.max(0, pendingWrites - 1);
      emitConnection();
    }
  }

  // ---- 家族 ---------------------------------------------------------------
  const listMembers = () => {
    membersPromise ??= loadFamily();
    return membersPromise;
  };

  async function nameForUser(user: User): Promise<string> {
    const displayName = user.displayName?.trim();
    if (displayName) return displayName;

    const email = (user.email ?? '').toLowerCase();
    try {
      const family = await listMembers();
      const matched = family.members.find(
        (member: FamilyMember) => member.email.toLowerCase() === email,
      );
      if (matched) return matched.name;
    } catch {
      // 読み込めなければメールアドレスから推測します。
    }

    return user.email?.split('@')[0] ?? 'メンバー';
  }

  async function toSession(user: User): Promise<Session> {
    return {
      userId: user.uid,
      userName: await nameForUser(user),
      email: user.email ?? '',
      signedInAt: Date.now(),
      provider: 'firebase',
    };
  }

  // ---- 予定の購読 ---------------------------------------------------------
  const handleSnapshot = (snapshot: QuerySnapshot, onPlans: (plans: Plan[]) => void) => {
    fromCache = snapshot.metadata.fromCache;
    emitConnection();
    onPlans(sortPlans(snapshot.docs.map((item) => toPlan(item.id, item.data()))));
  };

  return {
    id: 'firebase',

    listMembers,

    async restoreSession() {
      // Firebase が保存済みのログイン状態を復元し終えるまで待ちます。
      await auth.authStateReady();
      const user = auth.currentUser;
      return user ? toSession(user) : null;
    },

    async signIn(member: FamilyMember, password: string) {
      try {
        const credential = await signInWithEmailAndPassword(
          auth,
          member.email.trim(),
          password,
        );
        const user = credential.user;

        // 次回以降の表示名として家族の名前を保存しておきます（失敗しても続行）。
        if (user.displayName !== member.name) {
          void updateProfile(user, { displayName: member.name }).catch(() => undefined);
        }

        return {
          userId: user.uid,
          userName: member.name,
          email: user.email ?? member.email,
          signedInAt: Date.now(),
          provider: 'firebase' as const,
        };
      } catch (cause) {
        throw toAuthError(cause);
      }
    },

    async signOut() {
      await firebaseSignOut(auth);
    },

    subscribePlans(onPlans) {
      let unsubscribeSnapshot: Unsubscribe | null = null;

      const unsubscribeAuth = onAuthStateChanged(auth, (user) => {
        unsubscribeSnapshot?.();
        unsubscribeSnapshot = null;

        if (!user) {
          fromCache = false;
          emitConnection();
          onPlans([]);
          return;
        }

        const plansQuery = query(collection(db, PLANS_COLLECTION), orderBy('date'));
        unsubscribeSnapshot = onSnapshot(
          plansQuery,
          (snapshot) => handleSnapshot(snapshot, onPlans),
          (error) => {
            console.error('[planner] Firestore の購読に失敗しました', error);
            reportError(
              '予定を読み込めませんでした。Firestore のセキュリティルールとネットワークを確認してください。',
            );
            onPlans([]);
          },
        );
      });

      return () => {
        unsubscribeSnapshot?.();
        unsubscribeAuth();
      };
    },

    subscribeConnection(onChange) {
      connectionListeners.add(onChange);
      onChange(computeState());
      return () => {
        connectionListeners.delete(onChange);
      };
    },

    async createPlans(draft: PlanDraft, author: Session) {
      const groupId = `g_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
      const createdAt = Date.now();
      const created: Plan[] = [];

      await track(async () => {
        const batch = writeBatch(db);
        for (const date of draft.dates) {
          const reference = doc(collection(db, PLANS_COLLECTION));
          const plan: Plan = {
            id: reference.id,
            groupId,
            title: draft.title,
            date,
            startTime: draft.startTime,
            endTime: draft.endTime,
            memberIds: draft.memberIds,
            createdById: author.userId,
            createdByName: author.userName,
            createdAt,
          };
          const { id: _id, ...data } = plan;
          batch.set(reference, data);
          created.push(plan);
        }
        await batch.commit();
      });

      return created;
    },

    async deletePlan(id: string) {
      await track(() => deleteDoc(doc(db, PLANS_COLLECTION, id)));
    },

    async restorePlan(plan: Plan) {
      const { id, ...data } = plan;
      await track(() => setDoc(doc(db, PLANS_COLLECTION, id), data));
    },
  };
}
