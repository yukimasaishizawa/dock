import { AuthError, type PlannerBackend } from './backend';
import { loadFamily } from './family';
import type {
  ConnectionState,
  FamilyMember,
  FamilySource,
  Plan,
  PlanDraft,
  Session,
} from './types';

const PLANS_KEY = 'planner.plans.v1';
const SESSION_KEY = 'planner.session.v1';
const CHANNEL_NAME = 'planner.sync.v1';
const SESSION_TTL = 1000 * 60 * 60 * 24 * 30;

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });

const newId = (): string =>
  `p_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

function readPlans(): Plan[] {
  try {
    const raw = window.localStorage.getItem(PLANS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Plan[]) : [];
  } catch {
    return [];
  }
}

function readSession(): Session | null {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Session;
    if (!parsed?.userId || !parsed?.userName) return null;
    if (Date.now() - (parsed.signedInAt ?? 0) > SESSION_TTL) return null;
    return { ...parsed, provider: 'mock' };
  } catch {
    return null;
  }
}

/**
 * Firebase が入る前の仮実装。
 *
 * - 予定は localStorage に保存し、別タブ／別ウィンドウには BroadcastChannel で同期します
 *   （Firestore のリアルタイム購読と同じ「他の人の変更が流れてくる」体験を再現）。
 * - 認証はユーザー選択 + パスワードのみ。パスワードは検証せず、送信の流れだけを再現します。
 * - 接続状態は online / syncing / offline の3状態を返します。
 */
export function createMockBackend(): PlannerBackend {
  const planListeners = new Set<(plans: Plan[]) => void>();
  const connectionListeners = new Set<(state: ConnectionState) => void>();

  let pendingWrites = 0;
  let applyingRemote = 0;
  let networkOnline = typeof navigator === 'undefined' ? true : navigator.onLine;
  let channel: BroadcastChannel | null = null;

  const currentState = (): ConnectionState => {
    if (!networkOnline) return 'offline';
    if (pendingWrites > 0 || applyingRemote > 0) return 'syncing';
    return 'online';
  };

  const emitConnection = (): void => {
    const state = currentState();
    connectionListeners.forEach((listener) => listener(state));
  };

  const emitPlans = (plans: Plan[]): void => {
    planListeners.forEach((listener) => listener(plans));
  };

  const writePlans = (plans: Plan[]): void => {
    try {
      window.localStorage.setItem(PLANS_KEY, JSON.stringify(plans));
    } catch {
      // ストレージが使えない環境でも、画面はメモリ上の購読で動き続けます。
    }
    emitPlans(plans);
  };

  const broadcast = (): void => {
    try {
      channel?.postMessage({ type: 'plans-changed', at: Date.now() });
    } catch {
      // 送信できなくても致命的ではありません。
    }
  };

  const ensureChannel = (): void => {
    if (channel || typeof BroadcastChannel === 'undefined') return;
    channel = new BroadcastChannel(CHANNEL_NAME);
    channel.onmessage = (event: MessageEvent) => {
      if ((event.data as { type?: string } | null)?.type !== 'plans-changed') return;
      applyingRemote += 1;
      emitConnection();
      window.setTimeout(() => {
        emitPlans(readPlans());
        applyingRemote = Math.max(0, applyingRemote - 1);
        emitConnection();
      }, 300);
    };
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => {
      networkOnline = true;
      emitConnection();
    });
    window.addEventListener('offline', () => {
      networkOnline = false;
      emitConnection();
    });
  }

  return {
    id: 'mock',

    async listMembers(): Promise<FamilySource> {
      const family = await loadFamily();
      await wait(180);
      return family;
    },

    async restoreSession(): Promise<Session | null> {
      return readSession();
    },

    async signIn(member: FamilyMember, password: string): Promise<Session> {
      if (!password.trim()) {
        throw new AuthError('パスワードを入力してください。');
      }

      pendingWrites += 1;
      emitConnection();
      try {
        // Firebase 実装では、ここで member.email と password を
        // signInWithEmailAndPassword に渡します。
        await wait(680);
        const session: Session = {
          userId: member.id,
          userName: member.name,
          email: member.email,
          signedInAt: Date.now(),
          provider: 'mock',
        };
        window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
        return session;
      } finally {
        pendingWrites = Math.max(0, pendingWrites - 1);
        emitConnection();
      }
    },

    async signOut(): Promise<void> {
      window.localStorage.removeItem(SESSION_KEY);
    },

    subscribePlans(onPlans: (plans: Plan[]) => void): () => void {
      ensureChannel();
      planListeners.add(onPlans);
      onPlans(readPlans());

      const onStorage = (event: StorageEvent) => {
        if (event.key !== PLANS_KEY) return;
        applyingRemote += 1;
        emitConnection();
        window.setTimeout(() => {
          emitPlans(readPlans());
          applyingRemote = Math.max(0, applyingRemote - 1);
          emitConnection();
        }, 260);
      };

      window.addEventListener('storage', onStorage);
      return () => {
        planListeners.delete(onPlans);
        window.removeEventListener('storage', onStorage);
      };
    },

    subscribeConnection(onChange: (state: ConnectionState) => void): () => void {
      connectionListeners.add(onChange);
      onChange(currentState());
      return () => {
        connectionListeners.delete(onChange);
      };
    },

    async createPlans(draft: PlanDraft, author: Session): Promise<Plan[]> {
      pendingWrites += 1;
      emitConnection();
      try {
        await wait(420);

        const groupId = newId();
        const createdAt = Date.now();
        const created: Plan[] = draft.dates.map((date, index) => ({
          id: `${groupId}_${index}`,
          groupId,
          title: draft.title.trim(),
          date,
          startTime: draft.startTime,
          endTime: draft.endTime,
          memberIds: [...draft.memberIds],
          createdById: author.userId,
          createdByName: author.userName,
          createdAt: createdAt + index,
        }));

        writePlans([...readPlans(), ...created]);
        broadcast();
        await wait(200);
        return created;
      } finally {
        pendingWrites = Math.max(0, pendingWrites - 1);
        emitConnection();
      }
    },

    async deletePlan(id: string): Promise<void> {
      pendingWrites += 1;
      emitConnection();
      try {
        await wait(300);
        writePlans(readPlans().filter((plan) => plan.id !== id));
        broadcast();
        await wait(140);
      } finally {
        pendingWrites = Math.max(0, pendingWrites - 1);
        emitConnection();
      }
    },

    async restorePlan(plan: Plan): Promise<void> {
      const plans = readPlans();
      if (plans.some((item) => item.id === plan.id)) return;
      writePlans([...plans, plan]);
      broadcast();
    },
  };
}
