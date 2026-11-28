import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { backend } from '../data/backend';
import { buildSeedPlans } from '../data/seed';
import type {
  ConnectionState,
  FamilyMember,
  FamilySource,
  Plan,
  PlanDraft,
  Session,
} from '../data/types';

interface PlannerContextValue {
  ready: boolean;
  members: FamilyMember[];
  membersById: Map<string, FamilyMember>;
  familySource: FamilySource['source'];
  session: Session | null;
  connection: ConnectionState;
  plans: Plan[];
  signIn: (member: FamilyMember, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  createPlans: (draft: PlanDraft) => Promise<Plan[]>;
  deletePlan: (id: string) => Promise<void>;
  restorePlan: (plan: Plan) => Promise<void>;
}

const PlannerContext = createContext<PlannerContextValue | null>(null);

/** StrictMode の二重実行でもサンプル投入を一度だけにするためのフラグ。 */
let seedAttempted = false;

export function PlannerProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [members, setMembers] = useState<FamilyMember[]>([]);
  const [familySource, setFamilySource] = useState<FamilySource['source']>('fallback');
  const [session, setSession] = useState<Session | null>(null);
  const [connection, setConnection] = useState<ConnectionState>('online');
  const [plans, setPlans] = useState<Plan[]>([]);
  const [plansLoaded, setPlansLoaded] = useState(false);

  const plansRef = useRef<Plan[]>([]);
  const membersRef = useRef<FamilyMember[]>([]);

  plansRef.current = plans;
  membersRef.current = members;

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const [family, restored] = await Promise.all([
          backend.listMembers(),
          backend.restoreSession(),
        ]);
        if (cancelled) return;
        setMembers(family.members);
        setFamilySource(family.source);
        setSession(restored);
      } catch {
        if (!cancelled) setMembers([]);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => backend.subscribeConnection(setConnection), []);

  useEffect(() => {
    const unsubscribe = backend.subscribePlans((next) => {
      setPlans(next);
      setPlansLoaded(true);
    });
    return unsubscribe;
  }, []);

  // 初回だけ、空のカレンダーに代わりにサンプル予定を1セット入れます。
  // Firebase（本番）では家族の実データを汚さないよう投入しません。
  useEffect(() => {
    if (backend.id !== 'mock') return;
    if (!ready || !plansLoaded || seedAttempted) return;
    if (members.length === 0 || plansRef.current.length > 0) return;
    seedAttempted = true;
    for (const plan of buildSeedPlans(members)) {
      void backend.restorePlan(plan);
    }
  }, [ready, plansLoaded, members]);

  const signIn = useCallback(async (member: FamilyMember, password: string) => {
    const next = await backend.signIn(member, password);
    setSession(next);
  }, []);

  const signOut = useCallback(async () => {
    await backend.signOut();
    setSession(null);
  }, []);

  const createPlans = useCallback(
    async (draft: PlanDraft) => {
      if (!session) throw new Error('ログインが必要です。');
      return backend.createPlans(draft, session);
    },
    [session],
  );

  const deletePlan = useCallback(async (id: string) => {
    await backend.deletePlan(id);
  }, []);

  const restorePlan = useCallback(async (plan: Plan) => {
    await backend.restorePlan(plan);
  }, []);

  const membersById = useMemo(
    () => new Map(members.map((member) => [member.id, member])),
    [members],
  );

  const value = useMemo<PlannerContextValue>(
    () => ({
      ready,
      members,
      membersById,
      familySource,
      session,
      connection,
      plans,
      signIn,
      signOut,
      createPlans,
      deletePlan,
      restorePlan,
    }),
    [
      ready,
      members,
      membersById,
      familySource,
      session,
      connection,
      plans,
      signIn,
      signOut,
      createPlans,
      deletePlan,
      restorePlan,
    ],
  );

  return <PlannerContext.Provider value={value}>{children}</PlannerContext.Provider>;
}

export function usePlanner(): PlannerContextValue {
  const context = useContext(PlannerContext);
  if (!context) throw new Error('usePlanner は PlannerProvider の中で使ってください。');
  return context;
}
