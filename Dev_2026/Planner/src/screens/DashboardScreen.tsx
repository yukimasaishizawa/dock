import { useMemo, type CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlanner } from '../app/PlannerProvider';
import { EmptyState } from '../components/EmptyState';
import { MonthCalendar } from '../components/MonthCalendar';
import { PlanCard } from '../components/PlanCard';
import {
  addDaysKey,
  formatJP,
  groupPlansByDate,
  plansOn,
  todayKey,
} from '../lib/date';

export function DashboardScreen() {
  const { plans, members, membersById, session, signOut } = usePlanner();
  const navigate = useNavigate();

  const today = todayKey();
  const tomorrow = addDaysKey(today, 1);
  const todayPlans = plansOn(plans, today);
  const tomorrowPlans = plansOn(plans, tomorrow);
  const plansByDate = useMemo(() => groupPlansByDate(plans), [plans]);
  const signedInMember = session ? membersById.get(session.userId) : undefined;

  const openNew = () => navigate('/new');

  return (
    <main className="app-main">
      <section className="section" style={{ '--i': 0 } as CSSProperties}>
        <header className="section__head">
          <span className="section__index">01</span>
          <h2 className="section__title">今日</h2>
          <span className="section__meta">
            {formatJP(today)} · <b>{todayPlans.length}</b>件
          </span>
        </header>

        {todayPlans.length > 0 ? (
          <div className="plan-list">
            {todayPlans.map((plan, index) => (
              <PlanCard
                key={plan.id}
                plan={plan}
                members={membersById}
                hero
                index={index}
              />
            ))}
          </div>
        ) : (
          <EmptyState cta="予定を追加" onCta={openNew}>
            今日の予定はまだありません。空いた時間に、なにか入れておきますか？
          </EmptyState>
        )}
      </section>

      <section className="section" style={{ '--i': 1 } as CSSProperties}>
        <header className="section__head">
          <span className="section__index">02</span>
          <h2 className="section__title">明日</h2>
          <span className="section__meta">
            {formatJP(tomorrow)} · <b>{tomorrowPlans.length}</b>件
          </span>
        </header>

        {tomorrowPlans.length > 0 ? (
          <div className="plan-list">
            {tomorrowPlans.map((plan, index) => (
              <PlanCard key={plan.id} plan={plan} members={membersById} index={index} />
            ))}
          </div>
        ) : (
          <EmptyState cta="予定を追加" onCta={openNew}>
            明日の予定はまだありません。
          </EmptyState>
        )}
      </section>

      <section className="section" style={{ '--i': 2 } as CSSProperties}>
        <header className="section__head">
          <span className="section__index">03</span>
          <h2 className="section__title">カレンダー</h2>
          <span className="section__meta">日付をタップして詳細</span>
        </header>

        <MonthCalendar
          plansByDate={plansByDate}
          members={members}
          onSelectDate={(key) => navigate(`/day/${key}`)}
        />
      </section>

      <div className="account-bar">
        <span
          className="avatar"
          style={{ '--c': signedInMember?.color ?? 'var(--brand)' } as CSSProperties}
          aria-hidden="true"
        >
          {signedInMember?.initial ?? 'P'}
        </span>
        <span className="account-bar__text">
          <b>{session?.userName}</b>でログイン中
        </span>
        <button type="button" className="account-bar__out" onClick={() => void signOut()}>
          ログアウト
        </button>
      </div>
    </main>
  );
}
