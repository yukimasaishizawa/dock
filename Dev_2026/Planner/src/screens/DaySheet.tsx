import { useMemo, type CSSProperties } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { usePlanner } from '../app/PlannerProvider';
import { useToast } from '../app/ToastProvider';
import { EmptyState } from '../components/EmptyState';
import { MemberAvatars } from '../components/MemberAvatars';
import { PlusIcon, TrashIcon } from '../components/Icons';
import { Sheet } from '../components/Sheet';
import {
  formatJP,
  fromKey,
  isValidDateKey,
  plansOn,
  relativeLabel,
  todayKey,
} from '../lib/date';
import { useOverlayClose } from '../lib/useOverlayClose';
import type { Plan } from '../data/types';

export function DaySheet() {
  const { date } = useParams<{ date: string }>();
  const navigate = useNavigate();
  const { plans, membersById, deletePlan, restorePlan } = usePlanner();
  const { push } = useToast();
  const close = useOverlayClose('/');

  const key = isValidDateKey(date) ? date : todayKey();
  const dayPlans = useMemo(() => plansOn(plans, key), [plans, key]);
  const dateObject = fromKey(key);
  const relative = relativeLabel(key);

  const openNewForDay = () => {
    navigate(`/new?date=${key}&from=${encodeURIComponent(`/day/${key}`)}`);
  };

  const remove = async (plan: Plan) => {
    await deletePlan(plan.id);
    push({
      message: `「${plan.title}」を削除しました`,
      actionLabel: '元に戻す',
      onAction: () => {
        void restorePlan(plan);
      },
    });
  };

  return (
    <Sheet
      eyebrow="スケジュール"
      title={relative ? `${relative}の予定` : `${formatJP(key)}の予定`}
      onClose={close}
      footer={
        <button type="button" className="btn btn--primary btn--block" onClick={openNewForDay}>
          <PlusIcon size={16} />
          この日に予定を追加
        </button>
      }
    >
      <div className="dayhead">
        <span className="dayhead__num">{dateObject.getDate()}</span>
        <span className="dayhead__count">{dayPlans.length}件の予定</span>
      </div>

      {dayPlans.length === 0 ? (
        <EmptyState cta="予定を追加" onCta={openNewForDay}>
          この日の予定はまだありません。
        </EmptyState>
      ) : (
        <div>
          {dayPlans.map((plan, index) => (
            <div className="dayrow" key={plan.id} style={{ '--i': index } as CSSProperties}>
              <div className="dayrow__main">
                <div className="dayrow__top">
                  <span className={`dayrow__time${plan.startTime ? '' : ' dayrow__time--all'}`}>
                    {plan.startTime
                      ? plan.endTime
                        ? `${plan.startTime}–${plan.endTime}`
                        : plan.startTime
                      : '終日'}
                  </span>
                  <span className="dayrow__title">{plan.title}</span>
                </div>
                <div className="dayrow__meta">
                  <MemberAvatars memberIds={plan.memberIds} members={membersById} />
                  <span className="dayrow__author">
                    <b>{plan.createdByName}</b>が追加
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="iconbtn iconbtn--danger"
                onClick={() => void remove(plan)}
                aria-label={`「${plan.title}」を削除`}
              >
                <TrashIcon size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </Sheet>
  );
}
