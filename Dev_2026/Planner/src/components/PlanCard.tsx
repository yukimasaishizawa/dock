import type { CSSProperties } from 'react';
import { timeLabel } from '../lib/date';
import type { FamilyMember, Plan } from '../data/types';
import { MemberAvatars } from './MemberAvatars';

interface Props {
  plan: Plan;
  members: Map<string, FamilyMember>;
  hero?: boolean;
  index?: number;
}

export function PlanCard({ plan, members, hero = false, index = 0 }: Props) {
  const time = timeLabel(plan.startTime, plan.endTime);

  return (
    <article
      className={`plan${hero ? ' plan--hero' : ''}`}
      style={{ '--i': index } as CSSProperties}
    >
      {plan.startTime ? (
        <div className="plan__time">
          <span>{time.primary}</span>
          {time.secondary ? <span>{time.secondary}</span> : null}
        </div>
      ) : (
        <div className="plan__time plan__time--all">終日</div>
      )}

      <div className="plan__body">
        <h3 className="plan__title">{plan.title}</h3>
        <div className="plan__meta">
          <MemberAvatars memberIds={plan.memberIds} members={members} />
          <span className="plan__author">
            <b>{plan.createdByName}</b>が追加
          </span>
        </div>
      </div>
    </article>
  );
}
