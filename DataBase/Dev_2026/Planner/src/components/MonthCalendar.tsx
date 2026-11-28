import { useMemo, useState, type CSSProperties } from 'react';
import { fromKey, formatJP, monthGrid, todayKey, toKey, WEEKDAYS_JP } from '../lib/date';
import { useMediaQuery } from '../lib/useMediaQuery';
import type { FamilyMember, Plan } from '../data/types';
import { ChevronLeftIcon, ChevronRightIcon } from './Icons';

interface Props {
  plansByDate: Map<string, Plan[]>;
  members: FamilyMember[];
  onSelectDate: (key: string) => void;
}

export function MonthCalendar({ plansByDate, members, onSelectDate }: Props) {
  const today = todayKey();
  const [cursor, setCursor] = useState(() => {
    const date = fromKey(today);
    return { year: date.getFullYear(), month: date.getMonth() };
  });

  const wide = useMediaQuery('(min-width: 720px)');
  const tiny = useMediaQuery('(max-width: 420px)');
  const maxTiles = wide ? 3 : tiny ? 1 : 2;

  const cells = useMemo(() => monthGrid(cursor.year, cursor.month), [cursor.year, cursor.month]);
  const memberById = useMemo(() => new Map(members.map((member) => [member.id, member])), [members]);

  const shiftMonth = (delta: number) => {
    setCursor((current) => {
      const date = new Date(current.year, current.month + delta, 1);
      return { year: date.getFullYear(), month: date.getMonth() };
    });
  };

  return (
    <div className="cal">
      <div className="cal__head">
        <div className="cal__month">
          <span className="cal__monthnum">{cursor.month + 1}</span>
          <span className="cal__unit">月</span>
          <span className="cal__year">{cursor.year}</span>
        </div>
        <div className="cal__nav">
          <button
            type="button"
            className="iconbtn"
            onClick={() => shiftMonth(-1)}
            aria-label="前の月"
          >
            <ChevronLeftIcon size={17} />
          </button>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => {
              const date = fromKey(today);
              setCursor({ year: date.getFullYear(), month: date.getMonth() });
            }}
          >
            今日
          </button>
          <button
            type="button"
            className="iconbtn"
            onClick={() => shiftMonth(1)}
            aria-label="次の月"
          >
            <ChevronRightIcon size={17} />
          </button>
        </div>
      </div>

      <div className="cal__weekdays" aria-hidden="true">
        {WEEKDAYS_JP.map((weekday) => (
          <span key={weekday}>{weekday}</span>
        ))}
      </div>

      <div className="cal__grid" role="group" aria-label={`${cursor.year}年${cursor.month + 1}月のカレンダー`}>
        {cells.map((date) => {
          const key = toKey(date);
          const dayPlans = plansByDate.get(key) ?? [];
          const inMonth = date.getMonth() === cursor.month;
          const isToday = key === today;
          const hidden = Math.max(0, dayPlans.length - maxTiles);

          const className = [
            'cal__cell',
            inMonth ? '' : 'cal__cell--muted',
            isToday ? 'cal__cell--today' : '',
          ]
            .filter(Boolean)
            .join(' ');

          return (
            <button
              type="button"
              key={key}
              className={className}
              onClick={() => onSelectDate(key)}
              aria-label={`${formatJP(key)} ${dayPlans.length}件の予定`}
            >
              <span className="cal__num">{date.getDate()}</span>
              <span className="cal__tiles">
                {dayPlans.slice(0, maxTiles).map((plan) => {
                  const owner = plan.memberIds
                    .map((id) => memberById.get(id))
                    .find((member) => Boolean(member));
                  return (
                    <span
                      key={plan.id}
                      className="cal__tile"
                      style={{ '--c': owner?.color ?? 'var(--brand)' } as CSSProperties}
                    >
                      <i aria-hidden="true" />
                      <span className="cal__tile__label">{plan.title}</span>
                    </span>
                  );
                })}
              </span>
              {hidden > 0 ? <span className="cal__more">+{hidden}</span> : null}
            </button>
          );
        })}
      </div>

      {members.length > 0 ? (
        <div className="cal__legend">
          {members.map((member) => (
            <span key={member.id}>
              <i style={{ '--c': member.color } as CSSProperties} aria-hidden="true" />
              {member.name}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
