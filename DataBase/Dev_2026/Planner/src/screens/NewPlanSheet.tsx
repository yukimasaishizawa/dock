import { useMemo, useState, type CSSProperties, type FormEvent } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { usePlanner } from '../app/PlannerProvider';
import { useToast } from '../app/ToastProvider';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  InfoIcon,
  PlusIcon,
} from '../components/Icons';
import { Sheet } from '../components/Sheet';
import {
  addDaysKey,
  formatJP,
  fromKey,
  isValidDateKey,
  monthGrid,
  monthLabel,
  todayKey,
  toKey,
  WEEKDAYS_JP,
} from '../lib/date';
import { useOverlayClose } from '../lib/useOverlayClose';

interface FormErrors {
  title?: string;
  dates?: string;
  members?: string;
  time?: string;
}

export function NewPlanSheet() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { members, session, createPlans } = usePlanner();
  const { push } = useToast();
  const close = useOverlayClose('/');

  const dateParam = searchParams.get('date') ?? '';
  const fromParam = searchParams.get('from') ?? '';
  const initialDate = isValidDateKey(dateParam) ? dateParam : todayKey();

  const [title, setTitle] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [dates, setDates] = useState<string[]>([initialDate]);
  const [memberIds, setMemberIds] = useState<string[]>([]);
  const [cursor, setCursor] = useState(() => {
    const base = fromKey(initialDate);
    return { year: base.getFullYear(), month: base.getMonth() };
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);

  const cells = useMemo(() => monthGrid(cursor.year, cursor.month), [cursor.year, cursor.month]);
  const today = todayKey();

  const toggleDate = (key: string) => {
    setDates((current) =>
      current.includes(key)
        ? current.filter((item) => item !== key)
        : [...current, key].sort(),
    );
    setErrors((current) => ({ ...current, dates: undefined }));
  };

  const selectDate = (key: string) => {
    setDates((current) => (current.includes(key) ? current : [...current, key].sort()));
    setErrors((current) => ({ ...current, dates: undefined }));
  };

  const toggleMember = (id: string) => {
    setMemberIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
    setErrors((current) => ({ ...current, members: undefined }));
  };

  const shiftMonth = (delta: number) => {
    setCursor((current) => {
      const next = new Date(current.year, current.month + delta, 1);
      return { year: next.getFullYear(), month: next.getMonth() };
    });
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting) return;

    const nextErrors: FormErrors = {};
    if (!title.trim()) nextErrors.title = '予定の名称を入力してください。';
    if (dates.length === 0) nextErrors.dates = '日にちを1つ以上選んでください。';
    if (memberIds.length === 0) nextErrors.members = '対象者を1人以上選んでください。';
    if (startTime && endTime && endTime < startTime) {
      nextErrors.time = '終了時刻は開始時刻より後にしてください。';
    }

    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;

    setSubmitting(true);
    try {
      const created = await createPlans({
        title: title.trim(),
        dates: [...dates].sort(),
        startTime: startTime || null,
        endTime: endTime || null,
        memberIds,
      });
      push({ message: `${created.length}日分の予定を追加しました` });
      navigate(fromParam.startsWith('/') ? fromParam : '/', { replace: true });
    } catch {
      setErrors({ title: '登録できませんでした。通信状態を確認して、もう一度お試しください。' });
      setSubmitting(false);
    }
  };

  return (
    <Sheet
      eyebrow="新しい予定"
      title="予定を追加"
      sub="日にちをまとめて選ぶと、一括で登録できます。"
      onClose={close}
      footer={
        <div className="form__foot">
          <span className="form__count">
            <b>{dates.length}</b>日 · <b>{memberIds.length}</b>人
          </span>
          <button
            type="submit"
            form="new-plan-form"
            className="btn btn--primary"
            disabled={submitting}
          >
            {submitting ? (
              <>
                <span className="spinner" />
                登録中…
              </>
            ) : (
              <>
                <PlusIcon size={16} />
                予定を登録
              </>
            )}
          </button>
        </div>
      }
    >
      <form id="new-plan-form" className="form" onSubmit={submit} noValidate>
        <div className="field">
          <label className="field__label" htmlFor="plan-title">
            予定の名称
            <span className="field__req">必須</span>
          </label>
          <input
            id="plan-title"
            className="input input--lg"
            value={title}
            onChange={(event) => {
              setTitle(event.target.value);
              setErrors((current) => ({ ...current, title: undefined }));
            }}
            placeholder="例) 小学校の運動会"
            autoComplete="off"
            maxLength={60}
            aria-invalid={Boolean(errors.title)}
            autoFocus
          />
          {errors.title ? (
            <p className="field__error" role="alert">
              {errors.title}
            </p>
          ) : null}
        </div>

        <div className="field">
          <span className="field__label">
            時間帯
            <span className="field__opt">任意</span>
          </span>
          <div className="time-range">
            <input
              type="time"
              className="input input--time"
              value={startTime}
              onChange={(event) => {
                setStartTime(event.target.value);
                setErrors((current) => ({ ...current, time: undefined }));
              }}
              aria-label="開始時刻"
            />
            <span className="time-range__dash" aria-hidden="true">
              –
            </span>
            <input
              type="time"
              className="input input--time"
              value={endTime}
              onChange={(event) => {
                setEndTime(event.target.value);
                setErrors((current) => ({ ...current, time: undefined }));
              }}
              aria-label="終了時刻"
            />
          </div>
          <p className="field__hint">
            空欄のままでも登録できます。その場合は「終日」の予定になります。
          </p>
          {errors.time ? (
            <p className="field__error" role="alert">
              {errors.time}
            </p>
          ) : null}
        </div>

        <div className="field">
          <span className="field__label">
            日にち
            <span className="field__req">必須</span>
            <span className="field__count">{dates.length}日選択中</span>
          </span>

          <div className="datepick">
            <div className="datepick__head">
              <span className="datepick__month">{monthLabel(cursor.year, cursor.month)}</span>
              <div className="datepick__nav">
                <button
                  type="button"
                  className="iconbtn"
                  onClick={() => shiftMonth(-1)}
                  aria-label="前の月"
                >
                  <ChevronLeftIcon size={16} />
                </button>
                <button
                  type="button"
                  className="iconbtn"
                  onClick={() => shiftMonth(1)}
                  aria-label="次の月"
                >
                  <ChevronRightIcon size={16} />
                </button>
              </div>
            </div>

            <div className="datepick__grid">
              {WEEKDAYS_JP.map((weekday) => (
                <span key={weekday} className="datepick__weekday">
                  {weekday}
                </span>
              ))}
              {cells.map((date) => {
                const key = toKey(date);
                const inMonth = date.getMonth() === cursor.month;
                const selected = dates.includes(key);
                const isToday = key === today;
                const className = [
                  'daypick',
                  inMonth ? '' : 'daypick--muted',
                  isToday ? 'daypick--today' : '',
                  selected ? 'daypick--on' : '',
                ]
                  .filter(Boolean)
                  .join(' ');

                return (
                  <button
                    type="button"
                    key={key}
                    className={className}
                    onClick={() => toggleDate(key)}
                    aria-pressed={selected}
                    aria-label={formatJP(key)}
                  >
                    {date.getDate()}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="datepick__quick">
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => selectDate(today)}
            >
              今日
            </button>
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => selectDate(addDaysKey(today, 1))}
            >
              明日
            </button>
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => {
                setDates([]);
                setErrors((current) => ({ ...current, dates: undefined }));
              }}
            >
              選択をクリア
            </button>
          </div>

          {dates.length > 0 ? (
            <div className="tags">
              {dates.map((key) => (
                <span className="tag" key={key}>
                  {formatJP(key)}
                  <button
                    type="button"
                    onClick={() => toggleDate(key)}
                    aria-label={`${formatJP(key)} を選択から外す`}
                  >
                    <CloseIcon size={11} />
                  </button>
                </span>
              ))}
            </div>
          ) : null}

          {errors.dates ? (
            <p className="field__error" role="alert">
              {errors.dates}
            </p>
          ) : null}
        </div>

        <div className="field">
          <span className="field__label">
            対象者
            <span className="field__req">必須</span>
            <span className="field__count">{memberIds.length}人選択中</span>
          </span>

          <div className="chips">
            {members.map((member) => {
              const on = memberIds.includes(member.id);
              return (
                <button
                  type="button"
                  key={member.id}
                  className={`chip chip--member${on ? ' chip--on' : ''}`}
                  onClick={() => toggleMember(member.id)}
                  aria-pressed={on}
                  style={{ '--c': member.color } as CSSProperties}
                >
                  <span className="chip__avatar" aria-hidden="true">
                    {member.initial}
                  </span>
                  {member.name}
                </button>
              );
            })}
          </div>

          {errors.members ? (
            <p className="field__error" role="alert">
              {errors.members}
            </p>
          ) : (
            <p className="field__hint">対象者に選んだ家族の予定として表示されます。</p>
          )}
        </div>

        <p className="form__note">
          <InfoIcon size={14} />
          <span>
            登録者として <b>{session?.userName ?? 'ログイン中のユーザー'}</b>{' '}
            が記録されます。登録した予定は家族全員が閲覧できます。
          </span>
        </p>
      </form>
    </Sheet>
  );
}
