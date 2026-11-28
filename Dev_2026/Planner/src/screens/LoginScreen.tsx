import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlanner } from '../app/PlannerProvider';
import { BrandMark } from '../components/BrandMark';
import {
  ArrowRightIcon,
  CalendarIcon,
  EyeIcon,
  EyeOffIcon,
  InfoIcon,
  LockIcon,
  SparkIcon,
  UsersIcon,
} from '../components/Icons';
import { backend } from '../data/backend';
import type { FamilyMember } from '../data/types';

export function LoginScreen() {
  const { members, familySource, signIn } = usePlanner();
  const navigate = useNavigate();

  const [selected, setSelected] = useState<FamilyMember | null>(null);
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const passwordRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!selected) return;
    const timer = window.setTimeout(() => passwordRef.current?.focus(), 300);
    return () => window.clearTimeout(timer);
  }, [selected]);

  const back = () => {
    setSelected(null);
    setPassword('');
    setError(null);
    setShowPassword(false);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected || busy) return;

    setBusy(true);
    setError(null);
    try {
      await signIn(selected, password);
      navigate('/', { replace: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'ログインできませんでした。');
      setBusy(false);
    }
  };

  return (
    <main className="login">
      <div className="login__glow" aria-hidden="true" />
      <div className="login__card">
        <section className="login__brand">
          <div className="login__brandmark">
            <BrandMark />
            <span>Planner</span>
          </div>

          <h1 className="login__headline">
            家族の予定を、
            <br />
            <em>ひとつの場所</em>に。
          </h1>
          <p className="login__lead">
            今日と明日、そして月のカレンダー。Planner は家族みんなの予定を見渡すための場所です。
          </p>

          <ul className="login__points">
            <li>
              <CalendarIcon size={15} />
              今日・明日・月の予定をひと目で
            </li>
            <li>
              <UsersIcon size={15} />
              だれの予定かがわかる対象者表示
            </li>
            <li>
              <SparkIcon size={15} />
              スマホのホーム画面に追加してアプリのように
            </li>
          </ul>
        </section>

        <section className="login__panel">
          {selected ? (
            <div className="pwd">
              <div>
                <p className="eyebrow">パスワード</p>
                <h2 className="login__title">おかえりなさい</h2>
                <p className="login__sub">パスワードを入力してください。</p>
              </div>

              <div className="pwd__who">
                <span
                  className="avatar avatar--lg"
                  style={{ '--c': selected.color } as CSSProperties}
                  aria-hidden="true"
                >
                  {selected.initial}
                </span>
                <span className="pwd__name">
                  <b>{selected.name}</b>
                  <span>{selected.email}</span>
                </span>
              </div>

              <form className="pwd__form" onSubmit={submit} noValidate>
                <div className="field">
                  <label className="field__label" htmlFor="password">
                    パスワード
                    <span className="field__req">必須</span>
                  </label>
                  <div className="pwd__field">
                    <input
                      id="password"
                      ref={passwordRef}
                      className="input"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder="パスワードを入力"
                      autoComplete="current-password"
                      aria-invalid={Boolean(error)}
                    />
                    <button
                      type="button"
                      className="pwd__toggle"
                      onClick={() => setShowPassword((value) => !value)}
                      aria-label={showPassword ? 'パスワードを隠す' : 'パスワードを表示'}
                    >
                      {showPassword ? <EyeOffIcon size={17} /> : <EyeIcon size={17} />}
                    </button>
                  </div>
                  {error ? (
                    <p className="field__error" role="alert">
                      {error}
                    </p>
                  ) : null}
                </div>

                <button
                  type="submit"
                  className="btn btn--primary btn--block"
                  disabled={busy || password.length === 0}
                >
                  {busy ? (
                    <>
                      <span className="spinner" />
                      確認中…
                    </>
                  ) : (
                    <>
                      <LockIcon size={15} />
                      ログイン
                    </>
                  )}
                </button>
              </form>

              <button type="button" className="pwd__back" onClick={back}>
                ← 別のユーザーを選ぶ
              </button>
            </div>
          ) : (
            <div>
              <p className="eyebrow">ログイン</p>
              <h2 className="login__title">だれが使いますか？</h2>
              <p className="login__sub">家族のアカウントを選んでください。</p>

              <div className="userlist">
                {members.map((member, index) => (
                  <button
                    type="button"
                    key={member.id}
                    className="userrow"
                    style={{ '--i': index } as CSSProperties}
                    onClick={() => {
                      setSelected(member);
                      setPassword('');
                      setError(null);
                    }}
                  >
                    <span
                      className="avatar avatar--lg"
                      style={{ '--c': member.color } as CSSProperties}
                      aria-hidden="true"
                    >
                      {member.initial}
                    </span>
                    <span className="userrow__text">
                      <span className="userrow__name">{member.name}</span>
                      <span className="userrow__mail">{member.email}</span>
                    </span>
                    <ArrowRightIcon size={16} className="userrow__arrow" />
                  </button>
                ))}

                {members.length === 0 ? (
                  <p className="login__sub">
                    家族の情報を読み込めませんでした。user/users.csv を確認してください。
                  </p>
                ) : null}
              </div>

              <p className="login__hint">
                <InfoIcon size={13} />
                {backend.id === 'mock'
                  ? 'いまはモック認証です。任意のパスワードでログインできます。'
                  : 'パスワードは Firebase に登録されたものを使用します。わからないときは家族の管理者に確認してください。'}
              </p>
            </div>
          )}

          <p className="login__foot">
            <span>© Planner</span>
            <span>
              {familySource === 'csv'
                ? 'user/users.csv を読み込みました'
                : 'サンプルの家族情報を表示中'}
            </span>
          </p>
        </section>
      </div>
    </main>
  );
}
