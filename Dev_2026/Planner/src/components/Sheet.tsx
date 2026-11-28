import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { CloseIcon } from './Icons';

interface Props {
  eyebrow?: string;
  title: string;
  sub?: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}

const EXIT_DURATION = 250;

export function Sheet({ eyebrow, title, sub, onClose, children, footer }: Props) {
  const [closing, setClosing] = useState(false);
  const closingRef = useRef(false);
  const sheetRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<number | undefined>(undefined);
  const onCloseRef = useRef(onClose);

  onCloseRef.current = onClose;

  const close = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    setClosing(true);
    timerRef.current = window.setTimeout(() => onCloseRef.current(), EXIT_DURATION);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        close();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [close]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previous;
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    sheetRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <div className={`sheet-layer${closing ? ' is-closing' : ''}`}>
      <div className="sheet-backdrop" onClick={close} aria-hidden="true" />
      <div
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="sheet-title"
        tabIndex={-1}
        ref={sheetRef}
      >
        <div className="sheet__grab" aria-hidden="true" />
        <header className="sheet__head">
          <div className="sheet__headmain">
            {eyebrow ? <p className="sheet__eyebrow">{eyebrow}</p> : null}
            <h2 className="sheet__title" id="sheet-title">
              {title}
            </h2>
            {sub ? <p className="sheet__sub">{sub}</p> : null}
          </div>
          <button
            type="button"
            className="iconbtn sheet__close"
            onClick={close}
            aria-label="閉じる"
          >
            <CloseIcon size={17} />
          </button>
        </header>

        <div className="sheet__body">{children}</div>

        {footer ? <footer className="sheet__foot">{footer}</footer> : null}
      </div>
    </div>
  );
}
