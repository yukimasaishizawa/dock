import type { ReactNode } from 'react';
import { CalendarIcon } from './Icons';

interface Props {
  children: ReactNode;
  cta?: string;
  onCta?: () => void;
}

export function EmptyState({ children, cta, onCta }: Props) {
  return (
    <div className="empty">
      <span className="empty__glyph" aria-hidden="true">
        <CalendarIcon size={15} />
      </span>
      <span>{children}</span>
      {cta && onCta ? (
        <button type="button" className="empty__cta" onClick={onCta}>
          {cta}
        </button>
      ) : null}
    </div>
  );
}
