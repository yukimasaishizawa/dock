import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePlanner } from '../app/PlannerProvider';
import { BrandMark } from './BrandMark';
import { ConnectionBadge } from './ConnectionBadge';

export function AppHeader() {
  const navigate = useNavigate();
  const { connection } = usePlanner();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 6);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header className={`app-header${scrolled ? ' is-scrolled' : ''}`}>
      <div className="app-header__inner">
        <button
          type="button"
          className="brand"
          onClick={() => {
            navigate('/');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
          aria-label="Planner のダッシュボードへ戻る"
        >
          <BrandMark />
          <span className="brand__word">Planner</span>
        </button>

        <ConnectionBadge state={connection} />
      </div>
    </header>
  );
}
