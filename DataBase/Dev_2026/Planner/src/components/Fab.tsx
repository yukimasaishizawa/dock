import { Link, useLocation } from 'react-router-dom';
import { PlusIcon } from './Icons';

export function Fab() {
  const location = useLocation();
  const open = location.pathname === '/new';

  return (
    <Link
      to="/new"
      className={`fab${open ? ' fab--open' : ''}`}
      aria-label="新しい予定を追加"
      aria-expanded={open}
    >
      <PlusIcon size={22} className="fab__plus" />
      <span className="fab__label">予定を追加</span>
    </Link>
  );
}
