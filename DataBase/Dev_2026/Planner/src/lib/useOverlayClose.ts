import { useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

/**
 * シートを閉じるときの遷移。
 * アプリ内から開いた場合は履歴を1つ戻り、直接 URL を開いた場合はダッシュボードへ置き換えます。
 */
export function useOverlayClose(fallback = '/') {
  const navigate = useNavigate();
  const location = useLocation();

  return useCallback(() => {
    if (location.key === 'default') navigate(fallback, { replace: true });
    else navigate(-1);
  }, [navigate, location.key, fallback]);
}
