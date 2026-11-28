import { useState } from 'react';

/**
 * ロゴは `public/image/logo.png`（= GitHub の `image/logo.png`）から読み込みます。
 * まだ配置されていない場合は、同系色の「P」マークにフォールバックします。
 */
export function BrandMark({ className = '' }: { className?: string }) {
  const [failed, setFailed] = useState(false);
  const src = `${import.meta.env.BASE_URL}image/logo.png`;

  if (failed) {
    return (
      <span className={`brand__mark brand__fallback ${className}`.trim()} aria-hidden="true">
        P
      </span>
    );
  }

  return (
    <img
      className={`brand__mark ${className}`.trim()}
      src={src}
      alt=""
      aria-hidden="true"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}
