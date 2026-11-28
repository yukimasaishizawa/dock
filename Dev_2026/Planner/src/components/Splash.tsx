import { useState } from 'react';

export function Splash() {
  const [failed, setFailed] = useState(false);

  return (
    <div className="splash">
      <div className="splash__inner">
        <div className="splash__logo">
          {failed ? (
            'P'
          ) : (
            <img
              src={`${import.meta.env.BASE_URL}image/logo.png`}
              alt=""
              onError={() => setFailed(true)}
            />
          )}
        </div>
        <span className="splash__word">Planner</span>
        <span className="splash__bar" aria-hidden="true" />
      </div>
    </div>
  );
}
