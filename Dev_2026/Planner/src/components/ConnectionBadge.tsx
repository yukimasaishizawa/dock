import type { ConnectionState } from '../data/types';

const LABELS: Record<ConnectionState, { text: string; className: string; hint: string }> = {
  online: {
    text: 'オンライン',
    className: 'conn--online',
    hint: 'Firebase と接続されています',
  },
  syncing: {
    text: '連携中',
    className: 'conn--syncing',
    hint: '予定を Firebase と同期しています',
  },
  offline: {
    text: 'オフライン',
    className: 'conn--offline',
    hint: 'オフラインです。オンラインに戻ると同期されます',
  },
};

export function ConnectionBadge({ state }: { state: ConnectionState }) {
  const config = LABELS[state];
  return (
    <div
      className={`conn ${config.className}`}
      role="status"
      aria-live="polite"
      title={config.hint}
    >
      <span className="conn__dot" aria-hidden="true" />
      <span className="conn__label">{config.text}</span>
    </div>
  );
}
