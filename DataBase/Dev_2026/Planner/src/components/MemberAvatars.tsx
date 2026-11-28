import type { CSSProperties } from 'react';
import type { FamilyMember } from '../data/types';

interface Props {
  memberIds: string[];
  members: Map<string, FamilyMember>;
  max?: number;
}

export function MemberAvatars({ memberIds, members, max = 3 }: Props) {
  const resolved = memberIds
    .map((id) => members.get(id))
    .filter((member): member is FamilyMember => Boolean(member));
  const shown = resolved.slice(0, max);
  const rest = resolved.length - shown.length;
  const label = resolved.map((member) => member.name).join('、');

  return (
    <span
      className="people"
      title={label ? `対象: ${label}` : undefined}
      aria-label={label ? `対象: ${label}` : '対象者が未設定'}
    >
      {shown.map((member) => (
        <span
          key={member.id}
          className="avatar avatar--xs"
          style={{ '--c': member.color } as CSSProperties}
          aria-hidden="true"
        >
          {member.initial}
        </span>
      ))}
      {rest > 0 ? (
        <span className="avatar avatar--xs avatar--more" aria-hidden="true">
          +{rest}
        </span>
      ) : null}
    </span>
  );
}
