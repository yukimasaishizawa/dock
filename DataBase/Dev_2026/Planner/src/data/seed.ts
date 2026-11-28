import { addDaysKey, todayKey } from '../lib/date';
import type { FamilyMember, Plan } from './types';

interface SeedSpec {
  offset: number;
  title: string;
  start: string | null;
  end: string | null;
  /** 家族の並び順（0 始まり）。人数が足りない場合は自動で丸めます。 */
  members: number[];
}

const SPECS: SeedSpec[] = [
  { offset: 0, title: '小学校の運動会', start: '09:00', end: '12:00', members: [2, 3] },
  { offset: 0, title: '夕食の買い出し', start: '18:30', end: null, members: [1] },
  { offset: 1, title: 'ピアノの練習', start: '16:00', end: '17:00', members: [3] },
  { offset: 3, title: '家族で映画', start: '14:00', end: '16:30', members: [0, 1, 2, 3] },
  { offset: 6, title: '祖父母の家へ', start: null, end: null, members: [0, 1, 2, 3] },
  { offset: 9, title: '定期健診', start: '10:30', end: '11:30', members: [1] },
];

/**
 * 初回起動時だけ入れるサンプル予定。デザインとカレンダーの見え方を確認するためのもので、
 * 一度でも予定が存在すれば投入されません（全削除しても再投入されません）。
 */
export function buildSeedPlans(members: FamilyMember[]): Plan[] {
  if (members.length === 0) return [];
  const author = members[0];
  const today = todayKey();

  return SPECS.flatMap((spec, index) => {
    const memberIds = spec.members
      .map((position) => members[position % members.length])
      .filter((member, position, list) => list.findIndex((item) => item.id === member.id) === position)
      .map((member) => member.id);

    if (memberIds.length === 0) return [];

    return [
      {
        id: `seed_${index}`,
        groupId: `seed_group_${index}`,
        title: spec.title,
        date: addDaysKey(today, spec.offset),
        startTime: spec.start,
        endTime: spec.end,
        memberIds,
        createdById: author.id,
        createdByName: author.name,
        createdAt: Date.now() + index,
      },
    ];
  });
}
