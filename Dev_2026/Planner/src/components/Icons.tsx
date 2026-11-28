import type { ReactNode, SVGProps } from 'react';

type IconProps = Omit<SVGProps<SVGSVGElement>, 'children'> & { size?: number };

function Svg({ size = 16, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const PlusIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);

export const CloseIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Svg>
);

export const ChevronLeftIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="m15 18-6-6 6-6" />
  </Svg>
);

export const ChevronRightIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="m9 18 6-6-6-6" />
  </Svg>
);

export const ArrowRightIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </Svg>
);

export const TrashIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M4 7h16M10 11v6M14 11v6M6.5 7l.8 12.2a1 1 0 0 0 1 .8h7.4a1 1 0 0 0 1-.8L17.5 7M9 7V4.6a.6.6 0 0 1 .6-.6h4.8a.6.6 0 0 1 .6.6V7" />
  </Svg>
);

export const CalendarIcon = (props: IconProps) => (
  <Svg {...props}>
    <rect x="3.2" y="5" width="17.6" height="15.8" rx="2.4" />
    <path d="M8 3v4M16 3v4M3.2 10.5h17.6" />
  </Svg>
);

export const UsersIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="9.2" cy="8" r="3.2" />
    <path d="M3.4 19.6c0-3 2.6-5.2 5.8-5.2s5.8 2.2 5.8 5.2" />
    <path d="M16.2 5.6a3 3 0 0 1 0 5.4M17.4 19.6c0-1.9-.5-3.3-1.5-4.3" />
  </Svg>
);

export const SparkIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M12 3.4l1.7 4.9 4.9 1.7-4.9 1.7L12 16.6l-1.7-4.9L5.4 10l4.9-1.7z" />
    <path d="M18.4 16.2l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
  </Svg>
);

export const InfoIcon = (props: IconProps) => (
  <Svg {...props}>
    <circle cx="12" cy="12" r="8.8" />
    <path d="M12 11.2v5M12 7.9h.01" />
  </Svg>
);

export const LockIcon = (props: IconProps) => (
  <Svg {...props}>
    <rect x="4.6" y="10.2" width="14.8" height="10.2" rx="2.4" />
    <path d="M8.4 10.2V7.6a3.6 3.6 0 0 1 7.2 0v2.6" />
  </Svg>
);

export const EyeIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M2.6 12S6 5.8 12 5.8 21.4 12 21.4 12 18 18.2 12 18.2 2.6 12 2.6 12Z" />
    <circle cx="12" cy="12" r="2.8" />
  </Svg>
);

export const EyeOffIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M4 4l16 16" />
    <path d="M9.9 5.2A9.6 9.6 0 0 1 12 5c6 0 9.4 6.2 9.4 6.2a17 17 0 0 1-2.7 3.5M6.4 7.2A16.7 16.7 0 0 0 2.6 12S6 18.2 12 18.2c1.2 0 2.3-.2 3.3-.7" />
    <path d="M9.6 9.8a3 3 0 0 0 4.2 4.2" />
  </Svg>
);

export const UndoIcon = (props: IconProps) => (
  <Svg {...props}>
    <path d="M4 9h9.4a6 6 0 0 1 0 12H9" />
    <path d="M7.6 5.2 4 9l3.6 3.8" />
  </Svg>
);
