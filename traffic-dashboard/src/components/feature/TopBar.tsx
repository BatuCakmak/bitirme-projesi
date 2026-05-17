import { useState, useEffect } from 'react';
import { useTheme } from '@/hooks/useTheme';
import { useThemeColors } from '@/hooks/useThemeColors';

interface TopBarProps {
  title: string;
  subtitle?: string;
}

export default function TopBar({ title, subtitle }: TopBarProps) {
  const [time, setTime] = useState(new Date());
  const { isDark, toggleTheme } = useTheme();
  const t = useThemeColors();

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (d: Date) =>
    d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const formatDate = (d: Date) =>
    d.toLocaleDateString('tr-TR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

  return (
    <header className={`h-[60px] ${t.topbarBg} border-b ${t.border} flex items-center justify-between px-6 flex-shrink-0 transition-colors duration-300`}>
      <div>
        <h1 className={`${t.textPrimary} font-bold text-lg leading-tight`} style={{ fontFamily: "'Rajdhani', sans-serif" }}>
          {title}
        </h1>
        {subtitle && <p className={`${t.textSecondary} text-xs`}>{subtitle}</p>}
      </div>

      <div className="flex items-center gap-4">
        {/* Live indicator */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#ff4757] animate-pulse"></div>
          <span className="text-[#ff4757] text-xs font-semibold tracking-widest uppercase">Canlı</span>
        </div>

        {/* Date & Time */}
        <div className="text-right">
          <div className={`${t.textPrimary} text-sm font-mono font-semibold`}>{formatTime(time)}</div>
          <div className={`${t.textSecondary} text-[10px]`}>{formatDate(time)}</div>
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className={`w-9 h-9 flex items-center justify-center rounded-lg ${t.cardBg} border ${t.border} ${t.hoverBg} transition-colors cursor-pointer`}
          title={isDark ? 'Açık Mod' : 'Koyu Mod'}
        >
          <i className={`${isDark ? 'ri-sun-line' : 'ri-moon-line'} ${t.textSecondary} text-base`}></i>
        </button>

        {/* Notifications */}
        <div className="relative cursor-pointer">
          <div className={`w-9 h-9 flex items-center justify-center rounded-lg ${t.cardBg} border ${t.border} ${t.hoverBg} transition-colors`}>
            <i className={`ri-notification-3-line ${t.textSecondary} text-base`}></i>
          </div>
          <div className="absolute -top-1 -right-1 w-4 h-4 bg-[#ff4757] rounded-full flex items-center justify-center">
            <span className="text-white text-[9px] font-bold">3</span>
          </div>
        </div>

        {/* User */}
        <div className="flex items-center gap-2 cursor-pointer">
          <div className={`w-9 h-9 rounded-lg flex items-center justify-center`} style={{ backgroundColor: `${t.accent}20` }}>
            <i className="ri-user-line text-base" style={{ color: t.accent }}></i>
          </div>
          <div>
            <div className={`${t.textPrimary} text-xs font-medium`}>Operatör</div>
            <div className={`${t.textSecondary} text-[10px]`}>Admin</div>
          </div>
        </div>
      </div>
    </header>
  );
}
