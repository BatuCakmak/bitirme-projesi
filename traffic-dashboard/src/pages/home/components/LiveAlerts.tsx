import { liveAlerts } from '@/mocks/traffic';
import { useThemeColors } from '@/hooks/useThemeColors';

const alertConfig = {
  accident: { icon: 'ri-alarm-warning-line', color: 'text-[#ff4757]', bg: 'bg-[#ff4757]/10', border: 'border-[#ff4757]/30' },
  congestion: { icon: 'ri-error-warning-line', color: 'text-[#ffd700]', bg: 'bg-[#ffd700]/10', border: 'border-[#ffd700]/30' },
  info: { icon: 'ri-information-line', color: 'text-[#00d4ff]', bg: 'bg-[#00d4ff]/10', border: 'border-[#00d4ff]/30' },
};

export default function LiveAlerts() {
  const t = useThemeColors();

  return (
    <div className={`${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
      <div className={`flex items-center justify-between px-4 py-3 border-b ${t.border}`}>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#ff4757] animate-pulse"></div>
          <span className={`${t.textPrimary} text-sm font-semibold`}>Canlı Uyarılar</span>
        </div>
        <span className={`${t.textSecondary} text-xs`}>{liveAlerts.length} bildirim</span>
      </div>
      <div className={`divide-y ${t.divider}`}>
        {liveAlerts.map((alert) => {
          const cfg = alertConfig[alert.type as keyof typeof alertConfig];
          return (
            <div key={alert.id} className={`flex items-start gap-3 px-4 py-3 ${cfg.bg} border-l-2 ${cfg.border}`}>
              <div className="w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                <i className={`${cfg.icon} text-sm ${cfg.color}`}></i>
              </div>
              <div className="flex-1 min-w-0">
                <p className={`${t.textPrimary} text-xs leading-relaxed`}>{alert.message}</p>
                <p className={`${t.textSecondary} text-[10px] mt-0.5`}>{alert.time}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
