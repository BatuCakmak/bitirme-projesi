import { useState } from 'react';
import { cameras } from '@/mocks/cameras';
import { useThemeColors } from '@/hooks/useThemeColors';

const densityColors = {
  high: '#ff4757',
  medium: '#ffd700',
  low: '#00ff88',
};

export default function TrafficMap() {
  const [hoveredCam, setHoveredCam] = useState<string | null>(null);
  const t = useThemeColors();

  const minLat = 40.96, maxLat = 41.10;
  const minLng = 28.80, maxLng = 29.10;
  const W = 700, H = 320;

  const toX = (lng: number) => ((lng - minLng) / (maxLng - minLng)) * W;
  const toY = (lat: number) => H - ((lat - minLat) / (maxLat - minLat)) * H;

  return (
    <div className={`${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
      <div className={`flex items-center justify-between px-4 py-3 border-b ${t.border}`}>
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 flex items-center justify-center">
            <i className="ri-map-2-line text-sm" style={{ color: t.accent }}></i>
          </div>
          <span className={`${t.textPrimary} text-sm font-semibold`}>İstanbul Trafik Haritası</span>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#00ff88]"></div><span className={t.textSecondary}>Akıcı</span></div>
          <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#ffd700]"></div><span className={t.textSecondary}>Orta</span></div>
          <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#ff4757]"></div><span className={t.textSecondary}>Yoğun</span></div>
        </div>
      </div>

      <div className="relative overflow-hidden" style={{ height: '320px' }}>
        <img
          src="https://readdy.ai/api/search-image?query=Istanbul%20city%20map%20dark%20theme%20satellite%20view%20aerial%20urban%20roads%20highways%20bridges%20bosphorus%20strait%20night%20dark%20background%20minimal&width=700&height=320&seq=mapbg01&orientation=landscape"
          alt="Istanbul Map"
          className={`absolute inset-0 w-full h-full object-cover object-top ${t.isDark ? 'opacity-40' : 'opacity-60'}`}
        />
        <div className={`absolute inset-0 ${t.isDark ? 'bg-[#0f1117]/50' : 'bg-white/30'}`}></div>

        <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          <line x1="50" y1="160" x2="650" y2="160" stroke="#ff4757" strokeWidth="3" strokeOpacity="0.7" />
          <line x1="100" y1="80" x2="600" y2="80" stroke="#ffd700" strokeWidth="2.5" strokeOpacity="0.7" />
          <line x1="80" y1="240" x2="620" y2="240" stroke="#00ff88" strokeWidth="2" strokeOpacity="0.7" />
          <line x1="350" y1="40" x2="350" y2="300" stroke="#ffd700" strokeWidth="2" strokeOpacity="0.5" />
          <line x1="150" y1="200" x2="550" y2="120" stroke="#00d4ff" strokeWidth="2" strokeDasharray="8,4" strokeOpacity="0.8" />
        </svg>

        {cameras.map((cam) => {
          const x = (toX(cam.lng) / W) * 100;
          const y = (toY(cam.lat) / H) * 100;
          const color = densityColors[cam.density as keyof typeof densityColors];
          const isHovered = hoveredCam === cam.id;

          return (
            <div
              key={cam.id}
              className="absolute cursor-pointer transition-transform duration-200"
              style={{ left: `${x}%`, top: `${y}%`, transform: 'translate(-50%, -50%)' }}
              onMouseEnter={() => setHoveredCam(cam.id)}
              onMouseLeave={() => setHoveredCam(null)}
            >
              <div
                className="w-4 h-4 rounded-full border-2 border-white flex items-center justify-center transition-all duration-200"
                style={{ backgroundColor: color, transform: isHovered ? 'scale(1.5)' : 'scale(1)' }}
              >
                <div className="w-1.5 h-1.5 rounded-full bg-white"></div>
              </div>
              <div className="absolute inset-0 rounded-full animate-ping opacity-40" style={{ backgroundColor: color }}></div>
              {isHovered && (
                <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 ${t.cardBg} border ${t.border} rounded-lg px-3 py-2 whitespace-nowrap z-10`}>
                  <div className={`${t.textPrimary} text-xs font-semibold`}>{cam.name}</div>
                  <div className={`${t.textSecondary} text-[10px]`}>{cam.vehicleCount} araç/sa</div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
